import math
import numpy as np
import torch
from torch import nn
from torch.nn import Module, Parameter
from utils import Data, split_validation, build_adjacency_matrix_and_alias
import torch.nn.functional as F
import os
import time
from tqdm import tqdm


# ---------------------------------------------------------------------------
# GNN layer (unchanged from paper)
# ---------------------------------------------------------------------------

class GNN(Module):
    def __init__(self, hidden_size, step=1):
        super(GNN, self).__init__()
        self.step        = step
        self.hidden_size = hidden_size
        self.input_size  = hidden_size * 2
        self.gate_size   = 3 * hidden_size

        self.w_ih  = Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh  = Parameter(torch.Tensor(self.gate_size, self.hidden_size))
        self.b_ih  = Parameter(torch.Tensor(self.gate_size))
        self.b_hh  = Parameter(torch.Tensor(self.gate_size))
        self.b_iah = Parameter(torch.Tensor(self.hidden_size))
        self.b_oah = Parameter(torch.Tensor(self.hidden_size))

        self.linear_edge_in  = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_out = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_f   = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

    def GNNCell(self, A, hidden):
        input_in  = torch.matmul(A[:, :, :A.shape[1]],
                                 self.linear_edge_in(hidden)) + self.b_iah
        input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]],
                                 self.linear_edge_out(hidden)) + self.b_oah
        inputs = torch.cat([input_in, input_out], 2)
        gi = F.linear(inputs, self.w_ih, self.b_ih)
        gh = F.linear(hidden, self.w_hh, self.b_hh)
        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate   = torch.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (hidden - newgate)
        return hy

    def forward(self, A, hidden):
        for _ in range(self.step):
            hidden = self.GNNCell(A, hidden)
        return hidden


# ---------------------------------------------------------------------------
# Session graph model (unchanged from paper)
# ---------------------------------------------------------------------------

class SessionGraph(Module):
    def __init__(self, opt, n_node):
        super(SessionGraph, self).__init__()
        self.hidden_size = opt.hiddenSize
        self.n_node      = n_node
        self.batch_size  = opt.batchSize
        self.nonhybrid   = opt.nonhybrid

        self.embedding        = nn.Embedding(self.n_node, self.hidden_size)
        self.gnn              = GNN(self.hidden_size, step=opt.step)
        self.linear_one       = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two       = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three     = nn.Linear(self.hidden_size, 1,               bias=False)
        self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)

        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer     = torch.optim.Adam(self.parameters(),
                                              lr=opt.lr, weight_decay=opt.l2)
        self.scheduler     = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def compute_scores(self, hidden, mask):
        ht = hidden[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]
        q1    = self.linear_one(ht).view(ht.shape[0], 1, ht.shape[1])
        q2    = self.linear_two(hidden)
        alpha = self.linear_three(torch.sigmoid(q1 + q2))
        a = torch.sum(alpha * hidden * mask.view(mask.shape[0], -1, 1).float(), 1)
        if not self.nonhybrid:
            a = self.linear_transform(torch.cat([a, ht], 1))
        b = self.embedding.weight[1:]
        return torch.matmul(a, b.transpose(1, 0))

    def forward(self, inputs, A):
        hidden = self.embedding(inputs)
        hidden = self.gnn(A, hidden)
        return hidden


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

def trans_to_cuda(variable):
    return variable.cuda() if torch.cuda.is_available() else variable

def trans_to_cpu(variable):
    return variable.cpu() if torch.cuda.is_available() else variable


# ---------------------------------------------------------------------------
# Batch forward — uses np.array() for safe tensor conversion (user's fix)
# ---------------------------------------------------------------------------

def forward_per_batch(model, data):
    alias_inputs, A, items, mask, targets = data
    alias_inputs = trans_to_cuda(torch.tensor(np.array(alias_inputs), dtype=torch.long))
    items        = trans_to_cuda(torch.tensor(np.array(items),        dtype=torch.long))
    A            = trans_to_cuda(torch.tensor(np.array(A),            dtype=torch.float))
    mask         = trans_to_cuda(torch.tensor(np.array(mask),         dtype=torch.long))

    hidden = trans_to_cuda(model(items, A))

    get = lambda i: hidden[i][alias_inputs[i]]
    seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])
    return targets, model.compute_scores(seq_hidden, mask)


def forward(model, i, data, batching=True):
    return forward_per_batch(model, data)


# ---------------------------------------------------------------------------
# Single-sample inference (user's addition — for online / streaming use)
# ---------------------------------------------------------------------------

def run_single_inference(model, historical_deltas):
    """
    historical_deltas : list of delta-class ints (length window_size)
    Returns the predicted class index (int).
    """
    [input_feature], A, items = build_adjacency_matrix_and_alias(historical_deltas)

    mask         = [[1] * len(input_feature)]
    input_feature = trans_to_cuda(torch.Tensor([input_feature]).long())
    items         = trans_to_cuda(torch.Tensor(items).long())
    A             = trans_to_cuda(torch.Tensor(A).float())
    mask          = trans_to_cuda(torch.Tensor(mask).long())

    hidden = trans_to_cuda(model(items, A))
    get = lambda i: hidden[i][input_feature[i]]
    seq_hidden = torch.stack([get(i) for i in torch.arange(len(input_feature)).long()])

    raw_scores      = model.compute_scores(seq_hidden, mask)
    predicted_class = raw_scores.topk(1)[1]
    return trans_to_cpu(predicted_class).detach().tolist()[0][0]


# ---------------------------------------------------------------------------
# Training — one epoch
# ---------------------------------------------------------------------------

def training(model, train_data_list, train_slices):
    model.train()
    total_loss = 0.0
    for train_data, _, j in tqdm(
        zip(train_data_list, train_slices, np.arange(len(train_slices)))
    ):
        model.optimizer.zero_grad()
        targets, scores = forward(model, None, train_data)
        targets = trans_to_cuda(torch.Tensor(targets).long())
        loss = model.loss_function(scores, targets)
        loss.backward()
        model.optimizer.step()
        total_loss += loss.item()
    print(f'\tTotal loss: {total_loss:.3f}')
    return model


# ---------------------------------------------------------------------------
# Batched testing — returns (all_pred_topk, all_targets)
#
# CHANGED from user's flat-list version to match original paper format:
#   all_pred_topk : list of (top_k,) numpy int arrays
#   all_targets   : list of ints
# This lets main.py pass results directly to both score_compute() and
# single_cache_test_multi() without reshaping.
# ---------------------------------------------------------------------------

def testing_per_batch(model, test_data_list, top_k=20):
    """
    Returns:
        all_pred   : list of numpy arrays, shape (top_k,) per sample
        all_targets: list of ints
    """
    model.scheduler.step()
    model.eval()
    all_pred    = []
    all_targets = []
    with torch.no_grad():
        for test_data in test_data_list:
            targets, scores = forward(model, 0, test_data)
            sub_scores = scores.topk(top_k)[1]               # (batch, top_k)
            sub_scores = trans_to_cpu(sub_scores).detach().numpy()
            all_pred.extend(sub_scores)                       # list of (top_k,) arrays
            all_targets.extend(targets)
    return all_pred, all_targets


# ---------------------------------------------------------------------------
# Per-inference testing (user's addition — used for final timed evaluation)
# Returns flat list of ints, NOT (pred, targets), so timing stays accurate.
# ---------------------------------------------------------------------------

def testing_per_inference(model, test_data_list):
    """
    test_data_list : list of (alias_input, A, items, _, _) tuples
    Returns flat list of predicted class ints.
    """
    model.scheduler.step()
    model.eval()
    arr_raw_pred = []
    with torch.no_grad():
        for alias_input, A, items, _, _ in test_data_list:
            pred = run_single_inference(model, alias_input)
            arr_raw_pred.append(pred)
    return arr_raw_pred


# ---------------------------------------------------------------------------
# Combined train + test (one epoch)
# ---------------------------------------------------------------------------

def train_test_pred(model, train_data_list, train_slices, test_data_list,
                    top_k=20, batching=True):
    """
    Trains one epoch, then evaluates.
    batching=True  → returns (all_pred_topk, all_targets) for score_compute
    batching=False → returns flat pred list for per-inference timing
    """
    model = training(model, train_data_list, train_slices)
    if batching:
        return testing_per_batch(model, test_data_list, top_k=top_k)
    else:
        return testing_per_inference(model, test_data_list)