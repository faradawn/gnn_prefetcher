
import datetime
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


class GNN(Module):
    def __init__(self, hidden_size, step=1):
        super(GNN, self).__init__()
        self.step = step
        self.hidden_size = hidden_size
        self.input_size = hidden_size * 2
        self.gate_size = 3 * hidden_size
        self.w_ih = Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = Parameter(torch.Tensor(self.gate_size, self.hidden_size))
        self.b_ih = Parameter(torch.Tensor(self.gate_size))
        self.b_hh = Parameter(torch.Tensor(self.gate_size))
        self.b_iah = Parameter(torch.Tensor(self.hidden_size))
        self.b_oah = Parameter(torch.Tensor(self.hidden_size))

        self.linear_edge_in = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_out = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_f = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

    def GNNCell(self, A, hidden): #A is adjacency matrices
        input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah #outgoing edges and incoming edges
        input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
        inputs = torch.cat([input_in, input_out], 2)
        gi = F.linear(inputs, self.w_ih, self.b_ih)
        gh = F.linear(hidden, self.w_hh, self.b_hh)
        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (hidden - newgate)
        return hy

    def forward(self, A, hidden):
        for i in range(self.step):
            hidden = self.GNNCell(A, hidden)
        return hidden


class SessionGraph(Module):
    def __init__(self, opt, n_node):
        super(SessionGraph, self).__init__()
        self.hidden_size = opt.hiddenSize
        self.n_node = n_node
        self.batch_size = opt.batchSize
        self.nonhybrid = opt.nonhybrid
        self.embedding = nn.Embedding(self.n_node, self.hidden_size)
        self.gnn = GNN(self.hidden_size, step=opt.step)
        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three = nn.Linear(self.hidden_size, 1, bias=False)
        self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def compute_scores(self, hidden, mask):
        ht = hidden[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]  # batch_size x latent_size
        q1 = self.linear_one(ht).view(ht.shape[0], 1, ht.shape[1])  # batch_size x 1 x latent_size
        q2 = self.linear_two(hidden)  # batch_size x seq_length x latent_size
        alpha = self.linear_three(torch.sigmoid(q1 + q2))
        a = torch.sum(alpha * hidden * mask.view(mask.shape[0], -1, 1).float(), 1)
        if not self.nonhybrid:
            a = self.linear_transform(torch.cat([a, ht], 1))
        b = self.embedding.weight[1:]  # n_nodes x latent_size
        scores = torch.matmul(a, b.transpose(1, 0))
        return scores

    def forward(self, inputs, A):
        hidden = self.embedding(inputs)
        hidden = self.gnn(A, hidden)
        return hidden


def trans_to_cuda(variable):
# torch.cuda.is_available()
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable
 
def training(model,train_data_list,train_slices):
    model.train()
    total_loss = 0.0

    # Training
    for train_data, i, j in tqdm(zip(train_data_list,train_slices, np.arange(len(train_slices)))):
        model.optimizer.zero_grad()
        # print("train_data ", train_data[0][0])
        # return None
        targets, scores = forward(model, i, train_data)
        targets = trans_to_cuda(torch.Tensor(targets).long())
        loss = model.loss_function(scores, targets)
        loss.backward()
        model.optimizer.step()
        # exit(0)
        total_loss += loss
        # print('[%d/%d] Loss: %.4f' % (j, len(train_slices), loss.item()),end='\r')
        # if j % int(len(train_slices) / 5 + 1) == 0:
            # print('[%d/%d] Loss: %.4f' % (j, len(train_slices), loss.item()))
    print('\tTotal loss: %.3f' % total_loss)
    
    return model

def forward(model, i, data, batching = True):
    return forward_per_batch(model, i, data)
        
def forward_per_batch(model, i, data):
    input_features, A, items, mask, targets = data
    # alias input is the delta that have been normalized 
        # INPUT[0] [ 11 441 173 442 173  11  15 443 173  11  15  33 444 173  11 445 173  11
        # 446 173  11 447 173  11  15 448 173  11 449 173  11 450]
        # NODE 0 [ 11  15  33 173 441 442 443 444 445 446 447 448 449 450]

        # input_features [[0, 4, 3, 5, 3, 0, 1, 6, 3, 0, 1, 2, 7, 3, 0, 8, 3, 0, 9, 3, 0, 10, 3, 0, 1, 11, 3, 0, 12, 3, 0, 13]]
    input_features = trans_to_cuda(torch.Tensor(input_features).long())
    items = trans_to_cuda(torch.Tensor(items).long())
    A = trans_to_cuda(torch.Tensor(A).float())
    mask = trans_to_cuda(torch.Tensor(mask).long())
    hidden = trans_to_cuda(model(items, A)) # creating embedding

    get = lambda i: hidden[i][input_features[i]]
    seq_hidden = torch.stack([get(i) for i in torch.arange(len(input_features)).long()])
    return targets, model.compute_scores(seq_hidden, mask)
    
def run_single_inference(model, historical_deltas):
    [input_feature], A, items = build_adjacency_matrix_and_alias(historical_deltas)
 
    mask = [[1] * len(input_feature)]
    input_feature = trans_to_cuda(torch.Tensor([input_feature]).long())
    items = trans_to_cuda(torch.Tensor(items).long())
    A = trans_to_cuda(torch.Tensor(A).float())
    mask = trans_to_cuda(torch.Tensor(mask).long())
    hidden = trans_to_cuda(model(items, A)) # creating embedding

    get = lambda i: hidden[i][input_feature[i]]
    seq_hidden = torch.stack([get(i) for i in torch.arange(len(input_feature)).long()])
    # for every single inference, it outputs 1000 scores (representing 1000 classes)
    raw_1000_scores = model.compute_scores(seq_hidden, mask)
    # print("     raw_1000_scores ", raw_1000_scores.shape)
    # we want to get the class with the highest scores
    predicted_class = raw_1000_scores.topk(1)[1] # get the indexes of the top k scores
    predicted_class = trans_to_cpu(predicted_class).detach().tolist()
    return predicted_class[0][0]


def run_single_inference_old_ver(model, input_feature, A, items):

    # alias input is the delta that have been normalized 
        # INPUT[0] [ 11 441 173 442 173  11  15 443 173  11  15  33 444 173  11 445 173  11
        # 446 173  11 447 173  11  15 448 173  11 449 173  11 450]
        # NODE 0 [ 11  15  33 173 441 442 443 444 445 446 447 448 449 450]
        # input_feature [[0, 4, 3, 5, 3, 0, 1, 6, 3, 0, 1, 2, 7, 3, 0, 8, 3, 0, 9, 3, 0, 10, 3, 0, 1, 11, 3, 0, 12, 3, 0, 13]]
    # print(input_features)
    mask = [[1] * len(input_feature)]
    input_feature = trans_to_cuda(torch.Tensor([input_feature]).long())
    # print("     mask " , mask)
    # exit(0)
        #  mask is always [array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])]
    # exit(0)
    items = trans_to_cuda(torch.Tensor(items).long())
    A = trans_to_cuda(torch.Tensor(A).float())
    mask = trans_to_cuda(torch.Tensor(mask).long())
    hidden = trans_to_cuda(model(items, A)) # creating embedding

    get = lambda i: hidden[i][input_feature[i]]
    seq_hidden = torch.stack([get(i) for i in torch.arange(len(input_feature)).long()])
    # for every single inference, it outputs 1000 scores (representing 1000 classes)
    raw_1000_scores = model.compute_scores(seq_hidden, mask)
    # print("     raw_1000_scores ", raw_1000_scores.shape)
    # we want to get the class with the highest scores
    predicted_class = raw_1000_scores.topk(1)[1] # get the indexes of the top k scores
    predicted_class = trans_to_cpu(predicted_class).detach().tolist()
    return predicted_class[0][0]
   
def testing_per_inference(model,test_data_list, arr_hitorical_deltas):
    model.scheduler.step()
    model.eval()

    print("Do per inference testing:")
    print("N Batch test data list ", len(test_data_list))
    print( "\n   Predicting the delta of ", len(test_data_list), " test cases")
    with torch.no_grad():
        arr_raw_pred = []
        for idx, per_inference_test_data in enumerate(test_data_list):
            # Submit each inference to the model
            predicted_class = run_single_inference(model, arr_hitorical_deltas[idx])
            # [input_feature], A, items, _, _ = per_inference_test_data
            # predicted_class = run_single_inference_old_ver(model, input_feature, A, items)
            arr_raw_pred.append(predicted_class)
    return arr_raw_pred

# The original SGDP version
def testing_per_batch(model,test_data_list,top_k = 1):
    model.scheduler.step()
    model.eval()

    # Testing (on the last 10% of the data)
    print("N Batch test data list ", len(test_data_list))
    with torch.no_grad():
        arr_raw_pred=[]
        for test_data in zip(test_data_list):
            print( "   Predicting the delta (per 128 size batch)", len(test_data), len(test_data[0]))
            _, scores = forward(model, 0, test_data)
            predicted_classes = scores.topk(top_k)[1] # get the indexes of the top k scores
            predicted_classes = trans_to_cpu(predicted_classes).detach().numpy()
            # flatten out 2d array into 1d array
            predicted_classes = predicted_classes.flatten()
            arr_raw_pred += predicted_classes.tolist()
    return arr_raw_pred


def train_test_pred(model,train_data_list,train_slices,test_data_list,top_k = 20, batching = True):
    # Training the model
    model = training(model,train_data_list,train_slices)
    # Testing (on the last 10% of the data)
    if batching:
        return testing_per_batch(model,test_data_list,top_k = 1)
    else:
        # for flashnet online inference
        return testing_per_inference(model,test_data_list)
