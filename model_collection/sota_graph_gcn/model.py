"""
model.py — Spectral Prefetcher using torch_geometric GCNConv

Base architecture (paper Section 4):
    H = σ₂( Ã · σ₁( Ã · X · W⁽⁰⁾ ) · W⁽¹⁾ )

The adjacency matrix returned by build_adjacency_matrix_and_alias is already
row-normalised (transition probabilities), so we use:
    GCNConv(..., add_self_loops=False, normalize=False)
to avoid double-normalising or double-adding self-loops.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module
from torch_geometric.nn import GCNConv
from utils import build_adjacency_matrix_and_alias


# ---------------------------------------------------------------------------
# Helper: dense batched adjacency → sparse COO for torch_geometric
# ---------------------------------------------------------------------------

def batch_dense_adj_to_sparse(A_batch: torch.Tensor):
    """
    Convert a batched dense adjacency tensor to a single COO edge list.

    The SGDP pipeline stacks [in-adj | out-adj] giving shape (B, N, 2N).
    We use only the in-coming block (first N columns) — standard GCN convention.
    The adjacency is already row-normalised, so GCNConv is called without
    add_self_loops or normalize.

    Args:
        A_batch : (B, N, 2N) float tensor, already normalised, on any device.

    Returns:
        edge_index  : (2, E) LongTensor  — on same device as A_batch
        edge_weight : (E,)   FloatTensor — on same device as A_batch
    """
    B, N, two_N = A_batch.shape
    assert two_N == 2 * N
    A_in = A_batch[:, :, :N]  # (B, N, N) — incoming adjacency only

    # Find nonzero entries
    # Using threshold to avoid near-zero float noise from normalization
    mask = A_in > 1e-9                          # (B, N, N)
    indices = mask.nonzero(as_tuple=False)       # (E, 3)
    b_idx, rows, cols = indices[:, 0], indices[:, 1], indices[:, 2]

    # Offset node indices by batch — creates B disconnected subgraphs
    global_rows = rows + b_idx * N
    global_cols = cols + b_idx * N
    edge_index  = torch.stack([global_cols, global_rows], dim=0)  # COO: [src, dst]
    edge_weight = A_in[b_idx, rows, cols]

    return edge_index, edge_weight

# ---------------------------------------------------------------------------
# Spectral Prefetcher
# ---------------------------------------------------------------------------

class SpectralSessionGraph(Module):
    """
    Configurable-depth Spectral GCN + soft-attention scoring head
    (SR-GNN / SGDP style).

    Since the incoming adjacency is already normalised by the SGDP pipeline,
    GCNConv runs with add_self_loops=False and normalize=False to avoid
    corrupting the pre-computed weights.
    """

    def __init__(self, opt, n_node: int):
        super().__init__()
        d                = opt.hiddenSize
        self.hidden_size = d
        self.n_node      = n_node
        self.batch_size  = opt.batchSize
        self.nonhybrid   = opt.nonhybrid
        self.num_layers  = getattr(opt, 'numLayers', 2)
        dropout          = getattr(opt, 'dropout', 0.1)

        if self.num_layers < 1:
            raise ValueError(f"numLayers must be >= 1, got {self.num_layers}")

        self.embedding = nn.Embedding(n_node, opt.window)

        # Pre-normalised adjacency: skip re-normalisation and self-loop insertion
        self.gcn_layers = nn.ModuleList([
            GCNConv(opt.window, d, add_self_loops=False, normalize=False)
        ])
        for _ in range(1, self.num_layers):
            self.gcn_layers.append(
                GCNConv(d, d, add_self_loops=False, normalize=False)
            )
        self.dropout = nn.Dropout(p=dropout)

        # Soft-attention head (identical to SR-GNN / SGDP for fair comparison)
        self.linear_one       = nn.Linear(d,     d, bias=True)
        self.linear_two       = nn.Linear(d,     d, bias=True)
        self.linear_three     = nn.Linear(d,     1, bias=False)
        self.linear_transform = nn.Linear(d * 2, d, bias=True)
        self.linear_final     = nn.Linear(d, opt.window, bias=False)

        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=opt.lr, weight_decay=opt.l2
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc
        )

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for name, w in self.named_parameters():
            if 'gcn' not in name:
                w.data.uniform_(-stdv, stdv)

    def forward(self, items: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        items : (B, N)     — unique node ids per session
        A     : (B, N, 2N) — pre-normalised adjacency (in+out blocks)

        Returns hidden : (B, N, d)
        """
        try:
            B, N = items.shape
        except ValueError: 
            raise ValueError(f"Expected items shape (B, N), got {items.shape}")

        # Embed and flatten: (B, N, d) → (B*N, d)
        x = self.embedding(items).view(B * N, -1)

        # Convert dense adjacency to sparse COO (vectorised, device-safe)
        edge_index, edge_weight = batch_dense_adj_to_sparse(A)

        for layer_idx, gcn in enumerate(self.gcn_layers):
            x = F.relu(gcn(x, edge_index, edge_weight))
            if layer_idx < self.num_layers - 1:
                x = self.dropout(x)

        return x.view(B, N, -1)

    def compute_scores(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        hidden : (B, seq_len, d)
        mask   : (B, seq_len)

        Returns scores : (B, n_node - 1)
        """
        ht = hidden[
            torch.arange(mask.shape[0]).long(),
            torch.sum(mask, 1) - 1
        ]                                                        # (B, d)

        q1    = self.linear_one(ht).unsqueeze(1)                # (B, 1, d)
        q2    = self.linear_two(hidden)                         # (B, seq, d)
        alpha = self.linear_three(torch.sigmoid(q1 + q2))       # (B, seq, 1)

        a = torch.sum(alpha * hidden * mask.unsqueeze(-1).float(), dim=1)  # (B, d)

        if not self.nonhybrid:
            a = self.linear_transform(torch.cat([a, ht], dim=1))
            
        a = self.linear_final(a)                                # (B, 150) -> (B, 32)

        b = self.embedding.weight[1:]                           # (n_node-1, 32)
        return torch.matmul(a, b.T)                             # (B, 32) * (32, n_node-1) -> (B, n_node-1)


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

def trans_to_cuda(variable):
    return variable.cuda() if torch.cuda.is_available() else variable

def trans_to_cpu(variable):
    return variable.cpu() if torch.cuda.is_available() else variable


# ---------------------------------------------------------------------------
# Batch forward pass
# ---------------------------------------------------------------------------

def forward_per_batch(model: SpectralSessionGraph, data):
    alias_inputs, A, items, mask, targets = data

    alias_inputs = trans_to_cuda(torch.tensor(np.array(alias_inputs), dtype=torch.long))
    items        = trans_to_cuda(torch.tensor(np.array(items),        dtype=torch.long))
    A            = trans_to_cuda(torch.tensor(np.array(A),            dtype=torch.float))
    mask         = trans_to_cuda(torch.tensor(np.array(mask),         dtype=torch.long))

    hidden = model(items, A)                                    # (B, N, d)

    # Vectorised gather: alias_inputs is (B, seq) → seq_hidden is (B, seq, d)
    B, seq_len = alias_inputs.shape
    batch_idx  = torch.arange(B, device=hidden.device).unsqueeze(1).expand(-1, seq_len)
    seq_hidden = hidden[batch_idx, alias_inputs]               # (B, seq, d)

    return targets, model.compute_scores(seq_hidden, mask)


def forward(model, i, data, batching=True):
    return forward_per_batch(model, data)


# ---------------------------------------------------------------------------
# Single-inference  (online / streaming)
#
# Signature matches the SGDP run_single_inference convention:
#   call build_adjacency_matrix_and_alias(delta_classes) externally,
#   then pass the three results in.
# ---------------------------------------------------------------------------

def run_single_inference(model: SpectralSessionGraph,
                         alias_input, items, A) -> int:
    """
    alias_input : list of int OR [[int,...]] — length seq_len (or 1 x seq_len)
    items       : list of int OR [[int,...]] — length N (or 1 x N)
    A           : array-like, shape (N, 2N) OR (1, N, 2N)
    """
    def ensure_1d(x):
        arr = np.array(x)
        return arr.reshape(-1)          # always (K,)

    def ensure_2d(x):
        arr = np.array(x)
        if arr.ndim == 3:               # (1, N, 2N) → (N, 2N)
            arr = arr.squeeze(0)
        return arr                      # always (N, 2N)

    ai = trans_to_cuda(torch.tensor(ensure_1d(alias_input), dtype=torch.long).unsqueeze(0))   # (1, seq)
    it = trans_to_cuda(torch.tensor(ensure_1d(items),       dtype=torch.long).unsqueeze(0))   # (1, N)
    a  = trans_to_cuda(torch.tensor(ensure_2d(A),           dtype=torch.float).unsqueeze(0))  # (1, N, 2N)
    mask = torch.ones_like(ai)

    hidden     = model(it, a)                                          # (1, N, d)
    seq_hidden = hidden[0][ai[0]].unsqueeze(0)                         # (1, seq, d)

    scores = model.compute_scores(seq_hidden, mask)                    # (1, n_node-1)
    return scores.argmax(dim=1).item()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def training(model: SpectralSessionGraph, train_data_list, train_slices):
    from tqdm import tqdm
    model.train()
    total_loss = 0.0

    for train_data, _, j in tqdm(
        zip(train_data_list, train_slices, np.arange(len(train_slices)))
    ):
        model.optimizer.zero_grad()
        targets, scores = forward(model, None, train_data)
        targets = trans_to_cuda(torch.tensor(targets, dtype=torch.long))
        loss = model.loss_function(scores, targets)
        loss.backward()
        model.optimizer.step()
        total_loss += loss.item()

    print(f"\tTotal loss: {total_loss:.3f}")
    return model


# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------

def testing_per_batch(model: SpectralSessionGraph,
                      test_data_list, top_k: int = 1):
    model.scheduler.step()
    model.eval()
    arr_raw_pred = []
    with torch.no_grad():
        for test_data in test_data_list:
            _, scores  = forward(model, 0, test_data)
            predicted  = scores.topk(top_k)[1]
            arr_raw_pred += trans_to_cpu(predicted).detach().numpy().flatten().tolist()
    return arr_raw_pred


def testing_per_inference(model: SpectralSessionGraph, test_data_list):
    """
    test_data_list : list of (alias_input, A, items, _, _) tuples
                     as built by build_feature_from_test_data.
    """
    model.scheduler.step()
    model.eval()
    arr_raw_pred = []
    with torch.no_grad():
        for alias_input, A, items, _, _ in test_data_list:
            pred = run_single_inference(model, alias_input, items, A)
            arr_raw_pred.append(pred)
    return arr_raw_pred


def train_test_pred(model, train_data_list, train_slices, test_data_list,
                    top_k: int = 20, batching: bool = True):
    model = training(model, train_data_list, train_slices)
    if batching:
        return testing_per_batch(model, test_data_list, top_k=1)
    else:
        return testing_per_inference(model, test_data_list)
