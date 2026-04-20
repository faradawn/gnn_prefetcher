import networkx as nx
import numpy as np


# ---------------------------------------------------------------------------
# Shared adjacency builder — used by get_slice, get_one_slice, and
# build_adjacency_matrix_and_alias.
#
# Implements the hybrid connection matrix M_h from the SGDP paper:
#   M_S  — sequential connect matrix (Eq. 5): edges between consecutive deltas
#   M_F  — full-connect matrix (Eq. 6): every delta connected to every later
#           delta in the stream, with edge weight 1/|b-a| (inverse distance)
#   M_h  — normalised M_S and M_F are combined with equal weighting (0.5 each)
#           and concatenated as [in | out] per the SR-GNN convention.
#
# Bug fixes vs. original implementation:
#   1. Full-connect inner loop used len(node) (# unique nodes) as the upper
#      bound instead of len(u_input) (sequence length), so most future
#      positions were silently skipped when the sequence was longer than the
#      vocabulary.
#   2. Full-connect edges were unweighted (=1) — paper Eq. 6 specifies
#      weight 1/|b-a|, so closer pairs get stronger connections.
#   3. M_S and M_F were added into a single matrix before normalisation, so
#      the in/out sums mixed sequential counts with full-connect weights.
#      They are now normalised independently before the weighted sum.
# ---------------------------------------------------------------------------

def _build_hybrid_A(u_input, node):
    """
    Parameters
    ----------
    u_input : array-like of int, shape (seq_len,)
        Encoded delta-class sequence (may contain trailing 0-padding;
        processing stops at the first 0 after position 0).
    node : array-like of int
        Sorted unique non-zero values in u_input (the per-sample vocabulary).

    Returns
    -------
    u_A : np.ndarray, shape (len(node), 2 * len(node))
        Hybrid [M_h_in | M_h_out] adjacency block.
    alias : list of int
        Index of each u_input[i] within `node`.
    """
    n       = len(node)
    seq_len = len(u_input)

    # ---- Sequential connect matrix M_S (Eq. 5) ----------------------------
    A_seq = np.zeros((n, n))
    for i in range(seq_len - 1):
        if u_input[i + 1] == 0:
            break
        u = int(np.where(node == u_input[i])[0][0])
        v = int(np.where(node == u_input[i + 1])[0][0])
        A_seq[u][v] += 1

    col_sum = A_seq.sum(axis=0); col_sum[col_sum == 0] = 1
    A_seq_in  = A_seq / col_sum
    row_sum = A_seq.sum(axis=1); row_sum[row_sum == 0] = 1
    A_seq_out = (A_seq.T / row_sum)

    # ---- Full-connect matrix M_F (Eq. 6, 1/distance weighting) ------------
    # Bug fix: range must use seq_len (not n) so all future positions are
    # visited when the window is longer than the number of unique nodes.
    A_full = np.zeros((n, n))
    for i in range(seq_len - 1):
        if u_input[i + 1] == 0:
            break
        u = int(np.where(node == u_input[i])[0][0])
        for dist in range(1, seq_len - i - 1):
            if u_input[i + dist] == 0:
                break
            v = int(np.where(node == u_input[i + dist])[0][0])
            A_full[u][v] += 1.0 / dist   # inverse-distance per Eq. 6

    col_sum = A_full.sum(axis=0); col_sum[col_sum == 0] = 1
    A_full_in  = A_full / col_sum
    row_sum = A_full.sum(axis=1); row_sum[row_sum == 0] = 1
    A_full_out = (A_full.T / row_sum)

    # ---- Hybrid M_h: equal-weight sum then [in | out] concat ---------------
    A_in  = 0.5 * A_seq_in  + 0.5 * A_full_in
    A_out = 0.5 * A_seq_out + 0.5 * A_full_out
    u_A   = np.concatenate([A_in, A_out]).T   # shape (n, 2n)

    alias = [int(np.where(node == x)[0][0]) for x in u_input]
    return u_A, alias


def build_graph(train_data):
    graph = nx.DiGraph()
    for seq in train_data:
        for i in range(len(seq) - 1):
            if graph.get_edge_data(seq[i], seq[i + 1]) is None:
                weight = 1
            else:
                weight = graph.get_edge_data(seq[i], seq[i + 1])['weight'] + 1
            graph.add_edge(seq[i], seq[i + 1], weight=weight)
    for node in graph.nodes:
        sum = 0
        for j, i in graph.in_edges(node):
            sum += graph.get_edge_data(j, i)['weight']
        if sum != 0:
            for j, i in graph.in_edges(i):
                graph.add_edge(j, i, weight=graph.get_edge_data(j, i)['weight'] / sum)
    return graph


def data_masks(all_usr_pois, item_tail):
    us_lens = [len(upois) for upois in all_usr_pois]
    len_max = max(us_lens)
    us_pois = [upois + item_tail * (len_max - le) for upois, le in zip(all_usr_pois, us_lens)]
    us_msks = [[1] * le + [0] * (len_max - le) for le in us_lens]
    return us_pois, us_msks, len_max


def split_validation(train_set, valid_portion):
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = np.arange(n_samples, dtype='int32')
    np.random.shuffle(sidx)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    return (train_set_x, train_set_y), (valid_set_x, valid_set_y)


class Data():
    def __init__(self, data, shuffle=False, graph=None):
        inputs = data[0]
        inputs, mask, len_max = data_masks(inputs, [0])
        self.inputs = np.asarray(inputs)
        self.mask = np.asarray(mask)
        self.len_max = len_max
        self.targets = np.asarray(data[1])
        self.length = len(inputs)
        self.shuffle = shuffle
        self.graph = graph

    def get_data_as_list(self):
        return self.inputs.tolist()

    def generate_batch(self, batch_size):
        if self.shuffle:
            shuffled_arg = np.arange(self.length)
            np.random.shuffle(shuffled_arg)
            self.inputs = self.inputs[shuffled_arg]
            self.mask = self.mask[shuffled_arg]
            self.targets = self.targets[shuffled_arg]
        n_batch = int(self.length / batch_size)
        if self.length % batch_size != 0:
            n_batch += 1
        slices = np.split(np.arange(n_batch * batch_size), n_batch)
        slices[-1] = slices[-1][:(self.length - batch_size * (n_batch - 1))]
        return slices

    def get_slice(self, i):
        inputs, mask, targets = self.inputs[i], self.mask[i], self.targets[i]
        items, n_node, A, alias_inputs = [], [], [], []
        for u_input in inputs:
            n_node.append(len(np.unique(u_input)))
        max_n_node = np.max(n_node)
        for u_input in inputs:
            node = np.unique(u_input)
            items.append(node.tolist() + (max_n_node - len(node)) * [0])
            # Pad u_input to max_n_node so all samples have the same alias length
            u_input_padded = np.concatenate(
                [u_input, np.zeros(max_n_node - len(u_input), dtype=u_input.dtype)]
            ) if len(u_input) < max_n_node else u_input

            # Build hybrid adjacency using the shared corrected helper
            u_A_raw, alias = _build_hybrid_A(u_input_padded, node)
            # Pad adjacency to (max_n_node, 2*max_n_node) for batching
            pad_rows = max_n_node - u_A_raw.shape[0]
            pad_cols = 2 * max_n_node - u_A_raw.shape[1]
            u_A_padded = np.pad(u_A_raw, ((0, pad_rows), (0, pad_cols)))
            A.append(u_A_padded)
            # Pad alias to max_n_node length
            alias_padded = alias + [0] * (max_n_node - len(alias))
            alias_inputs.append(alias_padded)
        return alias_inputs, A, items, mask, targets

    def get_one_slice(self, i):
        historical_deltas = self.inputs[i][0]
        node = np.unique(historical_deltas)
        u_A, alias = _build_hybrid_A(historical_deltas, node)
        return [alias], [u_A], [node.tolist()]

def build_adjacency_matrix_and_alias(historical_deltas):
    if not isinstance(historical_deltas, np.ndarray):
        historical_deltas = np.array(historical_deltas)
    node = np.unique(historical_deltas)
    u_A, alias = _build_hybrid_A(historical_deltas, node)
    return [alias], [u_A], [node.tolist()]
    

def topn_test():
    pred = np.load('pred_result.npy')
    target = np.load('target_result.npy')
    print(pred.shape,target.shape)
    hit_topn = [0 for i in range(1,np.shape(pred)[1]+2)]
    mrr_topn = [0 for i in range(1,np.shape(pred)[1]+2)]

    for i in (range(1,np.shape(pred)[1]+1)):
        pred_temp = pred[:,:i]
        hit = []
        mrr = []
        for j in range(np.shape(pred_temp)[0]):
            hit.append(np.isin(target[j],pred_temp[j]))
            if len(np.where(pred_temp[j] == target[j])[0]) == 0:
                mrr.append(0)
            else:
                mrr.append(1 / (np.where(pred_temp[j] == target[j])[0][0] + 1))
        hit_topn[i] = np.mean(hit) * 100
        mrr_topn[i] = np.mean(mrr) * 100
        print(np.mean(hit) * 100,np.mean(mrr) * 100)
    print(hit_topn,mrr_topn)