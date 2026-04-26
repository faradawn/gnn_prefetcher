import networkx as nx
import numpy as np


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
        # print("INPUT[0]", inputs[0])
        # Q: What is A? graph matrix
        # Q: What is node? storing the delta info
        items, n_node, A, alias_inputs = [], [], [], []
        for u_input in inputs:
            n_node.append(len(np.unique(u_input)))
        max_n_node = np.max(n_node)
        for u_input in inputs:
            node = np.unique(u_input)
            items.append(node.tolist() + (max_n_node - len(node)) * [0])
            node_to_idx = {v: k for k, v in enumerate(node)}

            valid_len = len(u_input)
            for k in range(1, len(u_input)):
                if u_input[k] == 0:
                    valid_len = k
                    break

            indices = np.array([node_to_idx[x] for x in u_input[:valid_len]])

            u_A = np.zeros((max_n_node, max_n_node))
            if valid_len > 1:
                ii, jj = np.triu_indices(valid_len, k=1)
                u_A[indices[ii], indices[jj]] = 1
                src, dst = indices[:-1], indices[1:]
                coords = src * max_n_node + dst
                u_A += np.bincount(coords, minlength=max_n_node * max_n_node).reshape(max_n_node, max_n_node)

            u_sum_in = np.sum(u_A, 0)
            u_sum_in[np.where(u_sum_in == 0)] = 1
            u_A_in = np.divide(u_A, u_sum_in)
            u_sum_out = np.sum(u_A, 1)
            u_sum_out[np.where(u_sum_out == 0)] = 1
            u_A_out = np.divide(u_A.transpose(), u_sum_out)
            u_A = np.concatenate([u_A_in, u_A_out]).transpose()
            A.append(u_A)
            alias_inputs.append([node_to_idx[x] for x in u_input])

        # print(alias_inputs[0])
        # exit(0)
        return alias_inputs, A, items, mask, targets

    def get_one_slice(self, i):
        historical_deltas = self.inputs[i][0]
        node = np.unique(historical_deltas)
        n = len(node)
        node_to_idx = {v: k for k, v in enumerate(node)}

        valid_len = len(historical_deltas)
        for k in range(1, len(historical_deltas)):
            if historical_deltas[k] == 0:
                valid_len = k
                break

        indices = np.array([node_to_idx[x] for x in historical_deltas[:valid_len]])

        u_A = np.zeros((n, n))
        if valid_len > 1:
            ii, jj = np.triu_indices(valid_len, k=1)
            u_A[indices[ii], indices[jj]] = 1
            src, dst = indices[:-1], indices[1:]
            coords = src * n + dst
            u_A += np.bincount(coords, minlength=n * n).reshape(n, n)

        u_sum_in = np.sum(u_A, 0)
        u_sum_in[np.where(u_sum_in == 0)] = 1
        u_A_in = np.divide(u_A, u_sum_in)
        u_sum_out = np.sum(u_A, 1)
        u_sum_out[np.where(u_sum_out == 0)] = 1
        u_A_out = np.divide(u_A.transpose(), u_sum_out)
        u_A = np.concatenate([u_A_in, u_A_out]).transpose()
        return [[node_to_idx[x] for x in historical_deltas]], [u_A], [node.tolist()]

def build_adjacency_matrix_and_alias(historical_deltas):
    if type(historical_deltas) == list:
        historical_deltas = np.array(historical_deltas)
    node = np.unique(historical_deltas)
    n = len(node)
    node_to_idx = {v: k for k, v in enumerate(node)}

    valid_len = len(historical_deltas)
    for k in range(1, len(historical_deltas)):
        if historical_deltas[k] == 0:
            valid_len = k
            break

    indices = np.array([node_to_idx[x] for x in historical_deltas[:valid_len]])

    u_A = np.zeros((n, n))
    if valid_len > 1:
        ii, jj = np.triu_indices(valid_len, k=1)
        u_A[indices[ii], indices[jj]] = 1
        src, dst = indices[:-1], indices[1:]
        coords = src * n + dst
        u_A += np.bincount(coords, minlength=n * n).reshape(n, n)

    u_sum_in = np.sum(u_A, 0)
    u_sum_in[np.where(u_sum_in == 0)] = 1
    u_A_in = np.divide(u_A, u_sum_in)
    u_sum_out = np.sum(u_A, 1)
    u_sum_out[np.where(u_sum_out == 0)] = 1
    u_A_out = np.divide(u_A.transpose(), u_sum_out)
    u_A = np.concatenate([u_A_in, u_A_out]).transpose()
    return [[node_to_idx[x] for x in historical_deltas]], [u_A], [node.tolist()]
    

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