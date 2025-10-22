#!/usr/bin/env python36
# -*- coding: utf-8 -*-
import argparse
import pickle
import time
import os
import torch
import pandas as pd
import numpy as np


from utils import Data, split_validation
from model import *
from tqdm import tqdm
from collections import Counter
from cache import *

class Options:
    def __init__(self):
        self.batchSize = 128
        self.hiddenSize = 150
        self.epoch = 3
        self.lr = 0.001
        self.lr_dc = 0.1
        self.lr_dc_step = 3
        self.l2 = 1e-5
        self.step = 1
        self.patience = 3
        self.nonhybrid = False
        self.validation = False
        self.valid_portion = 0.1
        self.draw_graph = False
        self.see_ori_dataset = False
        self.topn = 20
        self.dataset_percent = 1.0
        self.window = 32
        self.topnum = 1000

opt = Options()

def my_dict(train_trace, top_num=1000):
    train_trace['ByteOffset_Delta'] = train_trace['ByteOffset'] - \
        train_trace['ByteOffset'].shift(-1)
    train_trace['ByteOffset_Delta'] = train_trace['ByteOffset_Delta'].fillna(0)

    x = Counter(train_trace['ByteOffset_Delta'])
    vals = x.most_common(top_num)
    top_deltas = {}
    rev_map = {}
    for i, tup in enumerate(vals):
        top_deltas[vals[0]] = i
        rev_map[i] = vals[0] # i -> raw delta
    
    forward_map = {}
    count = 0
    while (count < len(train_trace)):
        x = train_trace['ByteOffset_Delta'].iloc[count]
        if x in top_deltas:
            forward_map[x] = top_deltas[x]
    
    return forward_map, rev_map

def dict_generate(train_trace, top_num=1000):
    train_trace['ByteOffset_Delta'] = train_trace['ByteOffset'] - \
        train_trace['ByteOffset'].shift(-1)
    train_trace['ByteOffset_Delta'] = train_trace['ByteOffset_Delta'].fillna(0)

    a = train_trace['ByteOffset_Delta'].astype(int).unique().tolist()

    operation_id_map = {}
    for i, id in enumerate(a):
        operation_id_map[id] = i
    train_trace['ByteOffset_Delta_class'] = train_trace['ByteOffset_Delta'].map(
        lambda x: operation_id_map[x])

    x = Counter(train_trace['ByteOffset_Delta_class'])
    vals = {}
    vals = x.most_common(top_num)

    bo_list = []

    for x in vals:
        bo_list.append(x[0])

    count = 0
    label_list = []
    while (count < len(train_trace)):
        x = train_trace['ByteOffset_Delta_class'].iloc[count]
        if x in bo_list:
            label_list.append(x)
        else:
            label_list.append(999999)  # no Prefetch class
        count = count + 1

    train_trace['ByteOffset_Delta_class'] = label_list
    a = train_trace['ByteOffset_Delta_class'].unique().tolist()
    bo_map = {}
    for i, id in enumerate(a):
        bo_map[id] = i
    operation_id_map_div = {v: k for k, v in operation_id_map.items()}
    operation_id_map_div[999999] = 0
    bo_map_div = {v: k for k, v in bo_map.items()}

    return bo_map, bo_map_div, operation_id_map, operation_id_map_div


def trace2input(dicts, trace, window_size=32):
    bo_map, _, operation_id_map, _ = dicts
    # print(len(trace))

    keys = bo_map.keys()
    inputs = []
    targets = []
    for i in range(len(trace)-window_size-1):

        input_single = []
        for j in range(i, i+window_size+1):
            diff = int(trace[j]-trace[j+1])
            if operation_id_map[diff] in keys:
                input_single.append(bo_map[operation_id_map[diff]]+1)###
            else:
                input_single.append(bo_map[999999]+1)###
        inputs.append(input_single[:-1])
        targets.append(input_single[-1])
    return inputs, targets


def dataset2input(dataset, window_size=32, method='top', top_num=1000):
    if method == 'top':
        names = ['TimeStamp', 'ByteOffset']
        # For flashnet trace
        # ts_record, dev_num, offset, size, io_type 
        # 0          1        2       3     4
        flashnet_prefix = '/home/cc/flashnet/model_collection/5_block_prefetching/dataset/iotrace/'
        flashnet_suffix = '/read_io.trace'
        lba_trace = dataset
        df = pd.read_csv(lba_trace, engine='python', skiprows=1, header=None, na_values=['-1'], usecols=[0, 2], names=names)

        # For Seagate trace
        # time, dev, offset, size, readwrite
        # 131054 0 17408 32768 1
        # lba_trace = flashnet_prefix + dataset
        # df = pd.read_csv(lba_trace, engine='python', skiprows=1, header=None, na_values=['-1'], usecols=[0, 2], names=names, sep=' ')
        
        print('\nReading trace: ', lba_trace, '\n')
        print("Length of trace", len(df))
        # print(df.head(3))

        # df = df.sort_values(by=['TimeStamp'])
        # df.reset_index(inplace=True, drop=True)

        train_trace = df[:int(len(df)*-opt.valid_portion)]['ByteOffset'].tolist()
        test_trace = df[int(len(df)*-opt.valid_portion)+1:]['ByteOffset'].tolist()

        dicts = dict_generate(df, top_num=top_num) # maps delta -> class

        train_data = tuple(trace2input(dicts, train_trace, window_size=window_size)) # 
        test_data = tuple(trace2input(dicts, test_trace, window_size=window_size))

        train_data = Data(train_data, shuffle=True)
        test_data = Data(test_data, shuffle=False)

        # For train
        train_silces = train_data.generate_batch(opt.batchSize)
        train_data_list = []
        for i in train_silces:
            alias_inputs, A, items, mask, targets = train_data.get_slice(i)
            train_data_list.append((alias_inputs, A, items, mask, targets))

        # For test
        test_silces = test_data.generate_batch(opt.batchSize)
        test_data_list = []
        for i in test_silces:
            alias_inputs, A, items, mask, targets = test_data.get_slice(i)
            test_data_list.append((alias_inputs, A, items, mask, targets))

        n_node = top_num + 3

        return train_data_list, train_silces, test_data_list, test_silces, dicts, n_node, train_trace, test_trace


def single_cache_test(test_trace, all_pred, save_name, dicts):
    bo_map, bo_map_div, operation_id_map, operation_id_map_div = dicts
    hit_rate = []
    prehit_rate = []
    stats = []
    caches = {}
    # maxsize = [5] + \
        # [i*10 for i in range(1, 10)] + [i*100 for i in range(1, 11)]
    
    maxsize = [1000]

    for i in range(len(maxsize)):
        caches["LRU"+str(maxsize[i])] = CacheTest(maxsize[i])

    # print("single test len", len(test_trace))
    actual_pred = [0] * len(test_trace)
    for i in range(0, len(test_trace)):
        for name, cache in caches.items():
            cache.push_normal(test_trace[i])
            # print(f"=== i {i}, requesting {test_trace[i]}", end=' ')
            if all_pred[i][0] > 0:
                cache.push_prefetch(test_trace[i] - operation_id_map_div[bo_map_div[all_pred[i][0]-1]])###
                # print(f", prefetch {test_trace[i] - operation_id_map_div[bo_map_div[all_pred[i][0]-1]]}")
                actual_pred[i] = test_trace[i] - operation_id_map_div[bo_map_div[all_pred[i][0]-1]]
            else:
                # print("no prefetch")
                actual_pred[i] = 0

    for name, cache in caches.items():
        print(format(cache.get_hit_rate(), '.4f'), format(cache.get_prehit_rate(), '.4f'), '\t', name)
        hit_rate.append(cache.get_hit_rate())
        prehit_rate.append(cache.get_prehit_rate())
        stats.append(cache.get_stats())

    # np.savetxt('hit_results/'+save_name+'_hit_rate.txt', hit_rate, fmt='%.4f')
    # np.savetxt('hit_results/'+save_name +'_pre_hit_rate.txt', prehit_rate, fmt='%.4f')
    # np.savetxt('hit_results/'+save_name+'_stats.txt', stats, fmt='%d')
    return actual_pred

# ERROR FUNCTION to FIX
def score_compute(all_preds, all_targets, save_name):
    # print(f"=== Len of all_preds {len(all_preds)}, all_target {len(all_targets)}")
    # print("all pred shape", all_preds.shape)
    # print("pred elements len", len(all_preds[0]), len(all_preds[1]), len(all_preds[-1]))

    # print("all tar", all_targets)
    
    pre_list = []
    mmr_list = []
    for i in range(1,len(all_preds[0])):
        pre_list.append(np.mean([np.where(t in p[:i],1,0) for t,p in zip(all_targets, all_preds)]))
        # mmr_list.append(np.mean([1/(np.where(p[:i]==t)[0]+1) if t in p[:i] else 0 for t,p in zip(all_targets,all_preds)]))
        mmr_list.append(np.mean([1/(np.where(p[:i]==t)[0][0]+1) if t in p[:i] else 0 for t,p in zip(all_targets,all_preds)]))

    np.savetxt('hit_results/'+save_name+'_pre_list.txt', pre_list, fmt='%.4f')  
    np.savetxt('hit_results/'+save_name +'_mmr_list.txt', mmr_list, fmt='%.4f')
    return pre_list,mmr_list

    
# output test_trace, pred
def graph_wrapper(raw_trace):
    # dataset_col = ['proj_0_1000.csv'] # MSR raw
    # dataset_col = ['msr.cut.per_50k.rw_78_22.200'] # MSR flashnet cut
    # dataset_col = ['seagate.16k.all_read.fio_90seq_10rand_256k_8q_reads_8_lun_10min_container192_filtered']
    # dataset_col = ['tencent.cut.per_100k.most_size_thpt.109'] # msr.cut.per_50k.rw_78_22.200 # alibaba.cut.per_50k.rw_27_73.140
    # dataset_col = ['alibaba.cut.per_50k.rw_27_73.140'] #alibaba.cut.per_10k.most_size_thpt_iops_rand.719
    
    print("Inside graph main, raw_trace", raw_trace)
    dataset = raw_trace
 
    deviceID = 0 # only one GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(deviceID)
    device = torch.device('cuda:'+str(deviceID))
    print("Is CUDA available:", torch.cuda.is_available())

    train_data_list, train_silces, test_data_list, test_silces, dicts, n_node, train_trace, test_trace = dataset2input(dataset=dataset, window_size=opt.window, top_num=opt.topnum)

    model = trans_to_cuda(SessionGraph(opt, n_node))
    model_path = 'checkpoint/'+'model_' + str(dataset)+'_'+time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    folder = os.path.exists(model_path)
    if not folder:
        os.makedirs(model_path)
    
    print('\n=== Start training, model_path:', model_path)
    for epoch in range(opt.epoch):
        print('===== epoch:', epoch)
        # print('start training: ')
        all_pred, all_targets = train_test_pred(model, train_data_list, train_silces, test_data_list, test_silces)

        # print('start cache test: ')
        save_name = dataset+'_'+str(epoch)+'_epoch'
        
        actual_pred = single_cache_test(test_trace=test_trace[opt.window:-1], all_pred=all_pred, save_name=save_name, dicts=dicts)
        
        # pre, mmr = score_compute(all_preds = all_pred, all_targets = all_targets, save_name = save_name)
        # print('pre:',pre)
        # print('mmr:',mmr)
        torch.save(model, os.path.join(model_path, str(epoch)+'.pt'))
    
    torch.cuda.empty_cache()
    
    return actual_pred, len(test_trace[opt.window:-1])

if __name__ == "__main__":
    # cd cd /home/cc/flashnet/model_collection/5_block_prefetching/simulate_sota_prefetcher
    # graph_wrapper("../../dataset/iotrace/alibaba.cut.per_10k.rw_80_20.723/read_io.trace")
    graph_wrapper("../../dataset/iotrace/seagate.16k.all_read.fio_90seq_10rand_256k_8q_reads_8_lun_10min_container192_filtered/read_io.trace")
    # graph_wrapper("../../dataset/iotrace/alibaba.cut.per_10k.most_size_thpt_iops_rand.719/read_io.trace")