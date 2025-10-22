#!/usr/bin/env python36
# -*- coding: utf-8 -*-
import argparse
import pickle
import time
import os
import torch
import pandas as pd
import numpy as np


from utils import Data, split_validation, build_adjacency_matrix_and_alias
from model import *
from tqdm import tqdm
from collections import Counter
from cache import *

class Options:
    def __init__(self):
        self.batchSize = 128
        self.hiddenSize = 150
        self.epoch = 10
        self.lr = 0.001
        self.lr_dc = 0.1
        self.lr_dc_step = 3
        self.l2 = 1e-5
        self.step = 1
        self.patience = 3
        self.nonhybrid = False
        self.validation = False
        self.valid_portion = 0.5
        self.draw_graph = True
        self.see_ori_dataset = False
        self.topn = 20
        self.dataset_percent = 1.0
        self.window = 32
        self.topnum = 1000

opt = Options()

TRAINED_MODEL = None
GRAPH_DICTS = None

def dict_generate(train_trace, top_num=1000):
    train_trace['KB_Offset_Delta'] = train_trace['KB_Offset'] - \
        train_trace['KB_Offset'].shift(-1)
    # print(train_trace.head(5))
    # exit(0)
    train_trace['KB_Offset_Delta'] = train_trace['KB_Offset_Delta'].fillna(0)

    a = train_trace['KB_Offset_Delta'].astype(int).unique().tolist()

    operation_id_map = {}
    for i, id in enumerate(a):
        operation_id_map[id] = i
    train_trace['KB_Offset_Delta_class'] = train_trace['KB_Offset_Delta'].map(
        lambda x: operation_id_map[x])

    x = Counter(train_trace['KB_Offset_Delta_class'])
    vals = {}
    vals = x.most_common(top_num)
    bo_list = []

    for x in vals:
        bo_list.append(x[0])

    count = 0
    label_list = []
    while (count < len(train_trace)):
        x = train_trace['KB_Offset_Delta_class'].iloc[count]
        if x in bo_list:
            label_list.append(x)
        else:
            label_list.append(999999)  # no Prefetch class
        count = count + 1

    train_trace['KB_Offset_Delta_class'] = label_list
    a = train_trace['KB_Offset_Delta_class'].unique().tolist()
    bo_map = {}
    for i, id in enumerate(a):
        bo_map[id] = i
    operation_id_map_div = {v: k for k, v in operation_id_map.items()}
    operation_id_map_div[999999] = 0
    bo_map_div = {v: k for k, v in bo_map.items()}

    return bo_map, bo_map_div, operation_id_map, operation_id_map_div


def trace2input(dicts, trace, window_size=32):
    bo_map, _, operation_id_map, _ = dicts
    keys = bo_map.keys()
    inputs = []
    targets = []
    for i in range(len(trace)-window_size-1):

        input_single = []
        for j in range(i, i+window_size+1):
            # calculating the real delta
            diff = int(trace[j]-trace[j+1])
            if operation_id_map[diff] in keys:
                input_single.append(bo_map[operation_id_map[diff]]+1)###
            else:
                input_single.append(bo_map[999999]+1)###
        inputs.append(input_single[:-1])
        targets.append(input_single[-1])
    return inputs, targets

def build_feature_from_test_data(test_data):
    n_tests = len(test_data.inputs)
    test_data_list = []
    # Version 2. Batch size is 1
    for idx in range(n_tests):
        # Q: What is A? The dimension of A is changing for each batch (128, 21, 42), (128, 15, 30)
        # input_feature, A, items = test_data.get_one_slice([idx])

        delta_classes = test_data.inputs[[idx]][0]
        input_feature, A, items = build_adjacency_matrix_and_alias(delta_classes)

        # exit(0)
        # tmp = np.array(A)
        # print(" A ", len(A), tmp.shape)
        test_data_list.append((input_feature, A, items, None, None))
        # if idx > 10:
        #     exit(0)
    return test_data_list

def dataset2input(dataset, window_size=32, method='top', top_num=1000):
    if method == 'top':
        names = ['TimeStamp', 'KB_Offset']
        # For flashnet trace
        # ts_record, dev_num, offset, size, io_type 
        # 0          1        2       3     4
        flashnet_prefix = '/home/cc/flashnet/model_collection/5_block_prefetching/dataset/iotrace/'
        flashnet_suffix = '/read_io.trace'
        lba_trace = dataset
        df = pd.read_csv(lba_trace, engine='python', skiprows=1, header=None, na_values=['-1'], usecols=[0, 2], names=names)
        df['KB_Offset'] = df['KB_Offset'] // 1024 # Originally it was in bytes, now it is in KB

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

        train_trace = df[:int(len(df)*-opt.valid_portion)]['KB_Offset'].tolist()
        test_trace = df[int(len(df)*-opt.valid_portion)+1:]['KB_Offset'].tolist()
        print(" train_trace ", len(train_trace), train_trace[0], train_trace[1])
        n_tests = len(test_trace) - window_size - 1
        dicts = dict_generate(df, top_num=top_num) # maps delta -> class
        # exit(0)

        train_data = tuple(trace2input(dicts, train_trace, window_size=window_size)) # 
        # print(" len test_trace", len(test_trace))

        test_data = tuple(trace2input(dicts, test_trace, window_size=window_size))

        train_data = Data(train_data, shuffle=True)
        test_data = Data(test_data, shuffle=False)

        # For train
        train_slices = train_data.generate_batch(opt.batchSize)
        train_data_list = []
        for i in train_slices:
            alias_inputs, A, items, mask, targets = train_data.get_slice(i)
            train_data_list.append((alias_inputs, A, items, mask, targets))

        # For test
        test_data_list = []

        # Version 1. All the tests are contained in a single batch
        # all_indexes = [i for i in range(n_tests)]
        # alias_inputs, A, items, mask, targets = test_data.get_slice(all_indexes)
        # test_data_list.append((alias_inputs, A, items, mask, targets))
        # tmp = np.array(A)
        # print(" A ", len(A), tmp.shape)
        # exit(0)

        # Version 2. Batch size is 1
        if False:
            for idx in range(n_tests):
                # Q: What is A? The dimension of A is changing for each batch (128, 21, 42), (128, 15, 30)
                # alias_inputs, A, items = test_data.get_one_slice([idx])

                delta_classes = test_data.inputs[[idx]][0]
                alias_inputs, A, items = build_adjacency_matrix_and_alias(delta_classes)

                # exit(0)
                tmp = np.array(A)
                # print(" A ", len(A), tmp.shape)
                test_data_list.append((alias_inputs, A, items, None, None))
                # if idx > 10:
                #     exit(0)

        # Version 3. Original code (Prepare the test in 128 batch size)
        # test_silces = test_data.generate_batch(opt.batchSize)
        # print("test_silces len", len(test_silces))
        # for i in test_silces:
        #     print(" i ", len(i))
        #     # Q: What is A? The dimension of A is changing for each batch (128, 21, 42), (128, 15, 30)
        #     alias_inputs, A, items, mask, targets = test_data.get_slice(i)

        #     tmp = np.array(A)
        #     print(" A ", len(A), tmp.shape)
        #     test_data_list.append((alias_inputs, A, items, mask, targets))

        n_node = top_num + 3

        return train_data_list, train_slices, test_data, dicts, n_node, train_trace, test_trace


# Just calculating the hit rate and stats
# arr_raw_pred -> contains 0-1000 predicted classes
def single_cache_test(test_trace, arr_raw_pred, save_name, dicts):
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


    print("Total IO in the test set ", len(test_trace))
    arr_lba_to_prefetch = []
    # Q: Why it is not forming the delta based features from the test trace?
    for test_id, last_lba in enumerate(test_trace):
        # Q: Why is this iterating the item in the cache?
        for name, cache in caches.items():
            cache.push_normal(last_lba)
            # print(f"=== i {i}, requesting {test_trace[i]}")
            # print("arr_raw_pred[i][0]", arr_raw_pred[i][0])
            if arr_raw_pred[test_id] > 0:
                # print( " predicted classes ", arr_raw_pred[test_id])
                actual_delta = operation_id_map_div[bo_map_div[arr_raw_pred[test_id]-1]]
                # print( " actual_delta ", actual_delta)
                lba_to_prefetch = test_trace[test_id] - actual_delta
                cache.push_prefetch(lba_to_prefetch)###
                # print(test_trace[i], operation_id_map_div[bo_map_div[arr_raw_pred[i][0]-1]])
                arr_lba_to_prefetch.append(lba_to_prefetch)
                # print(lba_to_prefetch)
            else:
                # print("no prefetch")
                arr_lba_to_prefetch.append(0)
    
    # Just printing the cache hit rate and stats
    for name, cache in caches.items():
        print(format(cache.get_hit_rate(), '.4f'), format(cache.get_prehit_rate(), '.4f'), '\t', name)
        hit_rate.append(cache.get_hit_rate())
        prehit_rate.append(cache.get_prehit_rate())
        stats.append(cache.get_stats())

    # np.savetxt('hit_results/'+save_name+'_hit_rate.txt', hit_rate, fmt='%.4f')
    # np.savetxt('hit_results/'+save_name +'_pre_hit_rate.txt', prehit_rate, fmt='%.4f')
    # np.savetxt('hit_results/'+save_name+'_stats.txt', stats, fmt='%d')
    return arr_lba_to_prefetch

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

def train_model(dataset):
    train_data_list, train_slices, test_data, dicts, n_node, train_trace, test_trace = dataset2input(dataset=dataset, window_size=opt.window, top_num=opt.topnum) # type: ignore

    # Only the first 50% will be used for training (self.valid_portion = 0.5)
    model = trans_to_cuda(SessionGraph(opt, n_node))
    for epoch in range(opt.epoch):
        print('===== epoch:', epoch)
        print('start training: ')
        model = training(model,train_data_list,train_slices)
        # arr_raw_pred = train_test_pred(model, train_data_list, train_slices, test_data_list, batching=False)
    # Ready for Testing 
    model.scheduler.step()
    model.eval()
    return model, dicts

def set_model_globaly(model, dicts):
    global TRAINED_MODEL, GRAPH_DICTS
    TRAINED_MODEL = model
    GRAPH_DICTS = dicts

def convert_class_to_delta(predicted_class):
    global GRAPH_DICTS
    bo_map, bo_map_div, operation_id_map, operation_id_map_div = GRAPH_DICTS
    if predicted_class > 0:
        actual_delta = operation_id_map_div[bo_map_div[predicted_class - 1]]
        return actual_delta
    else:
        return None

def convert_hist_delta_to_classes(hitorical_deltas):
    global GRAPH_DICTS
    assert len(hitorical_deltas) == 32 # window size must be 32
    bo_map, _, operation_id_map, _ = GRAPH_DICTS
    keys = bo_map.keys()
    delta_classes = []
    # print(" operation_id_map ", len(operation_id_map), operation_id_map)
    for delta in hitorical_deltas:
        if operation_id_map[delta] in keys:
            delta_classes.append(bo_map[operation_id_map[delta]] + 1)###
        else:
            delta_classes.append(bo_map[999999]+1)
    return delta_classes

def predict_next_lba(last_lba, historical_deltas):
    global TRAINED_MODEL, GRAPH_DICTS
    # convert historical delta to its corresponding class
    delta_classes = convert_hist_delta_to_classes(historical_deltas)

    # Run inference
    predicted_class = run_single_inference(TRAINED_MODEL, delta_classes)

    actual_delta = convert_class_to_delta(predicted_class)
    if predicted_class > 0:
        # print( " actual_delta ", actual_delta)
        lba_to_prefetch = last_lba - actual_delta
        # print(lba_to_prefetch)
        return lba_to_prefetch
    else:
        return None

# output test_trace, pred
def graph_wrapper(raw_trace):
    # dataset_col = ['proj_0_1000.csv'] # MSR raw
    # dataset_col = ['msr.cut.per_50k.rw_78_22.200'] # MSR flashnet cut
    # dataset_col = ['seagate.16k.all_read.fio_90seq_10rand_256k_8q_reads_8_lun_10min_container192_filtered']
    # dataset_col = ['tencent.cut.per_100k.most_size_thpt.109'] # msr.cut.per_50k.rw_78_22.200 # alibaba.cut.per_50k.rw_27_73.140
    # dataset_col = ['alibaba.cut.per_50k.rw_27_73.140'] #alibaba.cut.per_10k.most_size_thpt_iops_rand.719
    
    print("Inside graph main, raw_trace", raw_trace)
    dataset = raw_trace
 
    # deviceID = 0 # only one GPU
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(deviceID)
    # device = torch.device('cuda:'+str(deviceID))
    # print("Is CUDA available:", torch.cuda.is_available())

    train_data_list, train_slices, test_data, dicts, n_node, train_trace, test_trace = dataset2input(dataset=dataset, window_size=opt.window, top_num=opt.topnum) # type: ignore

    # building the features from the raw test data 
    # test_data_list = build_feature_from_test_data(test_data)

    model = trans_to_cuda(SessionGraph(opt, n_node))
    model_path = 'checkpoint/'+'model_' + str(dataset)+'_'+time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    folder = os.path.exists(model_path)
    if not folder:
        os.makedirs(model_path)
    
    print("train_data_list len", len(train_data_list))
    print(train_data_list[0][0][1])

    print('\n=== Start training, model_path:', model_path)

    # Training
    for epoch in range(opt.epoch):
        print('===== epoch:', epoch)
        print('start training: ')
        model = training(model,train_data_list,train_slices)
        # arr_raw_pred = train_test_pred(model, train_data_list, train_slices, test_data_list, batching=False)

    # Testing 
    model.scheduler.step()
    model.eval()
    print("\nDo per inference testing:")
    # test_data is array of historical deltas
    arr_delta_classes = test_data.get_data_as_list()
    # print("     len arr_delta_classes", len(arr_delta_classes), arr_delta_classes[0])
    # exit(0)
    arr_raw_pred = []
    arr_lba_to_prefetch = []
    bo_map, bo_map_div, operation_id_map, operation_id_map_div = dicts
    test_trace=test_trace[opt.window:-1]
    print("     len test_trace", len(test_trace))

    with torch.no_grad():
        for idx, delta_classes in enumerate(arr_delta_classes):
            # Submit each inference to the model
            predicted_class = run_single_inference(model, delta_classes)
            # print("hitorical_deltas", hitorical_deltas)
            arr_raw_pred.append(predicted_class)

            if predicted_class > 0:
                actual_delta = operation_id_map_div[bo_map_div[predicted_class - 1]]
                # print( " actual_delta ", actual_delta)
                lba_to_prefetch = test_trace[idx] - actual_delta
                arr_lba_to_prefetch.append(lba_to_prefetch)
                # print(lba_to_prefetch)
            else:
                arr_lba_to_prefetch.append(0)
    # exit(0)
    # save_name = dataset+'_0_epoch'
    # arr_lba_to_prefetch = single_cache_test(test_trace, arr_raw_pred = arr_raw_pred, save_name=save_name, dicts=dicts)
    
    torch.save(model, os.path.join(model_path, str(0)+'.pt'))
    torch.cuda.empty_cache()
    return arr_lba_to_prefetch, len(test_trace)

# 

