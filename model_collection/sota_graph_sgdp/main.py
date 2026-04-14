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

# Assumed storage latencies (seconds) for access-speed metric
SSD_MISS_LATENCY_S = 0.0001   # 0.1 ms
HDD_MISS_LATENCY_S = 0.020    # 20  ms
BLOCK_SIZE_KB      = 4        # assumed block size for effectiveness/overhead


def dict_generate(train_trace, top_num=1000):
    train_trace['KB_Offset_Delta'] = train_trace['KB_Offset'] - \
        train_trace['KB_Offset'].shift(-1)
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
            diff = int(trace[j]-trace[j+1])
            if operation_id_map[diff] in keys:
                input_single.append(bo_map[operation_id_map[diff]]+1)
            else:
                input_single.append(bo_map[999999]+1)
        inputs.append(input_single[:-1])
        targets.append(input_single[-1])
    return inputs, targets

def build_feature_from_test_data(test_data):
    n_tests = len(test_data.inputs)
    test_data_list = []
    for idx in range(n_tests):
        delta_classes = test_data.inputs[[idx]][0]
        input_feature, A, items = build_adjacency_matrix_and_alias(delta_classes)
        test_data_list.append((input_feature, A, items, None, None))
    return test_data_list

def dataset2input(dataset, window_size=32, method='top', top_num=1000):
    if method == 'top':
        names = ['TimeStamp', 'KB_Offset']
        lba_trace = dataset
        df = pd.read_csv(lba_trace, engine='python', skiprows=0, header=None, na_values=['-1'], usecols=[0, 4], names=names)
        df['KB_Offset'] = df['KB_Offset'] // 1024

        print('\nReading trace: ', lba_trace, '\n')
        print("Length of trace", len(df))

        train_trace = df[:int(len(df)*-opt.valid_portion)]['KB_Offset'].tolist()
        test_trace = df[int(len(df)*-opt.valid_portion)+1:]['KB_Offset'].tolist()
        print(" train_trace ", len(train_trace), train_trace[0], train_trace[1])
        n_tests = len(test_trace) - window_size - 1
        dicts = dict_generate(df, top_num=top_num)

        train_data = tuple(trace2input(dicts, train_trace, window_size=window_size))
        test_data = tuple(trace2input(dicts, test_trace, window_size=window_size))

        train_data = Data(train_data, shuffle=True)
        test_data = Data(test_data, shuffle=False)

        train_slices = train_data.generate_batch(opt.batchSize)
        train_data_list = []
        for i in train_slices:
            alias_inputs, A, items, mask, targets = train_data.get_slice(i)
            train_data_list.append((alias_inputs, A, items, mask, targets))

        n_node = top_num + 3

        return train_data_list, train_slices, test_data, dicts, n_node, train_trace, test_trace


# ---------------------------------------------------------------------------
# Metrics computation  (matches spectral main.py Section 5.3)
# ---------------------------------------------------------------------------

def single_cache_test(test_trace, arr_raw_pred, save_name, dicts,
                      cache_size=1000,
                      inference_time_s=None,
                      n_inferences=None):
    """
    Runs cache simulation and computes all metrics matching spectral main.py.

    CacheTest.get_stats() must return (total_ios, total_pres, total_hits, total_prehits).
    """
    bo_map, bo_map_div, operation_id_map, operation_id_map_div = dicts

    cache = CacheTest(cache_size)
    arr_lba_to_prefetch = []

    print("Total IO in the test set ", len(test_trace))
    for test_id, last_lba in enumerate(test_trace):
        cache.push_normal(last_lba)

        if arr_raw_pred[test_id] > 0:
            actual_delta = operation_id_map_div[bo_map_div[arr_raw_pred[test_id] - 1]]
            lba_to_prefetch = test_trace[test_id] - actual_delta
            cache.push_prefetch(lba_to_prefetch)
            arr_lba_to_prefetch.append(lba_to_prefetch)
        else:
            arr_lba_to_prefetch.append(0)

    total_ios, total_pres, total_hits, total_prehits = cache.get_stats()

    hit_rate    = cache.get_hit_rate()
    prehit_rate = cache.get_prehit_rate()

    n_misses = total_ios - total_hits

    ssd_duration     = n_misses * SSD_MISS_LATENCY_S
    hdd_duration     = n_misses * HDD_MISS_LATENCY_S
    access_speed_ssd = total_ios / ssd_duration if ssd_duration > 0 else float('inf')
    access_speed_hdd = total_ios / hdd_duration if hdd_duration > 0 else float('inf')

    # Prefetch effectiveness: fraction of issued prefetches that resulted in a hit
    prefetch_effectiveness = (total_prehits / total_pres * 100) if total_pres > 0 else 0.0

    # Prefetch overhead: unused prefetch blocks as a fraction of total user IOs
    unused_prefetch   = total_pres - total_prehits
    prefetch_overhead = (unused_prefetch / total_ios * 100) if total_ios > 0 else 0.0

    metrics = {
        'hit_rate':               hit_rate * 100,
        'prehit_rate':            prehit_rate * 100,
        'access_speed_ssd':       access_speed_ssd,
        'access_speed_hdd':       access_speed_hdd,
        'prefetch_effectiveness': prefetch_effectiveness,
        'prefetch_overhead':      prefetch_overhead,
        'total_ios':              total_ios,
        'total_pres':             total_pres,
        'total_hits':             total_hits,
        'total_prehits':          total_prehits,
        'n_misses':               n_misses,
    }

    if inference_time_s is not None and n_inferences is not None:
        metrics['inference_latency_s']      = inference_time_s
        metrics['per_inference_latency_ms'] = (inference_time_s / n_inferences * 1000) if n_inferences > 0 else 0
        metrics['throughput_inf_per_s']     = n_inferences / inference_time_s if inference_time_s > 0 else 0

    return metrics, arr_lba_to_prefetch


def log_metrics(metrics, model, dataset_name):
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print('\n' + '=' * 60)
    print(f'  SGDP — EVALUATION RESULTS')
    print(f'  Dataset : {dataset_name}')
    print('=' * 60)
    print(f"  Hit Rate                  : {metrics['hit_rate']:.2f}%")
    print(f"  Prefetch Hit Rate         : {metrics['prehit_rate']:.2f}%")
    print(f"  Access Speed (SSD)        : {metrics['access_speed_ssd']:.1f} req/s")
    print(f"  Access Speed (HDD)        : {metrics['access_speed_hdd']:.1f} req/s")
    print(f"  Prefetch Effectiveness    : {metrics['prefetch_effectiveness']:.2f}%")
    print(f"  Prefetch Overhead         : {metrics['prefetch_overhead']:.2f}%")
    print(f"  Total IOs                 : {metrics['total_ios']:,}")
    print(f"  Total Prefetches Issued   : {metrics['total_pres']:,}")
    print(f"  Cache Hits                : {metrics['total_hits']:,}")
    print(f"  Prefetch Hits             : {metrics['total_prehits']:,}")
    print(f"  Cache Misses              : {metrics['n_misses']:,}")
    if 'inference_latency_s' in metrics:
        print(f"  Inference Time (total)    : {metrics['inference_latency_s']:.2f}s")
        print(f"  Inference Latency (per)   : {metrics['per_inference_latency_ms']:.3f}ms")
        print(f"  Throughput                : {metrics['throughput_inf_per_s']:.1f} inf/s")
    print(f"  Model Size (# params)     : {n_params:,}")
    print('=' * 60 + '\n')


# ---------------------------------------------------------------------------
# ERROR FUNCTION to FIX (kept from original)
# ---------------------------------------------------------------------------

def score_compute(all_preds, all_targets, save_name):
    pre_list = []
    mmr_list = []
    for i in range(1,len(all_preds[0])):
        pre_list.append(np.mean([np.where(t in p[:i],1,0) for t,p in zip(all_targets, all_preds)]))
        mmr_list.append(np.mean([1/(np.where(p[:i]==t)[0][0]+1) if t in p[:i] else 0 for t,p in zip(all_targets,all_preds)]))

    np.savetxt('hit_results/'+save_name+'_pre_list.txt', pre_list, fmt='%.4f')  
    np.savetxt('hit_results/'+save_name +'_mmr_list.txt', mmr_list, fmt='%.4f')
    return pre_list,mmr_list


# ---------------------------------------------------------------------------
# Global-state helpers for online / streaming use
# ---------------------------------------------------------------------------

def train_model(dataset):
    train_data_list, train_slices, test_data, dicts, n_node, train_trace, test_trace = dataset2input(dataset=dataset, window_size=opt.window, top_num=opt.topnum)

    model = trans_to_cuda(SessionGraph(opt, n_node))
    for epoch in range(opt.epoch):
        print('===== epoch:', epoch)
        print('start training: ')
        model = training(model,train_data_list,train_slices)
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
    assert len(hitorical_deltas) == 32
    bo_map, _, operation_id_map, _ = GRAPH_DICTS
    keys = bo_map.keys()
    delta_classes = []
    for delta in hitorical_deltas:
        if operation_id_map[delta] in keys:
            delta_classes.append(bo_map[operation_id_map[delta]] + 1)
        else:
            delta_classes.append(bo_map[999999]+1)
    return delta_classes

def predict_next_lba(last_lba, historical_deltas):
    global TRAINED_MODEL, GRAPH_DICTS
    delta_classes = convert_hist_delta_to_classes(historical_deltas)
    predicted_class = run_single_inference(TRAINED_MODEL, delta_classes)
    actual_delta = convert_class_to_delta(predicted_class)
    if predicted_class > 0:
        lba_to_prefetch = last_lba - actual_delta
        return lba_to_prefetch
    else:
        return None


# ---------------------------------------------------------------------------
# Main wrapper
# ---------------------------------------------------------------------------

def graph_wrapper(raw_trace):
    print("Inside graph main, raw_trace", raw_trace)
    dataset = raw_trace

    train_data_list, train_slices, test_data, dicts, n_node, train_trace, test_trace = dataset2input(dataset=dataset, window_size=opt.window, top_num=opt.topnum)

    model = trans_to_cuda(SessionGraph(opt, n_node))
    model_path = 'checkpoint/'+'model_' + str(dataset)+'_'+time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    folder = os.path.exists(model_path)
    if not folder:
        os.makedirs(model_path)

    print("train_data_list len", len(train_data_list))
    print(train_data_list[0][0][1])
    print('\n=== Start training, model_path:', model_path)

    # ---- Training ----
    for epoch in range(opt.epoch):
        print('===== epoch:', epoch)
        print('start training: ')
        model = training(model, train_data_list, train_slices)

    # ---- Inference ----
    model.scheduler.step()
    model.eval()
    print("\nDo per inference testing:")

    arr_delta_classes = test_data.get_data_as_list()
    arr_raw_pred = []
    bo_map, bo_map_div, operation_id_map, operation_id_map_div = dicts
    test_trace = test_trace[opt.window:-1]
    print("     len test_trace", len(test_trace))

    start_eval_time = time.time()

    with torch.no_grad():
        for idx, delta_classes in enumerate(arr_delta_classes):
            predicted_class = run_single_inference(model, delta_classes)
            arr_raw_pred.append(predicted_class)

    inference_time = time.time() - start_eval_time
    n_inferences   = len(arr_delta_classes)

    # ---- Cache simulation + metrics ----
    metrics, arr_lba_to_prefetch = single_cache_test(
        test_trace, arr_raw_pred,
        save_name=dataset,
        dicts=dicts,
        cache_size=1000,
        inference_time_s=inference_time,
        n_inferences=n_inferences,
    )
    log_metrics(metrics, model, raw_trace)

    # ---- Save checkpoint ----
    torch.save(model, os.path.join(model_path, str(0)+'.pt'))
    torch.cuda.empty_cache()

    return arr_lba_to_prefetch, len(test_trace)


if __name__ == '__main__':
    arr_lba_to_prefetch, n_tests = graph_wrapper('dataset/MSR-Cambridge/prxy_0.csv.gz')
    print("Number of test IOs:", n_tests)
    print("Sample prefetch addresses:", arr_lba_to_prefetch[:10])