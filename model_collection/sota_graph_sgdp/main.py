#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import Counter

from utils import Data, split_validation, build_adjacency_matrix_and_alias
from model import *
from cache import *


# ---------------------------------------------------------------------------
# Configuration (user's Options class, updated to match paper)
# ---------------------------------------------------------------------------

class Options:
    def __init__(self):
        self.batchSize       = 128
        self.hiddenSize      = 100        # ~262k params; paper reports ~192k
        self.epoch           = 10
        self.lr              = 0.001
        self.lr_dc           = 0.1
        self.lr_dc_step      = 3
        self.l2              = 1e-5
        self.step            = 1
        self.patience        = 3
        self.nonhybrid       = False
        self.validation      = False
        self.valid_portion   = 0.1        # CHANGED: paper uses 90/10 (was 0.5)
        self.topn            = 20
        self.dataset_percent = 1.0
        self.window          = 32
        self.topnum          = 1000

opt = Options()

# Global state for online/streaming inference (user's addition)
TRAINED_MODEL = None
GRAPH_DICTS   = None

# Storage latency constants (user's addition)
SSD_MISS_LATENCY_S = 0.0001   # 0.1 ms per SSD miss
HDD_MISS_LATENCY_S = 0.020    # 20 ms per HDD miss

# Cache sizes evaluated per-epoch and in the final summary
MULTI_CACHE_SIZES  = [10, 100, 1000]
PRIMARY_CACHE_SIZE = 1000     # used for the detailed per-run summary


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def dict_generate(train_trace, top_num=1000):
    # Paper Eq. 2: ldi = lba_{i+1} - lba_i  (forward difference)
    # shift(-1) brings the *next* row up, so shift(-1) - current = lba_{i+1} - lba_i
    train_trace['KB_Offset_Delta'] = (
        train_trace['KB_Offset'].shift(-1) - train_trace['KB_Offset']
    )
    train_trace['KB_Offset_Delta'] = train_trace['KB_Offset_Delta'].fillna(0)

    a = train_trace['KB_Offset_Delta'].astype(int).unique().tolist()
    operation_id_map = {id_: i for i, id_ in enumerate(a)}

    train_trace['KB_Offset_Delta_class'] = train_trace['KB_Offset_Delta'].map(
        lambda x: operation_id_map[x]
    )

    vals    = Counter(train_trace['KB_Offset_Delta_class']).most_common(top_num)
    bo_list = [v[0] for v in vals]

    label_list = []
    for count in range(len(train_trace)):
        x = train_trace['KB_Offset_Delta_class'].iloc[count]
        label_list.append(x if x in bo_list else 999999)
    train_trace['KB_Offset_Delta_class'] = label_list

    a      = train_trace['KB_Offset_Delta_class'].unique().tolist()
    bo_map = {id_: i for i, id_ in enumerate(a)}

    operation_id_map_div         = {v: k for k, v in operation_id_map.items()}
    operation_id_map_div[999999] = 0
    bo_map_div                   = {v: k for k, v in bo_map.items()}
    
    return bo_map, bo_map_div, operation_id_map, operation_id_map_div


def _encode_window(dicts, trace_segment, window_size):
    """
    Sliding window encoder.

    Returns (inputs, targets, lbas) where lbas[i] is the raw LBA value that
    sits at the end of window i — i.e. trace_segment[i + window_size].
    This is the LBA that gets pushed to the cache during simulation, so it
    must travel alongside the predictions rather than being reconstructed
    from a separate trace slice (which breaks under stream splitting).
    """
    bo_map, _, operation_id_map, _ = dicts
    keys    = bo_map.keys()
    inputs  = []
    targets = []
    lbas    = []
    for i in range(len(trace_segment) - window_size - 1):
        input_single = []
        for j in range(i, i + window_size + 1):
            # Paper Eq. 2: ldi = lba_{i+1} - lba_i (forward difference, must
            # match the direction used in dict_generate)
            diff = int(trace_segment[j + 1] - trace_segment[j])
            if operation_id_map[diff] in keys:
                input_single.append(bo_map[operation_id_map[diff]] + 1)
            else:
                input_single.append(bo_map[999999] + 1)
        inputs.append(input_single[:-1])
        targets.append(input_single[-1])
        lbas.append(trace_segment[i + window_size])   # LBA at end of context window
    return inputs, targets, lbas


def trace2input(dicts, trace, window_size=32):
    """Flat sliding window over a plain LBA list (no stream splitting).
    Kept for compatibility with online inference helpers.
    Returns (inputs, targets, lbas)."""
    return _encode_window(dicts, trace, window_size)



def expand_to_8kb_blocks(lba_arr, size_arr):
    """Expand each I/O request into consecutive 8KB block accesses."""
    n_blocks = np.maximum(1, np.ceil(np.nan_to_num(size_arr) / 8192).astype(np.int64))
    cumsum   = np.concatenate([[0], np.cumsum(n_blocks[:-1])])
    total    = int(n_blocks.sum())
    row_idx  = np.repeat(np.arange(len(lba_arr)), n_blocks)
    within   = np.arange(total) - np.repeat(cumsum, n_blocks)
    return (lba_arr[row_idx] + within).tolist()


def dataset2input(dataset, window_size=32, method='top', top_num=1000):
    """
    Loads trace, sorts by timestamp (original paper), splits 90/10,
    and returns BOTH batched test data (for per-epoch eval) and the
    raw Data object (for final per-inference eval).
    """
    if method != 'top':
        raise ValueError("Only 'top' method is supported.")

    lba_trace = dataset
    df = pd.read_csv(lba_trace, engine='python', skiprows=0, header=None,
                     na_values=['-1'], usecols=[0, 4, 5],
                     names=['TimeStamp', 'KB_Offset', 'Size'])
    df['KB_Offset'] = df['KB_Offset'] // 8192

    # Sort by timestamp before expansion so blocks stay in temporal order
    df = df.sort_values(by=['TimeStamp']).reset_index(drop=True)

    print(f'\nReading trace: {lba_trace}')
    print(f'Rows in trace: {len(df)}')

    lba_list = expand_to_8kb_blocks(
        df['KB_Offset'].values.astype(np.int64),
        df['Size'].fillna(0).values,
    )
    lba_df = pd.DataFrame({'KB_Offset': lba_list})
    print(f'Expanded to {len(lba_df)} 8KB block accesses')

    split_idx   = int(len(lba_df) * -opt.valid_portion)
    train_trace = lba_df[:split_idx]['KB_Offset'].tolist()
    test_trace  = lba_df[split_idx + 1:]['KB_Offset'].tolist()
    print(f' train: {len(train_trace)}, test: {len(test_trace)}')

    # Build vocabulary from the full expanded trace (original paper protocol)
    dicts = dict_generate(lba_df, top_num=top_num)

    # Flat sliding window over the full train/test traces (no stream splitting).
    # Window i ends at position (i + window_size) in each sub-trace, so
    # test_wpos is a simple contiguous range — no timestamp logic needed.
    train_inputs, train_targets, _train_lbas = trace2input(dicts, train_trace, window_size=window_size)
    test_inputs,  test_targets,  test_lbas   = trace2input(dicts, test_trace,  window_size=window_size)
    test_wpos = list(range(window_size, window_size + len(test_inputs)))
    train_data_tup = (train_inputs, train_targets)
    test_data_tup  = (test_inputs,  test_targets)

    train_data_obj = Data(train_data_tup, shuffle=True)
    test_data_obj  = Data(test_data_tup,  shuffle=False)   # kept for per-inference eval

    # Batched train data
    train_slices    = train_data_obj.generate_batch(opt.batchSize)
    train_data_list = []
    for i in train_slices:
        alias_inputs, A, items, mask, targets = train_data_obj.get_slice(i)
        train_data_list.append((alias_inputs, A, items, mask, targets))

    # Batched test data — ADDED: used for per-epoch score_compute + cache eval
    test_slices    = test_data_obj.generate_batch(opt.batchSize)
    test_data_list = []
    for i in test_slices:
        alias_inputs, A, items, mask, targets = test_data_obj.get_slice(i)
        test_data_list.append((alias_inputs, A, items, mask, targets))

    n_node = top_num + 3

    # test_data_obj returned separately for final per-inference timed evaluation.
    # test_lbas        : window-endpoint LBAs, aligned 1-to-1 with predictions.
    # test_wpos        : absolute position of each window endpoint in test_trace;
    #                    used by cache sims to inject prefetches into the FULL trace.
    # test_trace       : ALL test LBAs in order — needed for correct HR/EPR.
    return (train_data_list, train_slices,
            test_data_list,  test_slices,
            test_data_obj,
            dicts, n_node, train_trace, test_trace, test_lbas, test_wpos, test_inputs)


# ---------------------------------------------------------------------------
# Cache evaluation — multi-size (original paper protocol)
# ADDED: tests the full range of cache sizes from the paper's tables,
#        saves HR/EPR/stats files, and prints a summary row per size.
# ---------------------------------------------------------------------------

def single_cache_test_multi(test_trace_full, all_pred, save_name, dicts,
                             window_end_positions, cache_sizes=None):
    """
    all_pred              : list of (top_k,) numpy arrays, one per prediction window.
    test_trace_full       : ALL test LBAs in chronological order (correct denominator).
    window_end_positions  : index into test_trace_full for each window endpoint;
                            len == len(all_pred).

    The cache sees every LBA in test_trace_full.  Prefetches are injected only at
    prediction points, matching the paper's simulation protocol.
    """
    if cache_sizes is None:
        cache_sizes = MULTI_CACHE_SIZES

    bo_map, bo_map_div, operation_id_map, operation_id_map_div = dicts

    # Map absolute trace position → top-1 predicted class
    pos_to_pred = {pos: int(all_pred[j][0])
                   for j, pos in enumerate(window_end_positions)}

    caches = {sz: CacheTest(sz) for sz in cache_sizes}

    for pos, lba in enumerate(test_trace_full):
        for cache in caches.values():
            cache.push_normal(lba)
        if pos in pos_to_pred:
            pred_class = pos_to_pred[pos]
            if pred_class > 0:
                try:
                    delta = operation_id_map_div[bo_map_div[pred_class - 1]]
                    prefetch_lba = lba + delta   # paper Eq. 4
                    for cache in caches.values():
                        cache.push_prefetch(prefetch_lba)
                except (KeyError, IndexError):
                    pass

    hit_rates, prehit_rates, stats = [], [], []
    for sz, cache in caches.items():
        hr  = cache.get_hit_rate()
        phr = cache.get_prehit_rate()
        print(f'  cache={sz:5d}  HR={hr:.4f}  EPR={phr:.4f}')
        hit_rates.append(hr)
        prehit_rates.append(phr)
        stats.append(cache.get_stats())

    os.makedirs('hit_results', exist_ok=True)
    safe_save = save_name.replace('/', '_').replace('\\', '_').replace('.', '_')
    np.savetxt(f'hit_results/{safe_save}_hit_rate.txt',     hit_rates,    fmt='%.4f')
    np.savetxt(f'hit_results/{safe_save}_pre_hit_rate.txt', prehit_rates, fmt='%.4f')
    np.savetxt(f'hit_results/{safe_save}_stats.txt',        stats,        fmt='%d')

    return hit_rates, prehit_rates


# ---------------------------------------------------------------------------
# Cache evaluation — detailed single-size (user's addition)
# Kept intact; used for the final per-run summary after all epochs.
# ---------------------------------------------------------------------------

def single_cache_test(test_trace_full, arr_raw_pred, window_end_positions,
                      save_name, dicts,
                      cache_size=PRIMARY_CACHE_SIZE,
                      inference_time_s=None, n_inferences=None):
    """
    arr_raw_pred         : flat list of ints (top-1 prediction per window).
    test_trace_full      : ALL test LBAs in chronological order.
    window_end_positions : index into test_trace_full for each window endpoint;
                           len == len(arr_raw_pred).
    Returns (metrics dict, list of prefetched LBAs aligned with arr_raw_pred).
    """
    bo_map, bo_map_div, operation_id_map, operation_id_map_div = dicts

    pos_to_pred = {pos: pred
                   for pos, pred in zip(window_end_positions, arr_raw_pred)}

    cache               = CacheTest(cache_size)
    arr_lba_to_prefetch = [0] * len(arr_raw_pred)   # one slot per prediction

    print(f'Total IO in the test set  {len(test_trace_full)}')
    pred_idx = {pos: j for j, pos in enumerate(window_end_positions)}

    for pos, lba in enumerate(test_trace_full):
        cache.push_normal(lba)
        if pos in pos_to_pred:
            j    = pred_idx[pos]
            pred = pos_to_pred[pos]
            if pred > 0:
                try:
                    delta           = operation_id_map_div[bo_map_div[pred - 1]]
                    lba_to_prefetch = lba + delta   # paper Eq. 4
                    cache.push_prefetch(lba_to_prefetch)
                    arr_lba_to_prefetch[j] = lba_to_prefetch
                except (KeyError, IndexError):
                    pass

    total_ios, total_pres, total_hits, total_prehits = cache.get_stats()
    n_misses = total_ios - total_hits

    ssd_dur = n_misses * SSD_MISS_LATENCY_S
    hdd_dur = n_misses * HDD_MISS_LATENCY_S

    metrics = {
        'hit_rate':               cache.get_hit_rate() * 100,
        'prehit_rate':            cache.get_prehit_rate() * 100,
        'access_speed_ssd':       total_ios / ssd_dur if ssd_dur > 0 else float('inf'),
        'access_speed_hdd':       total_ios / hdd_dur if hdd_dur > 0 else float('inf'),
        'prefetch_effectiveness': (total_prehits / total_pres * 100)
                                   if total_pres > 0 else 0.0,
        'prefetch_overhead':      ((total_pres - total_prehits) / total_ios * 100)
                                   if total_ios > 0 else 0.0,
        'total_ios':    total_ios,
        'total_pres':   total_pres,
        'total_hits':   total_hits,
        'total_prehits':total_prehits,
        'n_misses':     n_misses,
    }
    if inference_time_s is not None and n_inferences is not None:
        metrics['inference_latency_s']      = inference_time_s
        metrics['per_inference_latency_ms'] = (
            inference_time_s / n_inferences * 1000 if n_inferences > 0 else 0
        )
        metrics['throughput_inf_per_s'] = (
            n_inferences / inference_time_s if inference_time_s > 0 else 0
        )

    return metrics, arr_lba_to_prefetch


def log_metrics(metrics, model, dataset_name):
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('\n' + '=' * 60)
    print('  SGDP — EVALUATION RESULTS')
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
# Precision / MRR (original paper)
# ADDED: restores the per-epoch quality metrics from the original code.
# Fixed the MRR indexing bug present in the original ([0][0] not [0]).
# ---------------------------------------------------------------------------

def score_compute(all_preds, all_targets, save_name):
    """
    all_preds  : list of (top_k,) numpy arrays
    all_targets: list of ints
    Returns (pre_list, mmr_list).
    """
    pre_list, mmr_list = [], []
    for i in range(1, len(all_preds[0]) + 1):
        pre_list.append(
            np.mean([1 if t in p[:i] else 0
                     for t, p in zip(all_targets, all_preds)])
        )
        mmr_list.append(
            np.mean([
                1 / (np.where(p[:i] == t)[0][0] + 1) if t in p[:i] else 0
                for t, p in zip(all_targets, all_preds)
            ])
        )
    os.makedirs('hit_results', exist_ok=True)
    np.savetxt(f'hit_results/{save_name}_pre_list.txt', pre_list, fmt='%.4f')
    np.savetxt(f'hit_results/{save_name}_mmr_list.txt', mmr_list, fmt='%.4f')
    return pre_list, mmr_list


# ---------------------------------------------------------------------------
# Global-state helpers for online / streaming inference (user's addition)
# ---------------------------------------------------------------------------

def train_model(dataset):
    """Convenience wrapper: trains and returns (model, dicts)."""
    (train_data_list, train_slices, _, _, _, dicts,
     n_node, _, _, _, _, _) = dataset2input(
        dataset=dataset, window_size=opt.window, top_num=opt.topnum
    )
    model = trans_to_cuda(SessionGraph(opt, n_node))
    for epoch in range(opt.epoch):
        print(f'===== epoch: {epoch}')
        model = training(model, train_data_list, train_slices)
    model.scheduler.step()
    model.eval()
    return model, dicts


def set_model_globaly(model, dicts):
    global TRAINED_MODEL, GRAPH_DICTS
    TRAINED_MODEL = model
    GRAPH_DICTS   = dicts


def convert_class_to_delta(predicted_class):
    bo_map, bo_map_div, operation_id_map, operation_id_map_div = GRAPH_DICTS
    if predicted_class > 0:
        return operation_id_map_div[bo_map_div[predicted_class - 1]]
    return None


def convert_hist_delta_to_classes(historical_deltas):
    assert len(historical_deltas) == opt.window, \
        f"Expected {opt.window} deltas, got {len(historical_deltas)}"
    bo_map, _, operation_id_map, _ = GRAPH_DICTS
    keys = bo_map.keys()
    delta_classes = []
    for delta in historical_deltas:
        if operation_id_map[delta] in keys:
            delta_classes.append(bo_map[operation_id_map[delta]] + 1)
        else:
            delta_classes.append(bo_map[999999] + 1)
    return delta_classes


def predict_next_lba(last_lba, historical_deltas):
    delta_classes   = convert_hist_delta_to_classes(historical_deltas)
    predicted_class = run_single_inference(TRAINED_MODEL, delta_classes)
    actual_delta    = convert_class_to_delta(predicted_class)
    if predicted_class > 0:
        return last_lba + actual_delta  # Paper Eq. 4: lba_{n+1} = lba_n + ld_n
    return None


# ---------------------------------------------------------------------------
# Main training + evaluation wrapper
# ---------------------------------------------------------------------------

def graph_wrapper(raw_trace):
    print('Inside graph main, raw_trace', raw_trace)

    (train_data_list, train_slices,
     test_data_list,  test_slices,
     test_data_obj,
     dicts, n_node,
     train_trace, test_trace, test_lbas, test_wpos, test_inputs) = dataset2input(
        dataset=raw_trace, window_size=opt.window, top_num=opt.topnum
    )

    model      = trans_to_cuda(SessionGraph(opt, n_node))
    model_path = ('checkpoint/model_' + str(raw_trace) + '_' +
                  time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime()))
    os.makedirs(model_path, exist_ok=True)

    print(f'train_data_list len {len(train_data_list)}')
    print(f'\n=== Start training, model_path: {model_path}')

    # -----------------------------------------------------------------------
    # Per-epoch training + evaluation (original paper protocol)
    # Each epoch: train → batch predict (top-20) → multi-cache eval → scores
    # -----------------------------------------------------------------------
    for epoch in range(opt.epoch):
        print(f'\n===== epoch: {epoch}')

        # Train one epoch; get top-20 predictions for score_compute
        all_pred_topk, all_targets = train_test_pred(
            model, train_data_list, train_slices,
            test_data_list, top_k=20, batching=True
        )

        safe_trace = raw_trace.replace('/', '_').replace('\\', '_').replace('.', '_')
        save_name  = f'{safe_trace}_epoch{epoch}'

        # Multi-cache-size HR/EPR over the FULL test trace (paper protocol)
        print(f'\n  Cache evaluation (epoch {epoch}):')
        hit_rates, prehit_rates = single_cache_test_multi(
            test_trace_full      = test_trace,
            all_pred             = all_pred_topk,
            save_name            = save_name,
            dicts                = dicts,
            window_end_positions = test_wpos,
        )

        # Precision and MRR (original paper)
        pre, mmr = score_compute(all_pred_topk, all_targets, save_name)
        print(f'  Precision@k: {[f"{v:.4f}" for v in pre]}')
        print(f'  MRR@k      : {[f"{v:.4f}" for v in mmr]}')

        # Save per-epoch checkpoint
        torch.save(model, os.path.join(model_path, f'{epoch}.pt'))

    # -----------------------------------------------------------------------
    # Final per-inference evaluation with timing (user's addition)
    # -----------------------------------------------------------------------
    model.eval()
    print('\nDo per inference testing:')

    print(f'     len test_windows {len(test_lbas)}  /  full test_trace {len(test_trace)}')
    
    arr_raw_pred = []
    start_eval = time.time()
    with torch.no_grad():
        for delta_classes in test_inputs:
            predicted_class = run_single_inference(model, delta_classes)
            arr_raw_pred.append(predicted_class)
    inference_time = time.time() - start_eval

    metrics, arr_lba_to_prefetch = single_cache_test(
        test_trace_full      = test_trace,
        arr_raw_pred         = arr_raw_pred,
        window_end_positions = test_wpos,
        save_name            = raw_trace,
        dicts                = dicts,
        cache_size           = PRIMARY_CACHE_SIZE,
        inference_time_s     = inference_time,
        n_inferences         = len(test_inputs),
    )
    log_metrics(metrics, model, raw_trace)

    # Final multi-size cache evaluation at [10, 100, 1000]
    print('\n=== Final multi-size cache evaluation:')
    arr_pred_wrapped = [np.array([p]) for p in arr_raw_pred]
    single_cache_test_multi(
        test_trace_full      = test_trace,
        all_pred             = arr_pred_wrapped,
        save_name            = raw_trace.replace('/', '_').replace('\\', '_').replace('.', '_') + '_final',
        dicts                = dicts,
        window_end_positions = test_wpos,
        cache_sizes          = MULTI_CACHE_SIZES,
    )

    torch.cuda.empty_cache()
    return arr_lba_to_prefetch, len(test_trace)


if __name__ == '__main__':
    arr_lba_to_prefetch, n_tests = graph_wrapper('dataset/MSR-Cambridge/hm_1.csv.gz')
    print('Number of test IOs:', n_tests)
    print('Sample prefetch addresses:', arr_lba_to_prefetch[:10])