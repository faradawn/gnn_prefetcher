#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py — Spectral Prefetcher

Evaluation uses build_adjacency_matrix_and_alias directly — the same
pattern as the SGDP graph_wrapper — so no non-existent Data methods
are called.

Metrics logged (per paper Section 5.3):
  - Hit Rate
  - Prefetch Hit Rate (prehit)
  - Access Speed  (SSD @ 0.1ms/miss, HDD @ 20ms/miss)
  - Prefetch Effectiveness  (bytes hit / bytes prefetched)
  - Prefetch Overhead       (extra bytes / user bytes)
  - Inference Latency       (total + per-inference)
  - Model Size              (# trainable parameters)
"""

import os
import time
import torch
import numpy as np
import pandas as pd
from collections import Counter
from tqdm import tqdm

from utils import Data, split_validation, build_adjacency_matrix_and_alias
from model import (
    SpectralSessionGraph,
    trans_to_cuda,
    trans_to_cpu,
    forward,
    forward_per_batch,
    training,
    testing_per_batch,
    testing_per_inference,
    run_single_inference,
)

try:
    from cache import CacheTest
except ImportError:
    class CacheTest:
        def __init__(self, maxsize):
            self.maxsize = maxsize
            self.hits = self.prehits = self.total = 0
            self.prefetch_issued = self.prefetch_used = 0
        def push_normal(self, lba): self.total += 1
        def push_prefetch(self, lba): self.prefetch_issued += 1
        def get_hit_rate(self): return 0.0
        def get_prehit_rate(self): return 0.0
        def get_stats(self): return [0, 0, 0]


# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------

class Options:
    def __init__(self):
        self.batchSize       = 128
        self.hiddenSize      = 150
        self.epoch           = 10
        self.lr              = 0.001
        self.lr_dc           = 0.1
        self.lr_dc_step      = 3
        self.l2              = 1e-5
        self.patience        = 3
        self.nonhybrid       = False
        self.validation      = False
        self.valid_portion   = 0.1
        self.topn            = 20
        self.dataset_percent = 1.0
        self.window          = 32
        self.topnum          = 1000
        self.dropout         = 0.1
        # Inference batch size for evaluation
        self.eval_batch_size = 512

opt = Options()

TRAINED_MODEL = None
GRAPH_DICTS   = None

# Assumed storage latencies (seconds) for access-speed metric
SSD_MISS_LATENCY_S = 0.0001   # 0.1 ms
HDD_MISS_LATENCY_S = 0.020    # 20  ms
BLOCK_SIZE_KB      = 8        # normalized block size

# Cache sizes evaluated per-epoch and in the final summary
MULTI_CACHE_SIZES = [10, 100, 1000]


# ---------------------------------------------------------------------------
# Data preprocessing
# ---------------------------------------------------------------------------

def dict_generate(train_trace, top_num=1000):
    train_trace = train_trace.copy()
    train_trace['KB_Offset_Delta'] = (
        train_trace['KB_Offset'].shift(-1) - train_trace['KB_Offset']
    ).fillna(0)

    a = train_trace['KB_Offset_Delta'].astype(int).unique().tolist()
    operation_id_map = {v: i for i, v in enumerate(a)}
    train_trace['KB_Offset_Delta_class'] = train_trace['KB_Offset_Delta'].map(
        lambda x: operation_id_map[x]
    )

    top_classes = set(
        cls for cls, _ in Counter(
            train_trace['KB_Offset_Delta_class']
        ).most_common(top_num)
    )
    train_trace['KB_Offset_Delta_class'] = train_trace['KB_Offset_Delta_class'].map(
        lambda x: x if x in top_classes else 999999
    )

    a = train_trace['KB_Offset_Delta_class'].unique().tolist()
    bo_map = {v: i for i, v in enumerate(a)}
    operation_id_map_div = {v: k for k, v in operation_id_map.items()}
    operation_id_map_div[999999] = 0
    bo_map_div = {v: k for k, v in bo_map.items()}

    return bo_map, bo_map_div, operation_id_map, operation_id_map_div


def trace2input(dicts, trace, window_size=32):
    bo_map, _, operation_id_map, _ = dicts
    keys = bo_map.keys()
    inputs, targets = [], []
    for i in range(len(trace) - window_size - 1):
        window = []
        for j in range(i, i + window_size + 1):
            diff = int(trace[j + 1] - trace[j])
            cls  = operation_id_map.get(diff, 999999)
            window.append(bo_map[cls] + 1 if cls in keys else bo_map[999999] + 1)
        inputs.append(window[:-1])
        targets.append(window[-1])
    return inputs, targets


def expand_to_8kb_blocks(lba_arr, size_arr):
    """Expand each I/O request into consecutive 8KB block accesses."""
    n_blocks = np.maximum(1, np.ceil(np.nan_to_num(size_arr) / 8192).astype(np.int64))
    cumsum   = np.concatenate([[0], np.cumsum(n_blocks[:-1])])
    total    = int(n_blocks.sum())
    row_idx  = np.repeat(np.arange(len(lba_arr)), n_blocks)
    within   = np.arange(total) - np.repeat(cumsum, n_blocks)
    return (lba_arr[row_idx] + within).tolist()


def dataset2input(dataset, window_size=32, top_num=1000):
    names = ['TimeStamp', 'KB_Offset', 'Size']
    df = pd.read_csv(
        dataset, engine='python', skiprows=0,
        header=None, na_values=['-1'],
        usecols=[0, 4, 5], names=names
    )
    df['KB_Offset'] = df['KB_Offset'] // 8192

    print(f'\nReading trace: {dataset}')
    print(f'Rows in trace: {len(df)}')

    lba_list = expand_to_8kb_blocks(
        df['KB_Offset'].values.astype(np.int64),
        df['Size'].fillna(0).values,
    )
    lba_df = pd.DataFrame({'KB_Offset': lba_list})
    print(f'Expanded to {len(lba_df)} 8KB block accesses')

    split = int(len(lba_df) * -opt.valid_portion)
    train_trace = lba_df[:split]['KB_Offset'].tolist()
    test_trace  = lba_df[split + 1:]['KB_Offset'].tolist()
    print(f' train: {len(train_trace)}, test: {len(test_trace)}')

    dicts  = dict_generate(lba_df, top_num=top_num)
    n_node = top_num + 3

    train_data = Data(tuple(trace2input(dicts, train_trace, window_size)), shuffle=True)
    test_data  = Data(tuple(trace2input(dicts, test_trace,  window_size)), shuffle=False)

    train_slices    = train_data.generate_batch(opt.batchSize)
    train_data_list = [train_data.get_slice(i) for i in train_slices]

    return train_data_list, train_slices, test_data, dicts, n_node, train_trace, test_trace


# ---------------------------------------------------------------------------
# Batched evaluation  (much faster than per-inference loop)
# ---------------------------------------------------------------------------

def run_batched_inference(model, arr_delta_classes, batch_size=512):
    """
    Build graph features for every test case, batch them up, and run
    forward_per_batch — same GCN path as training, no Python loop overhead
    per inference.

    Returns arr_raw_pred : list of int, length == len(arr_delta_classes)
    """
    model.eval()

    # Pre-build all graph features
    print('  Building graph features...')
    all_alias, all_items, all_A = [], [], []
    for delta_classes in tqdm(arr_delta_classes, desc='  Graph build'):
        [alias_input], A, items = build_adjacency_matrix_and_alias(delta_classes)
        # Flatten any spurious leading batch dim from build_adjacency_matrix_and_alias
        alias_input = np.array(alias_input).reshape(-1)       # (seq,)
        items_flat  = np.array(items).reshape(-1)             # (N,)
        A_flat      = np.array(A)
        if A_flat.ndim == 3:                                  # (1, N, 2N) → (N, 2N)
            A_flat = A_flat.squeeze(0)
        all_alias.append(alias_input)
        all_items.append(items_flat)
        all_A.append(A_flat)

    # Pad within each mini-batch so shapes align, then forward
    arr_raw_pred = []
    n = len(arr_delta_classes)

    print('  Running batched inference...')
    with torch.no_grad():
        for start in tqdm(range(0, n, batch_size), desc='  Inference'):
            end = min(start + batch_size, n)
            b_alias = all_alias[start:end]
            b_items = all_items[start:end]
            b_A     = all_A[start:end]

            # Pad nodes dimension to the max in this mini-batch
            max_N = max(len(it) for it in b_items)
            seq_len = len(b_alias[0])

            padded_alias = np.zeros((end - start, seq_len), dtype=np.int64)
            padded_items = np.zeros((end - start, max_N),  dtype=np.int64)
            padded_A     = np.zeros((end - start, max_N, 2 * max_N), dtype=np.float32)
            mask         = np.zeros((end - start, seq_len), dtype=np.int64)

            for i, (ali, ite, a) in enumerate(zip(b_alias, b_items, b_A)):
                n_nodes = len(ite)
                padded_alias[i] = ali
                padded_items[i, :n_nodes] = ite
                a_arr = np.array(a)                    # (n_nodes, 2*n_nodes)
                padded_A[i, :n_nodes, :n_nodes]                   = a_arr[:, :n_nodes]
                padded_A[i, :n_nodes, max_N:max_N + n_nodes]      = a_arr[:, n_nodes:]
                mask[i] = 1

            alias_t = trans_to_cuda(torch.tensor(padded_alias, dtype=torch.long))
            items_t = trans_to_cuda(torch.tensor(padded_items, dtype=torch.long))
            A_t     = trans_to_cuda(torch.tensor(padded_A,     dtype=torch.float))
            mask_t  = trans_to_cuda(torch.tensor(mask,         dtype=torch.long))

            hidden = model(items_t, A_t)               # (B, max_N, d)

            B, sl = alias_t.shape
            bidx  = torch.arange(B, device=hidden.device).unsqueeze(1).expand(-1, sl)
            seq_hidden = hidden[bidx, alias_t]         # (B, seq, d)

            scores    = model.compute_scores(seq_hidden, mask_t)   # (B, n_node-1)
            predicted = scores.argmax(dim=1)                       # (B,)
            arr_raw_pred.extend(trans_to_cpu(predicted).tolist())

    return arr_raw_pred


# ---------------------------------------------------------------------------
# Cache evaluation + full metric computation
# ---------------------------------------------------------------------------

def single_cache_test(test_trace, arr_raw_pred, dicts,
                      cache_size=1000,
                      inference_time_s=None,
                      n_inferences=None):
    """
    Runs cache simulation and computes all metrics from paper Section 5.3.
    Uses DequeLRU-backed CacheTest whose get_stats() returns:
        (total_ios, total_pres, total_hits, total_prehits)
    """
    bo_map, bo_map_div, operation_id_map, operation_id_map_div = dicts

    cache = CacheTest(cache_size)
    arr_lba_to_prefetch = []

    for test_id, last_lba in enumerate(test_trace):
        cache.push_normal(last_lba)

        pred = arr_raw_pred[test_id] if test_id < len(arr_raw_pred) else 0
        if pred > 0:
            actual_delta    = operation_id_map_div[bo_map_div[pred - 1]]
            lba_to_prefetch = last_lba + actual_delta
            cache.push_prefetch(lba_to_prefetch)
            arr_lba_to_prefetch.append(lba_to_prefetch)
        else:
            arr_lba_to_prefetch.append(0)

    total_ios, total_pres, total_hits, total_prehits = cache.get_stats()

    hit_rate    = cache.get_hit_rate()    # total_hits / total_ios
    prehit_rate = cache.get_prehit_rate() # total_prehits / total_pres

    n_misses = total_ios - total_hits

    # Access speed = total requests / workload duration
    # workload duration = misses * miss_latency (we have no prefetch latency model)
    ssd_duration     = n_misses * SSD_MISS_LATENCY_S
    hdd_duration     = n_misses * HDD_MISS_LATENCY_S
    access_speed_ssd = total_ios / ssd_duration if ssd_duration > 0 else float('inf')
    access_speed_hdd = total_ios / hdd_duration if hdd_duration > 0 else float('inf')

    # Prefetch effectiveness = prefetched blocks that were later accessed / total prefetched
    # total_prehits = prefetch blocks that caused a cache hit
    # total_pres    = total prefetch blocks inserted
    prefetch_effectiveness = (total_prehits / total_pres * 100) if total_pres > 0 else 0.0

    # Prefetch overhead = wasted prefetch bytes / user-requested bytes
    # wasted = prefetched blocks that were NEVER accessed (evicted without use)
    unused_prefetch  = total_pres - total_prehits
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


def single_cache_test_multi(test_trace, arr_raw_pred, test_wpos, dicts,
                            save_name, cache_sizes=None):
    """
    Run cache simulation at multiple sizes over the full test trace.
    Prefetch LBA = lba + delta (paper Eq. 4: lba_{n+1} = lba_n + ld_n).

    test_trace  : ALL test LBAs in order.
    arr_raw_pred: flat list of ints, one per prediction window.
    test_wpos   : index into test_trace for each window endpoint;
                  len == len(arr_raw_pred).
    """
    if cache_sizes is None:
        cache_sizes = MULTI_CACHE_SIZES

    bo_map, bo_map_div, operation_id_map, operation_id_map_div = dicts
    pos_to_pred = {pos: pred for pos, pred in zip(test_wpos, arr_raw_pred)}
    caches = {sz: CacheTest(sz) for sz in cache_sizes}

    for pos, lba in enumerate(test_trace):
        for cache in caches.values():
            cache.push_normal(lba)
        if pos in pos_to_pred:
            pred = pos_to_pred[pos]
            if pred > 0:
                try:
                    delta        = operation_id_map_div[bo_map_div[pred - 1]]
                    prefetch_lba = lba + delta
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
    safe = save_name.replace('/', '_').replace('\\', '_').replace('.', '_')
    np.savetxt(f'hit_results/{safe}_hit_rate.txt',     hit_rates,    fmt='%.4f')
    np.savetxt(f'hit_results/{safe}_pre_hit_rate.txt', prehit_rates, fmt='%.4f')
    np.savetxt(f'hit_results/{safe}_stats.txt',        stats,        fmt='%d')

    return hit_rates, prehit_rates


def log_metrics(metrics, model, dataset_name):
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print('\n' + '=' * 60)
    print(f'  SPECTRAL PREFETCHER — EVALUATION RESULTS')
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
# Global-state helpers for online / streaming use
# ---------------------------------------------------------------------------

def set_model_globally(model, dicts):
    global TRAINED_MODEL, GRAPH_DICTS
    TRAINED_MODEL = model
    GRAPH_DICTS   = dicts


def convert_class_to_delta(predicted_class):
    _, bo_map_div, _, operation_id_map_div = GRAPH_DICTS
    if predicted_class > 0:
        return operation_id_map_div[bo_map_div[predicted_class - 1]]
    return None


def convert_hist_delta_to_classes(historical_deltas):
    assert len(historical_deltas) == opt.window
    bo_map, _, operation_id_map, _ = GRAPH_DICTS
    keys = bo_map.keys()
    return [
        bo_map[operation_id_map[d]] + 1 if operation_id_map.get(d) in keys
        else bo_map[999999] + 1
        for d in historical_deltas
    ]


def predict_next_lba(last_lba, historical_deltas):
    delta_classes = convert_hist_delta_to_classes(historical_deltas)
    [alias_input], A, items = build_adjacency_matrix_and_alias(delta_classes)
    predicted_class = run_single_inference(TRAINED_MODEL, alias_input, items, A)
    actual_delta    = convert_class_to_delta(predicted_class)
    if actual_delta is not None:
        return last_lba + actual_delta
    return None


# ---------------------------------------------------------------------------
# Train convenience function
# ---------------------------------------------------------------------------

def train_model(dataset):
    train_data_list, train_slices, test_data, dicts, n_node, _, _ = \
        dataset2input(dataset=dataset, window_size=opt.window, top_num=opt.topnum)

    model = trans_to_cuda(SpectralSessionGraph(opt, n_node))
    for epoch in range(opt.epoch):
        print(f'===== epoch: {epoch}')
        model = training(model, train_data_list, train_slices)

    model.scheduler.step()
    model.eval()
    return model, dicts


# ---------------------------------------------------------------------------
# Main wrapper
# ---------------------------------------------------------------------------

def spectral_wrapper(raw_trace):
    print(f"spectral_wrapper: {raw_trace}")

    train_data_list, train_slices, test_data, dicts, n_node, train_trace, test_trace = \
        dataset2input(dataset=raw_trace, window_size=opt.window, top_num=opt.topnum)

    # Pre-compute test features used for both per-epoch eval and final inference
    arr_delta_classes  = test_data.get_data_as_list()
    test_trace_aligned = test_trace[opt.window:-1]
    n_test             = len(arr_delta_classes)
    # Position of each window endpoint within the full test_trace
    test_wpos          = list(range(opt.window, opt.window + n_test))

    model = trans_to_cuda(SpectralSessionGraph(opt, n_node))
    model_path = os.path.join(
        'checkpoint',
        'spectral_' + os.path.basename(str(raw_trace)) + '_' +
        time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
    )
    os.makedirs(model_path, exist_ok=True)
    safe_trace = raw_trace.replace('/', '_').replace('\\', '_').replace('.', '_')

    # ---- Training + per-epoch multi-cache evaluation ----
    print(f'\n=== Start training → {model_path}')
    for epoch in range(opt.epoch):
        print(f'===== epoch: {epoch}')
        model = training(model, train_data_list, train_slices)

        epoch_pred = run_batched_inference(model, arr_delta_classes,
                                           batch_size=opt.eval_batch_size)
        print(f'\n  Cache evaluation (epoch {epoch}):')
        single_cache_test_multi(test_trace, epoch_pred, test_wpos, dicts,
                                save_name=f'{safe_trace}_epoch{epoch}')
        torch.save(model, os.path.join(model_path, f'{epoch}.pt'))

    model.scheduler.step()
    model.eval()

    # ---- Final inference ----
    print(f'\nEvaluating {n_test} test cases...')

    t0 = time.time()
    arr_raw_pred = run_batched_inference(model, arr_delta_classes,
                                         batch_size=opt.eval_batch_size)
    inference_time = time.time() - t0

    # ---- Decode predictions → LBAs ----
    bo_map, bo_map_div, operation_id_map, operation_id_map_div = dicts
    arr_lba_to_prefetch = []
    for idx, predicted_class in enumerate(arr_raw_pred):
        if predicted_class > 0:
            actual_delta    = operation_id_map_div[bo_map_div[predicted_class - 1]]
            lba_to_prefetch = test_trace_aligned[idx] + actual_delta
            arr_lba_to_prefetch.append(lba_to_prefetch)
        else:
            arr_lba_to_prefetch.append(0)

    # ---- Cache simulation + metrics ----
    metrics, _ = single_cache_test(
        test_trace_aligned, arr_raw_pred, dicts,
        cache_size=1000,
        inference_time_s=inference_time,
        n_inferences=n_test,
    )
    log_metrics(metrics, model, raw_trace)

    # Final multi-size cache evaluation at [10, 100, 1000]
    print('\n=== Final multi-size cache evaluation:')
    single_cache_test_multi(test_trace, arr_raw_pred, test_wpos, dicts,
                            save_name=f'{safe_trace}_final')

    # ---- Save checkpoint ----
    torch.save(model, os.path.join(model_path, 'spectral_final.pt'))
    torch.cuda.empty_cache()

    return arr_lba_to_prefetch, len(test_trace_aligned)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    arr_lba_to_prefetch, n_tests = spectral_wrapper(
        'dataset/MSR-Cambridge/hm_1.csv.gz'
    )
    print(f'Number of test IOs : {n_tests}')
    print(f'Sample prefetch addresses: {arr_lba_to_prefetch[:10]}')