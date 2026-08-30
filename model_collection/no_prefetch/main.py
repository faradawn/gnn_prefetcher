#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
no_prefetch/main.py

Evaluates a pure LRU cache with no prefetching at cache sizes [10, 100, 1000].
Provides the lower-bound baseline for comparison with SGDP and Spectral GCN.

Same 90/10 train/test split and MSR-Cambridge CSV format as SGDP/GCN.
"""

import os
import numpy as np
import pandas as pd

from cache import CacheTest

CACHE_SIZES        = [10, 100, 1000]
VALID_PORTION      = 0.1
SSD_MISS_LATENCY_S = 0.0001   # 0.1 ms per SSD miss
HDD_MISS_LATENCY_S = 0.020    # 20  ms per HDD miss


# ---------------------------------------------------------------------------
# Trace loading
# ---------------------------------------------------------------------------

def load_test_trace(dataset):
    df = pd.read_csv(dataset, engine='python', skiprows=0, header=None,
                     na_values=['-1'], usecols=[0, 4],
                     names=['TimeStamp', 'Offset'])
    df = df.sort_values(by=['TimeStamp']).reset_index(drop=True)
    
    #df = df.head(600000)
    
    print(f'\nReading trace: {dataset}')
    print(f'Rows in trace : {len(df)}')

    lba_list = df['Offset'].values.tolist()

    split_idx   = int(len(lba_list) * -VALID_PORTION)
    train_trace = lba_list[:split_idx]
    test_trace  = lba_list[split_idx + 1:]
    print(f'  train: {len(train_trace)},  test: {len(test_trace)}')
    return test_trace


# ---------------------------------------------------------------------------
# Cache evaluation
# ---------------------------------------------------------------------------

def cache_eval(test_trace, save_name, cache_sizes=None):
    """
    Run a pure LRU cache simulation (no prefetching) over test_trace at every
    size in cache_sizes.  Saves hit_rate / pre_hit_rate / stats to hit_results/.
    """
    if cache_sizes is None:
        cache_sizes = CACHE_SIZES

    caches = {sz: CacheTest(sz) for sz in cache_sizes}

    for lba in test_trace:
        for cache in caches.values():
            cache.push_normal(lba)

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

    return hit_rates, prehit_rates, stats


# ---------------------------------------------------------------------------
# Metrics logging
# ---------------------------------------------------------------------------

def log_metrics(hit_rates, prehit_rates, stats, dataset_name):
    # Use the largest cache size as the reference for access-speed calculation
    ref_idx = len(CACHE_SIZES) - 1
    total_ios, _, total_hits, _ = stats[ref_idx]
    n_misses = total_ios - total_hits
    ssd_dur  = n_misses * SSD_MISS_LATENCY_S
    hdd_dur  = n_misses * HDD_MISS_LATENCY_S
    spd_ssd  = total_ios / ssd_dur if ssd_dur > 0 else float('inf')
    spd_hdd  = total_ios / hdd_dur if hdd_dur > 0 else float('inf')

    print('\n' + '=' * 60)
    print('  NO PREFETCH — EVALUATION RESULTS')
    print(f'  Dataset : {dataset_name}')
    print('=' * 60)
    for sz, hr, phr in zip(CACHE_SIZES, hit_rates, prehit_rates):
        print(f'  Cache Size {sz:5d}  |  HR: {hr*100:.2f}%   EPR: {phr*100:.2f}%')
    print(f'  Total IOs              : {total_ios:,}')
    print(f'  Cache Misses (sz={CACHE_SIZES[-1]}) : {n_misses:,}')
    print(f'  Access Speed (SSD)     : {spd_ssd:.1f} req/s')
    print(f'  Access Speed (HDD)     : {spd_hdd:.1f} req/s')
    print('=' * 60 + '\n')


# ---------------------------------------------------------------------------
# Main wrapper
# ---------------------------------------------------------------------------

def no_prefetch_wrapper(raw_trace):
    print(f'no_prefetch_wrapper: {raw_trace}')
    test_trace = load_test_trace(raw_trace)

    print('\n=== Evaluating LRU cache (no prefetching):')
    hit_rates, prehit_rates, stats = cache_eval(test_trace, raw_trace)
    log_metrics(hit_rates, prehit_rates, stats, raw_trace)

    return hit_rates


if __name__ == '__main__':
    no_prefetch_wrapper('dataset/MSR-Cambridge/src1_2.csv.gz')
