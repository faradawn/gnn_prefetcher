#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py — Stride prefetcher baseline.

Minimal implementation following the SGDP paper's description:
    "Stride Prefetcher [6] simultaneously records 128 LBA access streams,
     and each of them tracks the last 3 LBA accesses. Each access is mapped
     to a stream based on hashing the most significant LBA. If the
     difference between the 3 LBA accesses matches, it will detect a stride
     and conduct a prediction."

Reference: J.W. Fu, J.H. Patel, B.L. Janssens, "Stride directed prefetching
in scalar processors," ACM SIGMICRO 1992.

No training, no learned model. Same data pipeline (8 KB block expansion,
90/10 train-test split) and same LRU cache simulator as the GCN/SGDP/LSTM
runners so HR / EPR @ 10/100/1000 are directly comparable.

Wrapper:  stride_wrapper(trace_path) -> (arr_lba_to_prefetch, n_tests)
"""

import os, time
import numpy as np
import pandas as pd

try:
    from cache import CacheTest
except ImportError:
    class CacheTest:
        def __init__(self, m):
            self.m = m; self.t = self.h = self.p = self.ph = 0
        def push_normal(self, lba): self.t += 1
        def push_prefetch(self, lba): self.p += 1
        def get_hit_rate(self): return 0.0
        def get_prehit_rate(self): return 0.0
        def get_stats(self): return (0, 0, 0, 0)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class Options:
    def __init__(self):
        self.valid_portion = 0.1     # last 10% = test (matches GCN/SGDP/LSTM)
        self.n_streams     = 128     # paper-cited stride table size
        self.history_len   = 3       # last 3 LBAs per stream
        self.hash_shift    = 14      # group LBAs by upper bits (per-region streams)

opt = Options()
MULTI_CACHE_SIZES  = [10, 100, 1000]
SSD_MISS_LATENCY_S = 0.0001
HDD_MISS_LATENCY_S = 0.020


# ---------------------------------------------------------------------------
# Data loading  (mirrors GCN/SGDP/LSTM, no delta-class encoding here)
# ---------------------------------------------------------------------------

def expand_to_8kb_blocks(lba_arr, size_arr):
    n_blocks = np.maximum(1, np.ceil(np.nan_to_num(size_arr) / 8192).astype(np.int64))
    cumsum   = np.concatenate([[0], np.cumsum(n_blocks[:-1])])
    total    = int(n_blocks.sum())
    row_idx  = np.repeat(np.arange(len(lba_arr)), n_blocks)
    within   = np.arange(total) - np.repeat(cumsum, n_blocks)
    return (lba_arr[row_idx] + within).tolist()


def load_trace(dataset):
    names = ['TimeStamp', 'KB_Offset', 'Size']
    df = pd.read_csv(
        dataset, engine='python', skiprows=0,
        header=None, na_values=['-1'],
        usecols=[0, 4, 5], names=names
    )
    df['KB_Offset'] = df['KB_Offset'] // 8192
    df = df.sort_values(by=['TimeStamp']).reset_index(drop=True)

    print(f'\nReading trace: {dataset}')
    print(f'Rows in trace: {len(df)}')

    lba_list = expand_to_8kb_blocks(
        df['KB_Offset'].values.astype(np.int64),
        df['Size'].fillna(0).values,
    )
    print(f'Expanded to {len(lba_list)} 8KB block accesses')

    split = int(len(lba_list) * -opt.valid_portion)
    train_trace = lba_list[:split]
    test_trace  = lba_list[split + 1:]
    print(f' train: {len(train_trace)}, test: {len(test_trace)}')
    return train_trace, test_trace


# ---------------------------------------------------------------------------
# Stride prefetcher
# ---------------------------------------------------------------------------

class StridePrefetcher:
    """
    128 streams; each stream holds the last `history_len` LBAs.
    Stream index = (LBA >> hash_shift) % n_streams  — paper uses "hashing the
    most significant LBA". A stride is detected when the last two deltas in
    a stream are equal (i.e. lba[2]-lba[1] == lba[1]-lba[0]); the prefetch
    is then lba[2] + stride.
    """
    def __init__(self, n_streams=128, history_len=3, hash_shift=14):
        self.n_streams   = n_streams
        self.history_len = history_len
        self.hash_shift  = hash_shift
        # one history list per stream
        self.history = [[] for _ in range(n_streams)]

    def _stream_id(self, lba):
        return int((lba >> self.hash_shift) % self.n_streams)

    def predict(self, lba):
        """Update the stream for `lba` and return a prefetch LBA, or None."""
        sid  = self._stream_id(lba)
        hist = self.history[sid]
        hist.append(lba)
        if len(hist) > self.history_len:
            hist.pop(0)
        if len(hist) < self.history_len:
            return None
        d1 = hist[1] - hist[0]
        d2 = hist[2] - hist[1]
        if d1 == d2 and d1 != 0:
            return lba + d1
        return None


# ---------------------------------------------------------------------------
# Cache evaluation
# ---------------------------------------------------------------------------

def single_cache_test_multi(test_trace, prefetch, save_name,
                            cache_sizes=None):
    """
    `prefetch[i]` is the LBA to prefetch when the i-th access happens, or 0/None.
    Same simulator shape as the GCN/SGDP runners: every test LBA is pushed to
    the cache, prefetches are issued at the same step.
    """
    if cache_sizes is None:
        cache_sizes = MULTI_CACHE_SIZES
    caches = {sz: CacheTest(sz) for sz in cache_sizes}

    for i, lba in enumerate(test_trace):
        for c in caches.values():
            c.push_normal(lba)
        pf = prefetch[i] if i < len(prefetch) else None
        if pf:
            for c in caches.values():
                c.push_prefetch(pf)

    hit_rates, prehit_rates, stats = [], [], []
    for sz, c in caches.items():
        hr, phr = c.get_hit_rate(), c.get_prehit_rate()
        print(f'  cache={sz:5d}  HR={hr:.4f}  EPR={phr:.4f}')
        hit_rates.append(hr); prehit_rates.append(phr); stats.append(c.get_stats())

    os.makedirs('hit_results', exist_ok=True)
    safe = save_name.replace('/', '_').replace('\\', '_').replace('.', '_')
    np.savetxt(f'hit_results/{safe}_hit_rate.txt',     hit_rates,    fmt='%.4f')
    np.savetxt(f'hit_results/{safe}_pre_hit_rate.txt', prehit_rates, fmt='%.4f')
    np.savetxt(f'hit_results/{safe}_stats.txt',        stats,        fmt='%d')
    return hit_rates, prehit_rates


def log_summary(test_trace, prefetch, inference_time_s, n_inferences,
                dataset_name, n_params=0):
    cache = CacheTest(1000)
    for i, lba in enumerate(test_trace):
        cache.push_normal(lba)
        pf = prefetch[i] if i < len(prefetch) else None
        if pf:
            cache.push_prefetch(pf)
    total_ios, total_pres, total_hits, total_prehits = cache.get_stats()
    n_misses = total_ios - total_hits
    ssd_dur  = n_misses * SSD_MISS_LATENCY_S
    hdd_dur  = n_misses * HDD_MISS_LATENCY_S
    eff      = (total_prehits / total_pres * 100) if total_pres > 0 else 0.0
    over     = ((total_pres - total_prehits) / total_ios * 100) if total_ios > 0 else 0.0

    print('\n' + '=' * 60)
    print('  STRIDE PREFETCHER — EVALUATION RESULTS')
    print(f'  Dataset : {dataset_name}')
    print('=' * 60)
    print(f"  Hit Rate                  : {cache.get_hit_rate()*100:.2f}%")
    print(f"  Prefetch Hit Rate         : {cache.get_prehit_rate()*100:.2f}%")
    print(f"  Access Speed (SSD)        : "
          f"{total_ios/ssd_dur if ssd_dur>0 else float('inf'):.1f} req/s")
    print(f"  Access Speed (HDD)        : "
          f"{total_ios/hdd_dur if hdd_dur>0 else float('inf'):.1f} req/s")
    print(f"  Prefetch Effectiveness    : {eff:.2f}%")
    print(f"  Prefetch Overhead         : {over:.2f}%")
    print(f"  Total IOs                 : {total_ios:,}")
    print(f"  Total Prefetches Issued   : {total_pres:,}")
    print(f"  Cache Hits                : {total_hits:,}")
    print(f"  Prefetch Hits             : {total_prehits:,}")
    print(f"  Cache Misses              : {n_misses:,}")
    if inference_time_s > 0:
        print(f"  Inference Time (total)    : {inference_time_s:.2f}s")
        print(f"  Inference Latency (per)   : "
              f"{inference_time_s/n_inferences*1000:.3f}ms")
        print(f"  Throughput                : {n_inferences/inference_time_s:.1f} inf/s")
    print(f"  Model Size (# params)     : {n_params:,}  (no learned params)")
    print('=' * 60 + '\n')


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------

def stride_wrapper(raw_trace):
    print(f'stride_wrapper: {raw_trace}')
    train_trace, test_trace = load_trace(raw_trace)
    safe_trace = raw_trace.replace('/', '_').replace('\\', '_').replace('.', '_')

    # Warm up the prefetcher's stream history on the train portion. Strictly
    # speaking the classic Stride prefetcher doesn't need any training, but
    # warming up means streams are already populated when we hit the test
    # boundary -- it costs a single linear pass and avoids unfair "cold start"
    # misses.
    pf = StridePrefetcher(
        n_streams=opt.n_streams,
        history_len=opt.history_len,
        hash_shift=opt.hash_shift,
    )
    print('Warming up stream history on train trace...')
    for lba in train_trace:
        pf.predict(lba)

    print(f'Running stride prediction on {len(test_trace)} test cases...')
    t0 = time.time()
    prefetch = []
    for lba in test_trace:
        prefetch.append(pf.predict(lba))
    inference_time = time.time() - t0
    n_pred = sum(1 for p in prefetch if p)
    print(f'  {n_pred:,} prefetches issued '
          f'({n_pred / len(test_trace) * 100:.2f}% of accesses)')

    log_summary(test_trace, prefetch, inference_time, len(test_trace),
                raw_trace, n_params=0)

    print('\n=== Final multi-size cache evaluation:')
    single_cache_test_multi(test_trace, prefetch,
                            save_name=f'{safe_trace}_final')

    arr_lba_to_prefetch = [p if p is not None else 0 for p in prefetch]
    return arr_lba_to_prefetch, len(test_trace)


if __name__ == '__main__':
    import sys
    trace = sys.argv[1] if len(sys.argv) > 1 else \
        '/home/nvidia/gnn_prefetcher/dataset/MSR-Cambridge/hm_1.csv.gz'
    arr, n = stride_wrapper(trace)
    print(f'Number of test IOs : {n}')
    print(f'Sample prefetch addresses: {arr[:10]}')
