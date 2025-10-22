#!/usr/bin/env python3

import argparse
import csv
import numpy as np
import os
import sys
import subprocess
from subprocess import call
from pathlib import Path
import pandas as pd 
import math
import matplotlib.pyplot as plt
import collections
import sys
sys.path.append('./sota_lstm')
import sota_lstm.main as lstm_wrapper

sys.path.append('../../../commonutils')
import pattern_checker

from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error 
import pickle

def write_stats(statistics, output_file):
    with open(output_file, "w") as text_file:
        for line in statistics:
            text_file.write(str(line) + "\n")
    print("===== output file : " + output_file)

def create_output_dir(output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    return output_path

# save to a file
def write_to_file(df, filePath, has_header=True):
    # The raw (replayed) traces don't have the header
    df.to_csv(filePath, index=False, header=has_header, sep=',')
    print("===== output file : " + filePath)

def read_file(input_file):
    df = pd.read_csv(input_file, sep=',')
    # Make sure it has 5 columns (ts_record  dev_num        offset    size  io_type)
    assert 5 == df.shape[1]
    assert "offset" in df.columns.tolist()
    # Rename column
    # Format = ts_record(ms),latency(us),io_type(r=1/w=0),
    #          size(B),offset,ts_submit(ms),size_after_replay(B)
    return df

MAX_QUEUE_SIZE = 10         # Length of input feature, for calculating the threshold, etc
MAX_HIST_IO_BUFFER = 32     # This must be 32; same as LSTM input features size
MAJORITY_THRESHOLD = int (MAX_QUEUE_SIZE * 0.3)
CACHED_STREAMS = collections.OrderedDict() # the value is just the metadata = [size, hit_count, .. ]
CACHE_CAPACITY_KB = 0
SIZE_PER_KEY = 4 #KB    # We will push multiple times if the prefetch size is bigger than 1KB
N_HIT_FURTHER_PREFETCH = 2  # required hit before we do further prefetch on this stream

# Markov chain prefetcher
# MARKOV_TABLE = {}

# 6500236 kb 
PREFIX_LBA_DIVISOR = 1024 * 20 # 20MB
PICKLED_MODEL = None 
MAX_LBA = 20480 # KB == 20 MB
EVICTION_RECORD = ["(lba_key, [size, hit_count, io_stream] )"]
PER_KB_HIT_RECORDS = {}
HDD_FETCH_LATENCY = 20 # ms
DATA_IN_FLIGHT = []
DATA_IN_FLIGHT_KEYS = set()
N_PREV_TO_CHECK = 0
N_NEXT_TO_CHECK = 0
MAX_PREFETCH_BANDWDTH_MBPS = -1 # MBps
MAX_KB_PREFETCHED_PER_MS = -1 # KB per milisecond should be the same as MAX_PREFETCH_BANDWDTH_MBPS
TOTAL_DATA_IN_FLIGHT_KB = 0 # KB
AVAILABLE_READ_AHEAD_SIZE_KB = 0 # KB; We can only prefetch if there is enough quota
DOUBLE_PREFETCH = False
PAUSE_PREFETCH = False

def fetch_data_from_hdd(cache_key, io_stream, size_kb):
    global DATA_IN_FLIGHT, DATA_IN_FLIGHT_KEYS, HDD_FETCH_LATENCY, TOTAL_DATA_IN_FLIGHT_KB
    if cache_key in DATA_IN_FLIGHT_KEYS:
        # the data is already in-flight
        return
    TOTAL_DATA_IN_FLIGHT_KB += size_kb
    DATA_IN_FLIGHT.append([HDD_FETCH_LATENCY, cache_key, io_stream, size_kb]) # [time_delay, key, stream, size]
    DATA_IN_FLIGHT_KEYS.add(cache_key)

def insert_pending_data(time_gap_ms):
    global CACHED_STREAMS, DATA_IN_FLIGHT, DATA_IN_FLIGHT_KEYS, AVAILABLE_READ_AHEAD_SIZE_KB, TOTAL_DATA_IN_FLIGHT_KB
    # Pure based on the delay; agnostic to the device bandwidth limit
    idx = 0
    while idx < len(DATA_IN_FLIGHT):
        DATA_IN_FLIGHT[idx][0] -= time_gap_ms # reduce the delay_time as we go
        if DATA_IN_FLIGHT[idx][0] <= 0:
            # it's time to insert to memory
            lba_key = DATA_IN_FLIGHT[idx][1]
            add_to_cache(lba_key, DATA_IN_FLIGHT[idx][2], DATA_IN_FLIGHT[idx][3], True)
            TOTAL_DATA_IN_FLIGHT_KB -= DATA_IN_FLIGHT[idx][3]
            DATA_IN_FLIGHT_KEYS.remove(lba_key)
            DATA_IN_FLIGHT.pop(idx)
        else:
            idx += 1
    return

def sort_eviction_record(list_vals):
    label = list_vals.pop(0) # get the label
    list_vals.sort()
    return [label] + list_vals 

def add_to_cache(cache_key, io_stream, size_kb, insert_now = False):
    global CACHE_CAPACITY_KB, CACHED_STREAMS, EVICTION_RECORD, N_PREV_TO_CHECK, N_NEXT_TO_CHECK, PER_KB_HIT_RECORDS

    if not insert_now:
        fetch_data_from_hdd(cache_key, io_stream, size_kb) # insert to memory while obeying the timeliness
        return

    # We will add the value to the memory 

    if cache_key in CACHED_STREAMS :
        # the value is already cached
        return

    if CACHE_CAPACITY_KB == -1:
        pass # the cache size is unlimited 
    else:
        # inserting new key
        CACHE_CAPACITY_KB -= size_kb

        if CACHE_CAPACITY_KB >= size_kb:
            pass # We have enough space to cache this stream
        else:
            # evict then add to cache
            while CACHE_CAPACITY_KB < size_kb:
                # evicting the first key in LRU list
                if len(CACHED_STREAMS) == 0:
                    return
                    # print("ERROR: Cache is too small, can't insert the value!")
                    # exit(-1)
                evicted_val = CACHED_STREAMS.popitem(last=False)
                evicted_size = evicted_val[1][0]
                EVICTION_RECORD.append(evicted_val)   # hit count
                CACHE_CAPACITY_KB += evicted_size

    # initialize new metadata
    CACHED_STREAMS[cache_key] = [size_kb, 0, io_stream] # which is [size, init hit count, io streams]
    PER_KB_HIT_RECORDS[cache_key] = [False] * size_kb # initiating the per-kb hit record

def insert_to_stream_metadata(lba_key, lba_suffix):
    global CACHED_STREAMS
    io_stream = CACHED_STREAMS[lba_key][2]
    if len(io_stream) == MAX_QUEUE_SIZE:
        # pop first
        io_stream.pop(0)
    # insert
    io_stream.append(lba_suffix) 
    return io_stream

def reconstruct_address(prefix_lba, suffix_lba):
    pred_addrs = (prefix_lba * PREFIX_LBA_DIVISOR) + suffix_lba
    return int(pred_addrs // SIZE_PER_KEY)

# Helper function
def boyer_moore(queue):
    candidate = None
    count = 0

    for _, delta in queue:
        if count == 0:
            candidate = delta
            count = 1
        elif candidate == delta:
            count += 1
        else:
            count -= 1

    # Verify if the candidate is the majority element
    count = 0
    for _, delta in queue:
        if delta == candidate:
            count += 1

    if count > len(queue) // 2:
        return candidate
    else:
        return None

# 1 - find trend
def get_lba_to_prefetch(queue):
    window_size = 4
    while window_size <= len(queue):
        majority_delta = boyer_moore(queue[len(queue) - window_size:])
        if majority_delta != None:
            lba_to_prefetch = queue[-1][0] + majority_delta
            return lba_to_prefetch, majority_delta
        window_size *= 2
    return None, None

def get_historical_deltas(queue):
    deltas = []
    for idx in range(len(queue)):
        deltas.append(queue[idx][1])
    return deltas

def get_historical_sizes(queue):
    sizes = []
    for idx in range(len(queue)):
        sizes.append(queue[idx][2])
    return sizes


def lstm_prefetcher(io_queue): # 32 historical IOs
    # Will trigger LSTM inference
    last_lba = io_queue[-1][0]
    historical_deltas = get_historical_deltas(io_queue)
    historical_sizes = get_historical_sizes(io_queue)
    # print("last_lba", last_lba)
    # print("hitorical_deltas", historical_deltas)
    arr_delta, arr_size = lstm_wrapper.predict_delta_and_size(historical_deltas, historical_sizes)
    arr_lba = lstm_wrapper.get_list_lba(last_lba, arr_delta, arr_size)

    for offset_to_prefetch, curr_size in arr_lba:
        # Prefetch this offset
        lba_to_prefetch = offset_to_prefetch // SIZE_PER_KEY
        while curr_size > 0:
            # We may prefetch multiple keys if the size is bigger than a single SIZE_PER_KEY
            # print("lba_to_prefetch", lba_to_prefetch, "size", curr_size)
            add_to_cache(lba_to_prefetch, [], SIZE_PER_KEY)
            curr_size -= SIZE_PER_KEY
            lba_to_prefetch += 1

# Time gap is the gap between the current IO vs the previous one
def check_cached_streams(lba_key, time_gap_ms):
    global CACHED_STREAMS, MAJORITY_THRESHOLD, N_HIT_FURTHER_PREFETCH, DOUBLE_PREFETCH, PAUSE_PREFETCH, DATA_IN_FLIGHT_KEYS

    # Insert the pending (in-flight) data to memory
    insert_pending_data(time_gap_ms)

    # Check hit/miss
    if lba_key in CACHED_STREAMS:
        # a hit
        CACHED_STREAMS[lba_key][1] += 1 # incrementing the hit count metadata

        # Update position of the hit item to first. 
        CACHED_STREAMS.move_to_end(lba_key, last=True)
        return 1
    else: # a miss 
        return 0

def update_memory_util(cache_key, start_kb, end_kb):
    # print("cache_key", cache_key)
    global PER_KB_HIT_RECORDS
    for idx in range(int(start_kb), int(end_kb) + 1):
        PER_KB_HIT_RECORDS[cache_key][idx] = True

def calc_mem_utilization():
    global PER_KB_HIT_RECORDS
    utilized_mem_kb, total_mem_kb = 0, 0.1 # to avoid division by zero
    for cache_key in PER_KB_HIT_RECORDS:
        total_mem_kb += SIZE_PER_KEY
        utilized_mem_kb += sum(PER_KB_HIT_RECORDS[cache_key])
    return utilized_mem_kb, total_mem_kb

def init_vars(cache_size, input_path):
    global CACHED_STREAMS, CACHE_CAPACITY_KB, PER_KB_HIT_RECORDS, EVICTION_RECORD, DATA_IN_FLIGHT, DATA_IN_FLIGHT_KEYS, AVAILABLE_READ_AHEAD_SIZE_KB, TOTAL_DATA_IN_FLIGHT_KB, MAX_KB_PREFETCHED_PER_MS, MAX_PREFETCH_BANDWDTH_MBPS, DOUBLE_PREFETCH, PAUSE_PREFETCH

    CACHED_STREAMS = collections.OrderedDict()
    CACHE_CAPACITY_KB = cache_size  # -1 is to mark unlimited memory
    EVICTION_RECORD = ["(lba_key, [size, hit_count, io_stream] )"]
    DATA_IN_FLIGHT = []
    DATA_IN_FLIGHT_KEYS = set()
    PER_KB_HIT_RECORDS = {}
    AVAILABLE_READ_AHEAD_SIZE_KB = 1 * SIZE_PER_KEY # this will limit our prefetching aggresiveness
    TOTAL_DATA_IN_FLIGHT_KB = 0 # KB
    DOUBLE_PREFETCH = False
    PAUSE_PREFETCH = False

    # set the bandwidth limit of the current workload
    device_bandwidth_limit = -1
    min_bandwidth = 20 # MBps ; Some workloads are too slow, but the HDD is fast enough to prefetch upto 20 MBps
    stats_file = str(Path(input_path).parent) + "/read_io.stats"
    if os.path.exists(stats_file):
        with open(stats_file, "r") as f:
            for line in f:
                # if "Bandwidth P80" in line:
                if "Average bandwidth" in line:
                    device_bandwidth_limit = float(line.split("=")[1].strip().split(" ")[0]) # MBps
            if device_bandwidth_limit == -1:
                print("ERROR: Can't find the device bandwidth in the stats file (" + stats_file + ")")
                exit(-1)
    else:
        print("ERROR: Can't find the stats file (" + stats_file + ")")
        exit(-1)
    
    # avg_bandwidth = 50 # overwrite the avg_bandwidth
    MAX_PREFETCH_BANDWDTH_MBPS = max(min_bandwidth, device_bandwidth_limit) # MBps
    MAX_KB_PREFETCHED_PER_MS = MAX_PREFETCH_BANDWDTH_MBPS # KB per milisecond

def start_processing(memory_size, cache_size, input_path, df_raw): 
    global CACHED_STREAMS, CACHE_CAPACITY_KB, MAX_QUEUE_SIZE, SIZE_PER_KEY, EVICTION_RECORD, HDD_FETCH_LATENCY, N_PREV_TO_CHECK, N_NEXT_TO_CHECK, AVAILABLE_READ_AHEAD_SIZE_KB, DOUBLE_PREFETCH, PAUSE_PREFETCH

    # reset index of the dataframe
    df = pd.DataFrame(df_raw)
    df.reset_index(drop=True, inplace=True)

    # init cache 
    init_vars(cache_size, input_path)

    str_stats = []
    
    str_stats.append("Memory size = " + str(memory_size.upper()))
    str_stats.append("Cache capacity = " + str(cache_size) + " KB")
    
    # Create Ouput dir
    cur_script_name = os.path.basename(__file__)
    cur_script_name = str(Path(cur_script_name).with_suffix('') ) # remove .trace extension
    parent_dir = str(Path(input_path).parent)
    output_dir = ""

    if HDD_FETCH_LATENCY == 0:
        output_dir = parent_dir + "/" + "prefetch." + memory_size + "." + cur_script_name
    else:
        # The experiment obey timeliness of the prefetched data
        output_dir = parent_dir + "/" + "prefetch.timely_" + str(HDD_FETCH_LATENCY) + "." + memory_size + "." + cur_script_name
        
    create_output_dir(output_dir)
    print("Output dir = " + output_dir)


    # Calculate time_gap to evaluate the timeliness
    df['time_gap'] = df["ts_record"].diff()
    df.loc[0, 'time_gap'] = 0 # first time_gap is just 0
    df['time_gap'] = df['time_gap'].astype(int)

# Simulate IO submission 
    # the start offset and end offset are separated into two different array
    io_queue = []   # each  item has [prefix, suffix]
    arr_record_hit = []
    total_requested_data_kb = 0
    total_data_found_in_mem_kb = 0
    workload_duration_ms = 0   # how long does it take to submit and process all I/Os in this workload
    last_20_hit_record = [0] * 20 # to calculate the hit rate
    val_20_hit = 0
    prev_offset = df.loc[0, 'offset'] / 1024 # KB

    # for idx, io in df.head(100).iterrows():
    for idx, io in df.iloc[1:].iterrows():
        # if idx > 30:
        #     exit(1)
        if (idx % 10 == 0): # type: ignore
            # print(val_20_hit)
            DOUBLE_PREFETCH = True if val_20_hit >= 8 else False
            PAUSE_PREFETCH = False
            # PAUSE_PREFETCH = True if val_20_hit <= 2 else False

        if (idx % 1000 == 0): # type: ignore
            print(".", end='', flush=True)
            
        curr_offset_kb = int(io["offset"] / 1024)
        is_full_hit = True
        total_io_size = io["size"] / 1024 # in KB
        total_requested_data_kb += total_io_size
        start_kb = curr_offset_kb % SIZE_PER_KEY
        time_gap = io["time_gap"]

        delta = prev_offset - curr_offset_kb # This is the delta between two consecutive IOs
        io_queue.append([curr_offset_kb, delta, total_io_size]) # push

        if len(io_queue) == MAX_HIST_IO_BUFFER:
            #  LSTM prefetching
            lstm_prefetcher(io_queue)

            # Flush the queue so that we don't prefetch on every single IO
            io_queue = []
            # io_queue.pop(0) # pop

        prev_offset = curr_offset_kb

        # list all requested LBAs
        while (total_io_size > 0) :
            lba_key = curr_offset_kb // SIZE_PER_KEY
            
            end_kb = start_kb + total_io_size  # to precisely record per-kb hit
            if end_kb > SIZE_PER_KEY - 1:
                end_kb = SIZE_PER_KEY - 1
            requested_size_kb = end_kb - start_kb

            # fetch the requested lba from disk/cache
            a_hit = check_cached_streams(lba_key, time_gap)

            # partial hit is a miss
            if a_hit == 0:
                is_full_hit = False
            else:
                total_data_found_in_mem_kb += requested_size_kb
                # update per-kb hit record 
                update_memory_util(lba_key, start_kb, end_kb)

            workload_duration_ms += time_gap
            start_kb = 0
            total_io_size -= requested_size_kb
            curr_offset_kb += SIZE_PER_KEY # next offset_kb to look for
            time_gap = 1 # this is for the next sub request, if the IO size is so big

        # update hit record
        arr_record_hit.append(is_full_hit)
        val_20_hit -= last_20_hit_record.pop(0)
        last_20_hit_record.append(is_full_hit)
        if is_full_hit:
            val_20_hit += 1

        assert(len(io_queue) <= MAX_HIST_IO_BUFFER)
    
    print(".")
    
    # Empty out the cache to get the eviction record 
    while len(CACHED_STREAMS) != 0:
        evicted_val = CACHED_STREAMS.popitem(last=False)
        EVICTION_RECORD.append(evicted_val)   # type: ignore # hit count

    # collect and write stats
    n_hit = arr_record_hit.count(1)
    hit_rate = float("{:.2f}".format(n_hit/len(arr_record_hit)*(100)))
    workload_duration_s = workload_duration_ms/1000
    print("Hit rate = " + str(hit_rate) + " %")
    str_stats.append("Size per key = " + str(SIZE_PER_KEY) + " KB")
    str_stats.append("Hit rate = " + str(hit_rate) + " %")
    str_stats.append("Total read IO = " + str(len(arr_record_hit)))
    str_stats.append("Workload duration = " + "{:.2f}".format(workload_duration_s/60) + " mins")
    str_stats.append("IOPS = " + "{:.2f}".format(len(arr_record_hit)/workload_duration_s) + " IO per second")
    str_stats.append("Total hit = " + str(n_hit))
    str_stats.append("Total miss = " + str(len(arr_record_hit) - n_hit))

    # gather the memory utilization
    utilized_mem_kb, total_mem_kb = calc_mem_utilization()
    mem_utilization = float("{:.2f}".format(utilized_mem_kb / total_mem_kb *(100))) 
    total_prefetched_data_gb = float("{:.2f}".format(total_mem_kb / 1024 / 1024)) 
    total_requested_data_gb = float("{:.2f}".format(total_requested_data_kb / 1024 / 1024)) 
    total_data_found_in_mem_mb = float("{:.2f}".format(total_data_found_in_mem_kb / 1024 )) 
    percent_data_found_in_mem = float("{:.2f}".format((total_data_found_in_mem_mb / (total_requested_data_kb / 1024)) * 100 )) + 0.01 # +1 to avoid division by zero
    total_read_from_hdd_gb = float("{:.2f}".format(total_prefetched_data_gb + (100 - percent_data_found_in_mem)/100 * total_requested_data_gb ))
    prefetch_overhead_by_percent = ((total_read_from_hdd_gb - total_requested_data_gb) / total_requested_data_gb) * 100
    prefetch_overhead_by_percent = float("{:.2f}".format(prefetch_overhead_by_percent)) 
    overhead_rate = float("{:.2f}".format(prefetch_overhead_by_percent / percent_data_found_in_mem))

    print("Total requested data = " + str(total_requested_data_gb) + " GB")
    print("Total prefetched data = " + str(total_prefetched_data_gb) + " GB")
    print("Memory utilization = " + str(mem_utilization) + " %")
    print("Total data found in memory = " + str(total_data_found_in_mem_mb) + " MB")
    print("Total data read from HDD = " + str(total_read_from_hdd_gb) + " GB")
    print("Percent data found in memory = " + str(percent_data_found_in_mem) + " %")
    print("Prefetching overhead by percent (extra HDD bandwidth) = " + str(prefetch_overhead_by_percent) + " %")
    print("MAX_PREFETCH_BANDWDTH_MBPS = MAX_KB_PREFETCHED_PER_MS = " + str(MAX_KB_PREFETCHED_PER_MS) + " MBps")
    print("Final value AVAILABLE_READ_AHEAD_SIZE_KB = " + str(AVAILABLE_READ_AHEAD_SIZE_KB))
    print("Final value TOTAL_DATA_IN_FLIGHT_KB = " + str(TOTAL_DATA_IN_FLIGHT_KB))
    print("Overhead rate (overhead/hit rate by bytes) = " + str(overhead_rate))
    
    str_stats.append("Total requested data = " + str(total_requested_data_gb) + " GB")
    str_stats.append("Total prefetched data = " + str(total_prefetched_data_gb) + " GB")
    str_stats.append("Memory utilization = " + str(mem_utilization) + " %")
    str_stats.append("Total data found in memory = " + str(total_data_found_in_mem_mb) + " MB")
    str_stats.append("Total data read from HDD = " + str(total_read_from_hdd_gb) + " GB")
    str_stats.append("Percent data found in memory = " + str(percent_data_found_in_mem) + " %")
    str_stats.append("Prefetching overhead by percent (extra HDD bandwidth) = " + str(prefetch_overhead_by_percent) + " %")
    str_stats.append("HDD read latency (not being used; we use the bandwidth limit)= " + str(HDD_FETCH_LATENCY) + " ms")
    str_stats.append("N previous keys to check = " + str(N_PREV_TO_CHECK) + " keys")
    str_stats.append("N next keys to check = " + str(N_NEXT_TO_CHECK) + " keys")
    str_stats.append("MAX_PREFETCH_BANDWDTH_MBPS = MAX_KB_PREFETCHED_PER_MS = " + str(MAX_KB_PREFETCHED_PER_MS) + " MBps")
    str_stats.append("Final value AVAILABLE_READ_AHEAD_SIZE_KB = " + str(AVAILABLE_READ_AHEAD_SIZE_KB))
    str_stats.append("Final value TOTAL_DATA_IN_FLIGHT_KB = " + str(TOTAL_DATA_IN_FLIGHT_KB))
    str_stats.append("Overhead rate (overhead/hit rate by bytes) = " + str(overhead_rate))
    
    output_file = output_dir + "/hit_rate.stats"
    write_stats(str_stats, output_file)

    # write hit trace to a file 
    output_file = output_dir + "/hit_or_miss.txt"
    write_stats(arr_record_hit, output_file)

    # write the eviction record to file
    output_file = output_dir + "/eviction_record.txt"
    write_stats(sort_eviction_record(EVICTION_RECORD), output_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-file", help="File path of the trace profiles",type=str)
    parser.add_argument("-files", help="Arr of file path of the trace profiles", nargs='+',type=str)
    parser.add_argument("-size", help="size of the available memory in GB",type=str, default="unlimited")
    args = parser.parse_args()
    if not args.size or (not args.file and not args.files):
        print("    ERROR: You must provide these arguments: -size <memory size as xxxGB> -file <the input trace> ")
        exit(-1)

    memory_size = "unlimited"    # sign of unlimited memory size
    cache_size = 0

    # check the memory size
    if args.size != "unlimited":
        memory_size = args.size.lower() # for file naming
        assert ("gb" in memory_size) or ("mb" in memory_size) or ("kb" in memory_size)

        if "gb" in memory_size:
            cache_size = int(float(memory_size.replace("gb",""))*1024 *1024 ) # KB
        elif "mb" in memory_size:
            cache_size = int(float(memory_size.replace("mb","")) *1024 ) # KB
        elif "kb" in memory_size:
            cache_size = int(float(memory_size.replace("kb",""))) # KB
    elif args.size == "unlimited":
        cache_size = -1 # sign of unlimited memory size
    else:
        print("ERROR: Unrecognized memory size (" + args.size + "). Please use kb, mb, gb, or unlimited.")
        exit(-1)
        
    print("Memory size = " + str(args.size))
    print("Cache size = " + str(cache_size) + " KB")
    print("Size per key = " + str(SIZE_PER_KEY) + " KB")

    trace_files = []
    if args.files:
        trace_files += args.files
    elif args.file:
        trace_files.append(args.file)
    print("trace_files = " + str(trace_files))

    for trace_path in trace_files:
        print("\nProcessing " + str(trace_path))
        print("Start training the model")
        # get the full path
        df = read_file(trace_path)
        # split 50:50 for training and simulating real prefetching
        df_training, df_testing = train_test_split(df, test_size=0.5, shuffle=False)
        
        # Training: Convert offset to KB and size to KB
        df_training = pd.DataFrame(df_training)
        df_training['offset'] = df_training['offset'] // 1024
        df_training['size'] = df_training['size'] // 1024

        model = lstm_wrapper.train_model(df_training)
        print("Finish training the model")
        lstm_wrapper.set_model_globally(model)

        # use the last 50% for prefetching testing
        print("Start testing and simulating the online prefetching")

        # Testing: Will convert to KB inside the start_processing function
        start_processing(memory_size, cache_size, trace_path, df_testing)
