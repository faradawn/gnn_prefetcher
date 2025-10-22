#!/usr/bin/env python3

# (Faradawn): First, train the model using the .ipynb file.
# Then, import hte model as in below

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

sys.path.append('../../../commonutils')
import pattern_checker

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

def count_and_get_majority(nums):
    num_dict = {}
    count, value = -1, None
    for n in nums:
        if n in num_dict:
            num_dict[n] += 1
            if count < num_dict[n]:
                # Update the count and value 
                count = num_dict[n]
                value = n 
        else:
            num_dict[n] = 1
    return count, value

def count_majority(nums):
    count, value = count_and_get_majority(nums)
    return count

MAX_QUEUE_SIZE = 32
MAJORITY_THRESHOLD = 3
CACHED_STREAMS = collections.OrderedDict() # the value is just the metadata = [size, hit_count, .. ]
CACHE_CAPACITY_KB = 0
SIZE_PER_KEY = -1 #KB
PER_KB_HIT_RECORDS = {}

EVICTION_RECORD = ["(lba_key, [size, hit_count, io_stream] )"]
HDD_FETCH_LATENCY = 20 # ms
DATA_IN_FLIGHT = []


def fetch_data_from_hdd(value, size_kb):
    global DATA_IN_FLIGHT, HDD_FETCH_LATENCY
    DATA_IN_FLIGHT.append([HDD_FETCH_LATENCY, value, size_kb]) # [time_delay, value, size]

def insert_pending_data(time_gap_ms):
    global CACHED_STREAMS, DATA_IN_FLIGHT
    idx = 0
    while idx < len(DATA_IN_FLIGHT):
        DATA_IN_FLIGHT[idx][0] -= time_gap_ms # reduce the delay_time as we go
        if DATA_IN_FLIGHT[idx][0] <= 0:
            # it's time to insert to memory
            add_to_cache(DATA_IN_FLIGHT[idx][1], DATA_IN_FLIGHT[idx][2], True)
            DATA_IN_FLIGHT.pop(idx)
        else:
            idx += 1

def sort_eviction_record(list_vals):
    label = list_vals.pop(0) # get the label
    list_vals.sort()
    return [label] + list_vals 

def add_to_cache(value, size_kb, now = False):
    global SIZE_PER_KEY, CACHE_CAPACITY_KB, CACHED_STREAMS, EVICTION_RECORD
    if not now:
        fetch_data_from_hdd(value, size_kb) # insert to memory while obeying the timeliness
        return

    if value in CACHED_STREAMS:
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

    CACHED_STREAMS[value] = [size_kb, 0] # which is [size, init hit count]
    PER_KB_HIT_RECORDS[value] = [False] * size_kb # initiating the per-kb hit record

# === (Faradawn LSTM) ===
import keras
import tensorflow as tf
tf.get_logger().setLevel('ERROR')  # Set log level to ERROR

new_model = keras.models.load_model(f'./sota_lstm/lstm_hidden_150_epochs_5.keras')

# Load the four maps 
import pickle
with open('./sota_lstm/maps.pkl', 'rb') as f:
    delta_map, rev_delta_map, size_map, rev_size_map = pickle.load(f)

def do_prefetch(delta_queue, size_queue):
    global MAX_QUEUE_SIZE # this is 32, window size 
    offset_input = np.array(delta_queue).reshape(1, MAX_QUEUE_SIZE) # convert to batch size of 1
    size_input = np.array(size_queue).reshape(1, MAX_QUEUE_SIZE) # convert to batch size of 1
    offset_pred, size_pred = new_model.predict([offset_input, size_input], verbose=0)
    pred_1 = np.argmax(offset_pred[:,0,:], axis=1)[0] # pred_1 is a number, predicted delta_1000_class 
    pred_2 = np.argmax(size_pred[:,0,:], axis=1)[0] # pred_2 is size class
    # print("Pred_1", pred_1, "Pred_2", pred_2)
    if pred_1 == 1000 or (pred_1 not in rev_delta_map) or (pred_2 not in rev_size_map): # no prefetch
        return None, None
    else: # return (delta, size)
        return rev_delta_map[pred_1], rev_size_map[pred_2]
    
    

# Time gap is the gap between the current IO vs the previous one
def check_cached_streams(lba_key, queue, time_gap_ms):
    global CACHED_STREAMS

    # Insert the pending (in-flight) data to memory
    insert_pending_data(time_gap_ms)

    if lba_key in CACHED_STREAMS:
        # a hit
        CACHED_STREAMS[lba_key][1] += 1 # incrementing the hit count metadata
        # NO further prefetch

        # Update position of the hit item to first. 
        CACHED_STREAMS.move_to_end(lba_key, last=True)
        return 1, None
    else: # a miss -> should we fetch the current missing page?
        # if queue == None:
        #     return 0, None
        
        # # kernel read ahead (Faradawn)
        # page_to_fetch, is_consecutive = check_sequential_access(queue)

        # if is_consecutive:
        #     # prefetch this address
        #     add_to_cache(page_to_fetch, SIZE_PER_KEY)
        #     add_to_cache(page_to_fetch + 1, SIZE_PER_KEY)
        #     return 0, page_to_fetch
        return 0, None

def update_memory_util(cache_key, start_kb, end_kb):
    global PER_KB_HIT_RECORDS
    for idx in range(int(start_kb), int(end_kb) + 1):
        PER_KB_HIT_RECORDS[cache_key][idx] = True

def calc_mem_utilization():
    global PER_KB_HIT_RECORDS
    utilized_mem_kb, total_mem_kb = 0, 0.1
    for cache_key in PER_KB_HIT_RECORDS:
        total_mem_kb += SIZE_PER_KEY
        utilized_mem_kb += sum(PER_KB_HIT_RECORDS[cache_key])
    return utilized_mem_kb, total_mem_kb

def init_vars(cache_size, chunk_size):
    global CACHED_STREAMS, CACHE_CAPACITY_KB, SIZE_PER_KEY, PER_KB_HIT_RECORDS, EVICTION_RECORD, DATA_IN_FLIGHT

    CACHED_STREAMS = collections.OrderedDict()
    CACHE_CAPACITY_KB = cache_size  # -1 is to mark unlimited memory
    SIZE_PER_KEY = chunk_size
    EVICTION_RECORD = ["(lba_key, [size, hit_count, io_stream] )"]
    DATA_IN_FLIGHT = []
    PER_KB_HIT_RECORDS = {}

def start_processing(memory_size, cache_size, chunk_size, input_path): 
    global SIZE_PER_KEY, CACHED_STREAMS, CACHE_CAPACITY_KB, MAX_QUEUE_SIZE
    
    # init cache 
    init_vars(cache_size, chunk_size)

    str_stats = []
    df = read_file(input_path)
    
    str_stats.append("Memory size = " + str(memory_size.upper()))
    str_stats.append("Cache capacity = " + str(cache_size) + " KB")
    str_stats.append("Size per key = " + str(SIZE_PER_KEY) + " KB")
    
    # Create Ouput dir
    cur_script_name = os.path.basename(__file__)
    cur_script_name = str(Path(cur_script_name).with_suffix('') ) # remove .trace extension
    parent_dir = str(Path(input_path).parent)
    output_dir = ""
    if HDD_FETCH_LATENCY == 0:
        output_dir = parent_dir + "/" + "prefetch." + memory_size + ".chunk_" + str(SIZE_PER_KEY) + "." + cur_script_name 
    else:
        # The experiment obey timeliness of the prefetched data
        output_dir = parent_dir + "/" + "prefetch.timely_" + str(HDD_FETCH_LATENCY) + "." + memory_size + ".chunk_" + str(SIZE_PER_KEY) + "." + cur_script_name 

    create_output_dir(output_dir)
    print("Output dir = " + output_dir)

    # Calculate time_gap to evaluate the timeliness
    df['time_gap'] = df["ts_record"].diff() # ms 
    df.loc[0, 'time_gap'] = 0 # first time_gap is just 0
    df['time_gap'] = df['time_gap'].astype(int)

    # Include the previous offset on the current IO
    # df["prev_offset"] = df["offset"].shift(-1)

    # print(df.head())
    # exit(0)

# Simulate IO submission 
    # the start offset and end offset are separated into two different array
    io_queue = [] # pieces of IO address
    addr_queue = [] # raw addr
    delta_queue = [] # addr delta
    size_queue = [] # size

    arr_record_hit = []
    total_requested_data_kb = 0
    total_data_found_in_mem_kb = 0

    # for idx, io in df.head(100).iterrows():
    last_offset = 0
    for idx, io in df.head(1000).iterrows():
        if (idx % 1000 == 0):
            print(".", end='', flush=True)

        raw_addr = io["offset"]
        offset_kb = int(io["offset"] / 1024)
        is_full_hit = True
        total_io_size = io["size"] / 1024 # in KB
        total_requested_data_kb += total_io_size

        start_kb = offset_kb % SIZE_PER_KEY
        time_gap = io["time_gap"]

        addr_queue.append(io["offset"]) # (Faradawn) Push raw address and raw size
        size_queue.append(size_map.get(round(np.log2(io["size"])) , 0)) # Get size class
        delta_queue.append(delta_map.get(io["offset"] - last_offset, 1000)) # Get delta class
        last_offset = io["offset"]

        # DO PREFETCH
        if len(delta_queue) == MAX_QUEUE_SIZE:
            # print(f"=== prefetch on idx {idx}")
            delta_pred, size_pred = do_prefetch(delta_queue, size_queue)
            # print(f"final delta {delta_pred}, size {size_pred}")
            if delta_pred is not None:
                add_to_cache(addr_queue[-1] - delta_pred, int(max(0, size_pred - 12)))

        # break a big IO into pieces 
        while (total_io_size > 0) :
            cache_key = offset_kb // SIZE_PER_KEY
            io_queue.append(cache_key) # push
    
            end_kb = start_kb + total_io_size  # to precisely record per-kb hit
            if end_kb > SIZE_PER_KEY - 1:
                end_kb = SIZE_PER_KEY - 1
            requested_size_kb = end_kb - start_kb
            # print(idx, offset_kb, total_io_size, requested_size_kb, start_kb, end_kb )
            
            # fetch the requested lba from disk/cache
            a_hit, recently_cached_key = check_cached_streams(cache_key, io_queue, time_gap)

            if len(addr_queue) == MAX_QUEUE_SIZE:
                io_queue.pop(0) # pop
            
            # partial hit is a miss
            if a_hit == 0:
                is_full_hit = False
            else:
                total_data_found_in_mem_kb += requested_size_kb
                # update per-kb hit record 
                update_memory_util(cache_key, start_kb, end_kb)

            start_kb = 0
            total_io_size -= requested_size_kb
            offset_kb += SIZE_PER_KEY # next offset_kb to look for
            time_gap = 1 # this is for the next sub request, if the IO size is so big

        # Pop one
        if len(delta_queue) == MAX_QUEUE_SIZE:
            addr_queue.pop(0)
            delta_queue.pop(0)
            size_queue.pop(0)

        arr_record_hit.append(is_full_hit)
        # print(total_requested_data_kb, total_data_found_in_mem_kb)
        # assert(total_requested_data_kb > total_data_found_in_mem_kb)
        assert(len(addr_queue) <= MAX_QUEUE_SIZE)
            
    print(".")

    # Empty out the cache to get the eviction record 
    while len(CACHED_STREAMS) != 0:
        evicted_val = CACHED_STREAMS.popitem(last=False)
        EVICTION_RECORD.append(evicted_val)   # hit count

    # collect and write stats
    n_hit = arr_record_hit.count(1)
    hit_rate = float("{:.2f}".format(n_hit/len(arr_record_hit)*(100)))
    print("Hit rate = " + str(hit_rate) + " %")
    str_stats.append("Hit rate = " + str(hit_rate) + " %")
    str_stats.append("Total read IO = " + str(len(arr_record_hit)))
    str_stats.append("Total hit = " + str(n_hit))
    str_stats.append("Total miss = " + str(len(arr_record_hit) - n_hit))

    # gather the memory utilization
    utilized_mem_kb, total_mem_kb = calc_mem_utilization()
    mem_utilization = float("{:.2f}".format(utilized_mem_kb / total_mem_kb *(100))) 
    total_prefetched_data_gb = float("{:.2f}".format(total_mem_kb / 1024 / 1024)) 
    total_requested_data_gb = float("{:.2f}".format(total_requested_data_kb / 1024 / 1024)) 
    total_data_found_in_mem_mb = float("{:.2f}".format(total_data_found_in_mem_kb / 1024 )) 
    percent_data_found_in_mem = float("{:.2f}".format((total_data_found_in_mem_mb / (total_requested_data_kb / 1024)) * 100 )) 
    total_read_from_hdd_gb = float("{:.2f}".format(total_prefetched_data_gb + (100 - percent_data_found_in_mem)/100 * total_requested_data_gb ))
    prefetch_overhead = ((total_read_from_hdd_gb - total_requested_data_gb) / total_requested_data_gb) * 100
    prefetch_overhead = float("{:.2f}".format(prefetch_overhead)) 
    
    print("Total requested data = " + str(total_requested_data_gb) + " GB")
    print("Total prefetched data = " + str(total_prefetched_data_gb) + " GB")
    print("Memory utilization = " + str(mem_utilization) + " %")
    print("Total data found in memory = " + str(total_data_found_in_mem_mb) + " MB")
    print("Total data read from HDD = " + str(total_read_from_hdd_gb) + " GB")
    print("Percent data found in memory = " + str(percent_data_found_in_mem) + " %")
    print("Prefetching overhead (extra HDD bandwidth) = " + str(prefetch_overhead) + " %")
    
    str_stats.append("Total requested data = " + str(total_requested_data_gb) + " GB")
    str_stats.append("Total prefetched data = " + str(total_prefetched_data_gb) + " GB")
    str_stats.append("Memory utilization = " + str(mem_utilization) + " %")
    str_stats.append("Total data found in memory = " + str(total_data_found_in_mem_mb) + " MB")
    str_stats.append("Total data read from HDD = " + str(total_read_from_hdd_gb) + " GB")
    str_stats.append("Percent data found in memory = " + str(percent_data_found_in_mem) + " %")
    str_stats.append("Prefetching overhead (extra HDD bandwidth) = " + str(prefetch_overhead) + " %")
    str_stats.append("HDD read latency = " + str(HDD_FETCH_LATENCY) + " ms")

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
    parser.add_argument("-chunk", help="The size of a single cached data",type=str)
    args = parser.parse_args()
    if not args.chunk or not args.size or (not args.file and not args.files):
        print("    ERROR: You must provide these arguments: -chunk <cached data size as xxxKB> -size <memory size as xxxGB> -file <the input trace> ")
        exit(-1)

    # check chunk size 
    if "gb" in args.chunk:
        chunk_size = int(float(args.chunk.replace("gb",""))*1024 *1024 ) # KB
    elif "mb" in args.chunk:
        chunk_size = int(float(args.chunk.replace("mb","")) *1024 ) # KB
    elif "kb" in args.chunk:
        chunk_size = int(float(args.chunk.replace("kb",""))) # KB

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
    print("Size per key = " + str(chunk_size) + " KB")

    trace_files = []
    if args.files:
        trace_files += args.files
    elif args.file:
        trace_files.append(args.file)
    print("trace_files = " + str(trace_files))
    
    for trace_path in trace_files:
        print("\nProcessing " + str(trace_path))
        start_processing(memory_size, cache_size, chunk_size, trace_path)

# How to run:
# cd /home/daniar/flashnet/model_collection/5_block_prefetching/simulate_basic_prefetcher/
# ./base_prefetcher.lstm.py -chunk 256kb -file ../dataset/iotrace/alibaba.cut.per_10k.rw_80_20.723/read_io.trace
