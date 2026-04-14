# SPECTRAL GCN
```
spectral_wrapper: dataset/MSR-Cambridge/hm_1.csv.gz

Reading trace: dataset/MSR-Cambridge/hm_1.csv.gz
Length of trace: 609311
 train: 304656, test: 304654

=== Start training → checkpoint/spectral_hm_1.csv.gz_2026_04_06_02_32_28
===== epoch: 0
2380it [00:36, 65.46it/s]
        Total loss: 8185.421
===== epoch: 1
2380it [00:37, 64.00it/s]
        Total loss: 7855.761
===== epoch: 2
2380it [00:36, 64.43it/s]
        Total loss: 7723.727
===== epoch: 3
2380it [00:39, 59.93it/s]
        Total loss: 7631.752
===== epoch: 4
2380it [00:41, 57.18it/s]
        Total loss: 7556.697
===== epoch: 5
2380it [00:41, 57.16it/s]
        Total loss: 7487.516
===== epoch: 6
2380it [00:41, 57.60it/s]
        Total loss: 7424.672
===== epoch: 7
2380it [00:41, 57.51it/s]
        Total loss: 7366.154
===== epoch: 8
2380it [00:41, 57.26it/s]
        Total loss: 7314.224
===== epoch: 9
2380it [00:41, 57.86it/s]
        Total loss: 7263.147

Evaluating 304621 test cases...
  Building graph features...
  Graph build: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████| 304621/304621 [01:23<00:00, 3637.99it/s]
  Running batched inference...
  Inference: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 595/595 [00:09<00:00, 60.00it/s]

============================================================
  SPECTRAL PREFETCHER — EVALUATION RESULTS
  Dataset : dataset/MSR-Cambridge/hm_1.csv.gz
============================================================
  Hit Rate                  : 98.89%
  Prefetch Hit Rate         : 2.96%
  Access Speed (SSD)        : 900180.3 req/s
  Access Speed (HDD)        : 4500.9 req/s
  Prefetch Effectiveness    : 2.96%
  Prefetch Overhead         : 0.35%
  Total IOs                 : 304,621
  Total Prefetches Issued   : 1,113
  Cache Hits                : 301,237
  Prefetch Hits             : 33
  Cache Misses              : 3,384
  Inference Time (total)    : 93.92s
  Inference Latency (per)   : 0.308ms
  Throughput                : 3243.5 inf/s
  Model Size (# params)     : 286,350
============================================================

Number of test IOs : 304621
Sample prefetch addresses: [465104, 387484, 363596, 7224, 377416, 392948, 388444, 388884, 388596, 387428]
```

# SGDP

```
Inside graph main, raw_trace dataset/MSR-Cambridge/hm_1.csv.gz

Reading trace:  dataset/MSR-Cambridge/hm_1.csv.gz 

Length of trace 609311
 train_trace  304656 374508 2756
train_data_list len 2380
[np.int64(0), np.int64(0), np.int64(0), np.int64(0), np.int64(0), np.int64(0), np.int64(0), np.int64(0), np.int64(0), np.int64(1), np.int64(7), np.int64(5), np.int64(6), np.int64(3), np.int64(0), np.int64(0), np.int64(0), np.int64(0), np.int64(0), np.int64(0), np.int64(0), np.int64(0), np.int64(0), np.int64(4), np.int64(8), np.int64(0), np.int64(0), np.int64(0), np.int64(0), np.int64(0), np.int64(2), np.int64(9)]

=== Start training, model_path: checkpoint/model_dataset/MSR-Cambridge/hm_1.csv.gz_2026_04_07_11_47_56
===== epoch: 0
start training: 
0it [00:00, ?it/s]/Users/brianjiang/Desktop/gnn_prefetcher/model_collection/sota_graph_sgdp/model.py:144: UserWarning: Creating a tensor from a list of numpy.ndarrays is extremely slow. Please consider converting the list to a single numpy.ndarray with numpy.array() before converting to a tensor. (Triggered internally at /Users/runner/work/pytorch/pytorch/pytorch/torch/csrc/utils/tensor_new.cpp:256.)
  A = trans_to_cuda(torch.Tensor(A).float())
2380it [02:01, 19.55it/s]
/Users/brianjiang/Desktop/gnn_prefetcher/model_collection/sota_graph_sgdp/model.py:127: UserWarning: Converting a tensor with requires_grad=True to a scalar may lead to unexpected behavior.
Consider using tensor.detach() first. (Triggered internally at /Users/runner/work/pytorch/pytorch/pytorch/torch/csrc/autograd/generated/python_variable_methods.cpp:837.)
  print('\tTotal loss: %.3f' % total_loss)
        Total loss: 8052.670
===== epoch: 1
start training: 
2380it [02:05, 18.90it/s]
        Total loss: 7435.974
===== epoch: 2
start training: 
2380it [02:04, 19.15it/s]
        Total loss: 7124.359
===== epoch: 3
start training: 
2380it [02:07, 18.65it/s]
        Total loss: 6936.402
===== epoch: 4
start training: 
2380it [02:10, 18.30it/s]
        Total loss: 6797.062
===== epoch: 5
start training: 
2380it [02:06, 18.82it/s]
        Total loss: 6677.397
===== epoch: 6
start training: 
2380it [02:06, 18.82it/s]
        Total loss: 6566.467
===== epoch: 7
start training: 
2380it [02:08, 18.56it/s]
        Total loss: 6460.562
===== epoch: 8
start training: 
2380it [02:10, 18.27it/s]
        Total loss: 6358.694
===== epoch: 9
start training: 
2380it [02:04, 19.08it/s]
        Total loss: 6261.974

Do per inference testing:
     len test_trace 304621
Total IO in the test set  304621

============================================================
  SGDP — EVALUATION RESULTS
  Dataset : dataset/MSR-Cambridge/hm_1.csv.gz
============================================================
  Hit Rate                  : 98.70%
  Prefetch Hit Rate         : 6.23%
  Access Speed (SSD)        : 772169.8 req/s
  Access Speed (HDD)        : 3860.8 req/s
  Prefetch Effectiveness    : 6.23%
  Prefetch Overhead         : 0.65%
  Total IOs                 : 304,621
  Total Prefetches Issued   : 2,118
  Cache Hits                : 300,676
  Prefetch Hits             : 132
  Cache Misses              : 3,945
  Inference Time (total)    : 170.38s
  Inference Latency (per)   : 0.559ms
  Throughput                : 1787.8 inf/s
  Model Size (# params)     : 512,700
============================================================

Number of test IOs: 304621
Sample prefetch addresses: [465104, 387484, 363596, 7224, 377416, 392948, 388444, 388884, 388596, 387428]
```