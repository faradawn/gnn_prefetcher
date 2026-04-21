# SPECTRAL GCN
```
spectral_wrapper: dataset/MSR-Cambridge/hm_1.csv.gz

Reading trace: dataset/MSR-Cambridge/hm_1.csv.gz
Length of trace: 609311
 train: 548380, test: 60930

=== Start training → checkpoint/spectral_hm_1.csv.gz_2026_04_21_10_35_47
===== epoch: 0
4284it [00:52, 81.51it/s]
        Total loss: 12978.495
===== epoch: 1
4284it [00:52, 81.62it/s]
        Total loss: 12648.264
===== epoch: 2
4284it [00:52, 81.52it/s]
        Total loss: 12560.084
===== epoch: 3
4284it [00:53, 80.08it/s]
        Total loss: 12494.909
===== epoch: 4
4284it [00:52, 81.42it/s]
        Total loss: 12439.905
===== epoch: 5
4284it [00:52, 81.75it/s]
        Total loss: 12397.121
===== epoch: 6
4284it [00:52, 81.93it/s]
        Total loss: 12361.669
===== epoch: 7
4284it [00:52, 82.02it/s]
        Total loss: 12332.182
===== epoch: 8
4284it [00:52, 81.96it/s]
        Total loss: 12306.405
===== epoch: 9
4284it [00:52, 82.01it/s]
        Total loss: 12281.105

Evaluating 60897 test cases...
  Building graph features...
  Graph build: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 60897/60897 [00:18<00:00, 3278.62it/s]
  Running batched inference...
  Inference: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 119/119 [00:01<00:00, 60.59it/s]

============================================================
  SPECTRAL PREFETCHER — EVALUATION RESULTS
  Dataset : dataset/MSR-Cambridge/hm_1.csv.gz
============================================================
  Hit Rate                  : 96.38%
  Prefetch Hit Rate         : 22.64%
  Access Speed (SSD)        : 276553.1 req/s
  Access Speed (HDD)        : 1382.8 req/s
  Prefetch Effectiveness    : 22.64%
  Prefetch Overhead         : 0.07%
  Total IOs                 : 60,897
  Total Prefetches Issued   : 53
  Cache Hits                : 58,695
  Prefetch Hits             : 12
  Cache Misses              : 2,202
  Inference Time (total)    : 20.57s
  Inference Latency (per)   : 0.338ms
  Throughput                : 2960.4 inf/s
  Model Size (# params)     : 155,096
============================================================

Number of test IOs : 60897
Sample prefetch addresses: [2652, 1452, 1740, 1468, 4788, 384624, 2332, 369900, 354096, 2636]
```

# SGDP

```
Inside graph main, raw_trace dataset/MSR-Cambridge/hm_1.csv.gz

Reading trace: dataset/MSR-Cambridge/hm_1.csv.gz
Length of trace: 609311
 train: 548380, test: 60930
train_data_list len 4284

=== Start training, model_path: checkpoint/model_dataset/MSR-Cambridge/hm_1.csv.gz_2026_04_21_11_23_09

===== epoch: 0
4284it [01:41, 42.30it/s]
        Total loss: 12989.433

  Cache evaluation (epoch 0):
  cache=    5  HR=0.0099  EPR=0.5118
  cache=   10  HR=0.0193  EPR=0.6086
  cache=   20  HR=0.0453  EPR=0.6775
  cache=   30  HR=0.0809  EPR=0.6705
  cache=   40  HR=0.1176  EPR=0.6465
  cache=   50  HR=0.1510  EPR=0.6300
  cache=   60  HR=0.1840  EPR=0.6196
  cache=   70  HR=0.2189  EPR=0.5943
  cache=   80  HR=0.2536  EPR=0.5560
  cache=   90  HR=0.2837  EPR=0.5331
  cache=  100  HR=0.3080  EPR=0.4753
  cache=  200  HR=0.4920  EPR=0.3250
  cache=  300  HR=0.6254  EPR=0.2935
  cache=  400  HR=0.9112  EPR=0.2667
  cache=  500  HR=0.9515  EPR=0.2500
  cache=  600  HR=0.9573  EPR=0.2424
  cache=  700  HR=0.9628  EPR=0.2500
  cache=  800  HR=0.9631  EPR=0.2881
  cache=  900  HR=0.9635  EPR=0.2881
  cache= 1000  HR=0.9644  EPR=0.2881
  Precision@k: ['0.5914', '0.6104', '0.6243', '0.6352', '0.6436', '0.6510', '0.6571', '0.6633', '0.6680', '0.6721', '0.6760', '0.6800', '0.6832', '0.6863', '0.6890', '0.6917', '0.6940', '0.6966', '0.6989', '0.7010']
  MRR@k      : ['0.5914', '0.6009', '0.6055', '0.6083', '0.6099', '0.6112', '0.6120', '0.6128', '0.6133', '0.6138', '0.6141', '0.6144', '0.6147', '0.6149', '0.6151', '0.6153', '0.6154', '0.6155', '0.6157', '0.6158']

===== epoch: 1
4284it [01:40, 42.79it/s]
        Total loss: 12048.706

  Cache evaluation (epoch 1):
  cache=    5  HR=0.0167  EPR=0.5651
  cache=   10  HR=0.0272  EPR=0.7123
  cache=   20  HR=0.0534  EPR=0.8000
  cache=   30  HR=0.0879  EPR=0.8077
  cache=   40  HR=0.1247  EPR=0.8066
  cache=   50  HR=0.1572  EPR=0.7977
  cache=   60  HR=0.1895  EPR=0.7836
  cache=   70  HR=0.2237  EPR=0.7648
  cache=   80  HR=0.2579  EPR=0.7356
  cache=   90  HR=0.2874  EPR=0.7109
  cache=  100  HR=0.3111  EPR=0.6790
  cache=  200  HR=0.4933  EPR=0.4874
  cache=  300  HR=0.6250  EPR=0.3419
  cache=  400  HR=0.9111  EPR=0.1589
  cache=  500  HR=0.9516  EPR=0.1340
  cache=  600  HR=0.9572  EPR=0.1290
  cache=  700  HR=0.9626  EPR=0.1333
  cache=  800  HR=0.9630  EPR=0.1500
  cache=  900  HR=0.9634  EPR=0.1500
  cache= 1000  HR=0.9643  EPR=0.1500
  Precision@k: ['0.5929', '0.6206', '0.6370', '0.6491', '0.6593', '0.6676', '0.6757', '0.6831', '0.6890', '0.6940', '0.6985', '0.7028', '0.7068', '0.7100', '0.7137', '0.7165', '0.7195', '0.7223', '0.7253', '0.7282']
  MRR@k      : ['0.5929', '0.6068', '0.6122', '0.6152', '0.6173', '0.6187', '0.6198', '0.6208', '0.6214', '0.6219', '0.6223', '0.6227', '0.6230', '0.6232', '0.6235', '0.6236', '0.6238', '0.6240', '0.6241', '0.6243']

===== epoch: 2
4284it [01:40, 42.48it/s]
        Total loss: 11626.843

  Cache evaluation (epoch 2):
  cache=    5  HR=0.0199  EPR=0.5491
  cache=   10  HR=0.0309  EPR=0.7060
  cache=   20  HR=0.0572  EPR=0.7939
  cache=   30  HR=0.0915  EPR=0.8043
  cache=   40  HR=0.1280  EPR=0.8034
  cache=   50  HR=0.1604  EPR=0.8019
  cache=   60  HR=0.1923  EPR=0.7882
  cache=   70  HR=0.2261  EPR=0.7760
  cache=   80  HR=0.2597  EPR=0.7527
  cache=   90  HR=0.2889  EPR=0.7350
  cache=  100  HR=0.3126  EPR=0.7148
  cache=  200  HR=0.4937  EPR=0.5132
  cache=  300  HR=0.6254  EPR=0.3724
  cache=  400  HR=0.9112  EPR=0.2086
  cache=  500  HR=0.9517  EPR=0.1885
  cache=  600  HR=0.9573  EPR=0.1966
  cache=  700  HR=0.9624  EPR=0.1770
  cache=  800  HR=0.9631  EPR=0.1923
  cache=  900  HR=0.9635  EPR=0.1923
  cache= 1000  HR=0.9645  EPR=0.1923
  Precision@k: ['0.5936', '0.6241', '0.6422', '0.6549', '0.6651', '0.6747', '0.6835', '0.6913', '0.6974', '0.7033', '0.7088', '0.7136', '0.7182', '0.7219', '0.7251', '0.7281', '0.7311', '0.7341', '0.7368', '0.7391']
  MRR@k      : ['0.5936', '0.6088', '0.6149', '0.6181', '0.6201', '0.6217', '0.6230', '0.6239', '0.6246', '0.6252', '0.6257', '0.6261', '0.6265', '0.6267', '0.6269', '0.6271', '0.6273', '0.6275', '0.6276', '0.6277']

===== epoch: 3
4284it [01:40, 42.51it/s]
        Total loss: 11236.887

  Cache evaluation (epoch 3):
  cache=    5  HR=0.0231  EPR=0.5749
  cache=   10  HR=0.0341  EPR=0.7161
  cache=   20  HR=0.0605  EPR=0.8019
  cache=   30  HR=0.0949  EPR=0.8095
  cache=   40  HR=0.1310  EPR=0.8051
  cache=   50  HR=0.1629  EPR=0.8016
  cache=   60  HR=0.1942  EPR=0.7890
  cache=   70  HR=0.2283  EPR=0.7753
  cache=   80  HR=0.2614  EPR=0.7481
  cache=   90  HR=0.2904  EPR=0.7319
  cache=  100  HR=0.3140  EPR=0.7164
  cache=  200  HR=0.4937  EPR=0.5154
  cache=  300  HR=0.6260  EPR=0.3852
  cache=  400  HR=0.9112  EPR=0.2338
  cache=  500  HR=0.9517  EPR=0.1944
  cache=  600  HR=0.9574  EPR=0.1886
  cache=  700  HR=0.9623  EPR=0.1726
  cache=  800  HR=0.9633  EPR=0.1887
  cache=  900  HR=0.9637  EPR=0.1911
  cache= 1000  HR=0.9646  EPR=0.1911
  Precision@k: ['0.5945', '0.6263', '0.6444', '0.6577', '0.6687', '0.6778', '0.6868', '0.6947', '0.7008', '0.7066', '0.7118', '0.7165', '0.7214', '0.7249', '0.7285', '0.7320', '0.7351', '0.7384', '0.7412', '0.7439']
  MRR@k      : ['0.5945', '0.6104', '0.6164', '0.6197', '0.6219', '0.6235', '0.6248', '0.6257', '0.6264', '0.6270', '0.6275', '0.6279', '0.6282', '0.6285', '0.6287', '0.6290', '0.6291', '0.6293', '0.6295', '0.6296']

===== epoch: 4
4284it [01:41, 42.09it/s]
        Total loss: 11165.905

  Cache evaluation (epoch 4):
  cache=    5  HR=0.0235  EPR=0.5771
  cache=   10  HR=0.0347  EPR=0.7228
  cache=   20  HR=0.0611  EPR=0.8065
  cache=   30  HR=0.0954  EPR=0.8163
  cache=   40  HR=0.1314  EPR=0.8109
  cache=   50  HR=0.1634  EPR=0.8110
  cache=   60  HR=0.1947  EPR=0.7966
  cache=   70  HR=0.2284  EPR=0.7828
  cache=   80  HR=0.2615  EPR=0.7563
  cache=   90  HR=0.2905  EPR=0.7392
  cache=  100  HR=0.3141  EPR=0.7254
  cache=  200  HR=0.4940  EPR=0.5294
  cache=  300  HR=0.6262  EPR=0.4030
  cache=  400  HR=0.9113  EPR=0.2513
  cache=  500  HR=0.9518  EPR=0.2123
  cache=  600  HR=0.9575  EPR=0.2069
  cache=  700  HR=0.9624  EPR=0.1964
  cache=  800  HR=0.9633  EPR=0.2138
  cache=  900  HR=0.9638  EPR=0.2179
  cache= 1000  HR=0.9647  EPR=0.2179
  Precision@k: ['0.5949', '0.6267', '0.6451', '0.6581', '0.6692', '0.6792', '0.6876', '0.6950', '0.7016', '0.7074', '0.7127', '0.7177', '0.7222', '0.7259', '0.7298', '0.7330', '0.7361', '0.7392', '0.7422', '0.7447']
  MRR@k      : ['0.5949', '0.6108', '0.6169', '0.6202', '0.6224', '0.6241', '0.6253', '0.6262', '0.6269', '0.6275', '0.6280', '0.6284', '0.6288', '0.6290', '0.6293', '0.6295', '0.6297', '0.6298', '0.6300', '0.6301']

===== epoch: 5
4284it [01:40, 42.46it/s]
        Total loss: 11126.384

  Cache evaluation (epoch 5):
  cache=    5  HR=0.0236  EPR=0.5704
  cache=   10  HR=0.0349  EPR=0.7154
  cache=   20  HR=0.0613  EPR=0.7990
  cache=   30  HR=0.0956  EPR=0.8087
  cache=   40  HR=0.1316  EPR=0.8027
  cache=   50  HR=0.1636  EPR=0.8017
  cache=   60  HR=0.1947  EPR=0.7869
  cache=   70  HR=0.2284  EPR=0.7731
  cache=   80  HR=0.2617  EPR=0.7473
  cache=   90  HR=0.2905  EPR=0.7295
  cache=  100  HR=0.3142  EPR=0.7156
  cache=  200  HR=0.4939  EPR=0.5159
  cache=  300  HR=0.6262  EPR=0.3808
  cache=  400  HR=0.9112  EPR=0.2333
  cache=  500  HR=0.9518  EPR=0.2094
  cache=  600  HR=0.9575  EPR=0.2043
  cache=  700  HR=0.9624  EPR=0.1944
  cache=  800  HR=0.9634  EPR=0.2118
  cache=  900  HR=0.9638  EPR=0.2156
  cache= 1000  HR=0.9647  EPR=0.2156
  Precision@k: ['0.5949', '0.6270', '0.6456', '0.6588', '0.6698', '0.6796', '0.6882', '0.6957', '0.7020', '0.7080', '0.7132', '0.7182', '0.7230', '0.7268', '0.7303', '0.7333', '0.7365', '0.7397', '0.7424', '0.7452']
  MRR@k      : ['0.5949', '0.6109', '0.6171', '0.6204', '0.6226', '0.6243', '0.6255', '0.6264', '0.6271', '0.6277', '0.6282', '0.6286', '0.6290', '0.6293', '0.6295', '0.6297', '0.6299', '0.6300', '0.6302', '0.6303']

===== epoch: 6
4284it [01:40, 42.52it/s]
        Total loss: 11069.275

  Cache evaluation (epoch 6):
  cache=    5  HR=0.0237  EPR=0.5956
  cache=   10  HR=0.0348  EPR=0.7367
  cache=   20  HR=0.0612  EPR=0.8170
  cache=   30  HR=0.0955  EPR=0.8252
  cache=   40  HR=0.1315  EPR=0.8195
  cache=   50  HR=0.1637  EPR=0.8205
  cache=   60  HR=0.1951  EPR=0.8069
  cache=   70  HR=0.2287  EPR=0.7954
  cache=   80  HR=0.2618  EPR=0.7716
  cache=   90  HR=0.2909  EPR=0.7546
  cache=  100  HR=0.3143  EPR=0.7404
  cache=  200  HR=0.4944  EPR=0.5521
  cache=  300  HR=0.6262  EPR=0.4269
  cache=  400  HR=0.9113  EPR=0.2698
  cache=  500  HR=0.9519  EPR=0.2456
  cache=  600  HR=0.9576  EPR=0.2381
  cache=  700  HR=0.9625  EPR=0.2256
  cache=  800  HR=0.9634  EPR=0.2420
  cache=  900  HR=0.9638  EPR=0.2436
  cache= 1000  HR=0.9647  EPR=0.2436
  Precision@k: ['0.5953', '0.6276', '0.6458', '0.6591', '0.6702', '0.6803', '0.6886', '0.6959', '0.7024', '0.7082', '0.7138', '0.7189', '0.7234', '0.7273', '0.7305', '0.7336', '0.7367', '0.7398', '0.7429', '0.7456']
  MRR@k      : ['0.5953', '0.6114', '0.6175', '0.6208', '0.6230', '0.6247', '0.6259', '0.6268', '0.6275', '0.6281', '0.6286', '0.6291', '0.6294', '0.6297', '0.6299', '0.6301', '0.6303', '0.6304', '0.6306', '0.6307']

===== epoch: 7
4284it [01:41, 42.14it/s]
        Total loss: 11062.514

  Cache evaluation (epoch 7):
  cache=    5  HR=0.0238  EPR=0.5920
  cache=   10  HR=0.0349  EPR=0.7360
  cache=   20  HR=0.0613  EPR=0.8173
  cache=   30  HR=0.0957  EPR=0.8256
  cache=   40  HR=0.1316  EPR=0.8197
  cache=   50  HR=0.1639  EPR=0.8208
  cache=   60  HR=0.1952  EPR=0.8073
  cache=   70  HR=0.2288  EPR=0.7964
  cache=   80  HR=0.2620  EPR=0.7728
  cache=   90  HR=0.2910  EPR=0.7560
  cache=  100  HR=0.3145  EPR=0.7418
  cache=  200  HR=0.4945  EPR=0.5556
  cache=  300  HR=0.6262  EPR=0.4318
  cache=  400  HR=0.9114  EPR=0.2723
  cache=  500  HR=0.9519  EPR=0.2428
  cache=  600  HR=0.9576  EPR=0.2353
  cache=  700  HR=0.9625  EPR=0.2229
  cache=  800  HR=0.9634  EPR=0.2420
  cache=  900  HR=0.9638  EPR=0.2436
  cache= 1000  HR=0.9647  EPR=0.2436
  Precision@k: ['0.5954', '0.6275', '0.6462', '0.6591', '0.6706', '0.6802', '0.6887', '0.6961', '0.7025', '0.7084', '0.7138', '0.7192', '0.7234', '0.7273', '0.7303', '0.7335', '0.7367', '0.7400', '0.7429', '0.7456']
  MRR@k      : ['0.5954', '0.6114', '0.6177', '0.6209', '0.6232', '0.6248', '0.6260', '0.6269', '0.6277', '0.6283', '0.6287', '0.6292', '0.6295', '0.6298', '0.6300', '0.6302', '0.6304', '0.6306', '0.6307', '0.6309']

===== epoch: 8
4284it [01:41, 42.34it/s]
        Total loss: 11057.986

  Cache evaluation (epoch 8):
  cache=    5  HR=0.0238  EPR=0.5914
  cache=   10  HR=0.0350  EPR=0.7364
  cache=   20  HR=0.0613  EPR=0.8160
  cache=   30  HR=0.0956  EPR=0.8228
  cache=   40  HR=0.1315  EPR=0.8163
  cache=   50  HR=0.1637  EPR=0.8180
  cache=   60  HR=0.1952  EPR=0.8046
  cache=   70  HR=0.2287  EPR=0.7932
  cache=   80  HR=0.2619  EPR=0.7717
  cache=   90  HR=0.2908  EPR=0.7545
  cache=  100  HR=0.3144  EPR=0.7402
  cache=  200  HR=0.4946  EPR=0.5547
  cache=  300  HR=0.6261  EPR=0.4297
  cache=  400  HR=0.9113  EPR=0.2684
  cache=  500  HR=0.9519  EPR=0.2428
  cache=  600  HR=0.9576  EPR=0.2353
  cache=  700  HR=0.9625  EPR=0.2229
  cache=  800  HR=0.9634  EPR=0.2420
  cache=  900  HR=0.9638  EPR=0.2436
  cache= 1000  HR=0.9647  EPR=0.2436
  Precision@k: ['0.5954', '0.6276', '0.6464', '0.6593', '0.6706', '0.6800', '0.6885', '0.6960', '0.7027', '0.7085', '0.7138', '0.7192', '0.7235', '0.7273', '0.7304', '0.7337', '0.7368', '0.7401', '0.7429', '0.7457']
  MRR@k      : ['0.5954', '0.6115', '0.6178', '0.6210', '0.6233', '0.6248', '0.6260', '0.6270', '0.6277', '0.6283', '0.6288', '0.6292', '0.6296', '0.6298', '0.6300', '0.6303', '0.6304', '0.6306', '0.6308', '0.6309']

===== epoch: 9
4284it [01:40, 42.49it/s]
        Total loss: 11051.120

  Cache evaluation (epoch 9):
  cache=    5  HR=0.0236  EPR=0.5946
  cache=   10  HR=0.0347  EPR=0.7383
  cache=   20  HR=0.0611  EPR=0.8175
  cache=   30  HR=0.0955  EPR=0.8255
  cache=   40  HR=0.1314  EPR=0.8206
  cache=   50  HR=0.1637  EPR=0.8216
  cache=   60  HR=0.1950  EPR=0.8080
  cache=   70  HR=0.2286  EPR=0.7977
  cache=   80  HR=0.2619  EPR=0.7770
  cache=   90  HR=0.2909  EPR=0.7613
  cache=  100  HR=0.3145  EPR=0.7480
  cache=  200  HR=0.4946  EPR=0.5617
  cache=  300  HR=0.6262  EPR=0.4402
  cache=  400  HR=0.9114  EPR=0.2742
  cache=  500  HR=0.9519  EPR=0.2440
  cache=  600  HR=0.9575  EPR=0.2364
  cache=  700  HR=0.9625  EPR=0.2236
  cache=  800  HR=0.9634  EPR=0.2418
  cache=  900  HR=0.9638  EPR=0.2434
  cache= 1000  HR=0.9647  EPR=0.2434
  Precision@k: ['0.5954', '0.6277', '0.6465', '0.6595', '0.6707', '0.6801', '0.6886', '0.6959', '0.7025', '0.7083', '0.7137', '0.7193', '0.7236', '0.7272', '0.7305', '0.7337', '0.7371', '0.7402', '0.7430', '0.7459']
  MRR@k      : ['0.5954', '0.6116', '0.6178', '0.6211', '0.6233', '0.6249', '0.6261', '0.6270', '0.6277', '0.6283', '0.6288', '0.6293', '0.6296', '0.6299', '0.6301', '0.6303', '0.6305', '0.6307', '0.6308', '0.6309']

Do per inference testing:
     len test_windows 60897  /  full test_trace 60930
/Users/brianjiang/Desktop/gnn_prefetcher/model_collection/sota_graph_sgdp/model.py:154: UserWarning: Creating a tensor from a list of numpy.ndarrays is extremely slow. Please consider converting the list to a single numpy.ndarray with numpy.array() before converting to a tensor. (Triggered internally at /Users/runner/work/pytorch/pytorch/pytorch/torch/csrc/utils/tensor_new.cpp:256.)
  A             = trans_to_cuda(torch.Tensor(A).float())
Total IO in the test set  60930

============================================================
  SGDP — EVALUATION RESULTS
  Dataset : dataset/MSR-Cambridge/hm_1.csv.gz
============================================================
  Hit Rate                  : 96.51%
  Prefetch Hit Rate         : 28.80%
  Access Speed (SSD)        : 286864.4 req/s
  Access Speed (HDD)        : 1434.3 req/s
  Prefetch Effectiveness    : 28.80%
  Prefetch Overhead         : 0.36%
  Total IOs                 : 60,930
  Total Prefetches Issued   : 309
  Cache Hits                : 58,806
  Prefetch Hits             : 89
  Cache Misses              : 2,124
  Inference Time (total)    : 53.64s
  Inference Latency (per)   : 0.881ms
  Throughput                : 1135.3 inf/s
  Model Size (# params)     : 261,800
============================================================

Number of test IOs: 60930
Sample prefetch addresses: [2652, 1452, 1740, 1468, 4788, 384624, 2332, 369900, 384656, 4468]
```