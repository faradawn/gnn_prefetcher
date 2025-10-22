# SOTA: How to run

## Basic 
cd /home/cc/flashnet/model_collection/5_block_prefetching/

./base_prefetcher.linux_kernel.py -chunk 256kb -file ../dataset/iotrace/seagate.16k.all_read.fio_90seq_10rand_256k_8q_reads_8_lun_10min_container192_filtered/read_io.trace

## ML Based 
cd /home/cc/flashnet/model_collection/5_block_prefetching/simulate_sota_prefetcher

# 1
./sota.graph.py -file ../dataset/iotrace/seagate.16k.all_read.fio_90seq_10rand_256k_8q_reads_8_lun_10min_container192_filtered/read_io.trace

Hit rate = 0 %

# 2
./sota.graph.py -file ../dataset/iotrace/alibaba.cut.per_10k.rw_80_20.723/read_io.trace

Hit rate = 17.98 %

# 3
./sota.graph.py -file ../dataset/iotrace/alibaba.cut.per_10k.most_size_thpt_iops_rand.719/read_io.trace

Hit rate = 12.54 %

# For LSTM install gpu
https://www.tensorflow.org/install/pip
pip install --upgrade pip

# Export tensorflow model and import it 