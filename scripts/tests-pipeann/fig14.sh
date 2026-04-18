echo "This script generates the data in Figure 14: throughput on 1B vectors."

source $(dirname $0)/eval_f.sh

echo "Run on DiskANN..."
run_10_1B 10 8 0 0 /mnt/nvme2/indices/SIFT1B/1B /mnt/nvme2/indices/SPACEV1B/1B 56  | tee ./data/fig14_diskann.txt

echo "Run on PipeANN..."
run_10_1B 10 32 2 10 /mnt/nvme2/indices/SIFT1B/1B /mnt/nvme2/indices/SPACEV1B/1B 56 | tee ./data/fig14_pipeann.txt
