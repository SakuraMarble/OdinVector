echo "This script generates the data in Figure 12: throughput on 100M vectors."

source $(dirname $0)/eval_f.sh

echo "Run on DiskANN..."
run_10_100M 10 8 0 0 /mnt/nvme2/indices/bigann/100m /mnt/nvme2/indices/spacev/100M /mnt/nvme2/indices/deep/100M 56  | tee ./data/fig12_diskann.txt

echo "Run on Starling..."
run_10_100M 10 4 1 10 /mnt/nvme2/indices_starling/bigann_100m_M256_R96_L128_B3.3/ /mnt/nvme2/indices_starling/spacev_100m_M500_R96_L128_B3.3/ /mnt/nvme2/indices_starling/deep_100m_M500_R96_L128_B3.3/ 56  | tee ./data/fig12_starling.txt

echo "Run on PipeANN..."
run_10_100M 10 32 2 10 /mnt/nvme2/indices/bigann/100m /mnt/nvme2/indices/spacev/100M /mnt/nvme2/indices/deep/100M 56 | tee ./data/fig12_pipeann.txt

echo "Run on SPANN in /mnt/nvme2/SPTAG..."
cd /mnt/nvme2/SPTAG
bash ./search_tput.sh | tee /mnt/nvme2/PipeANN/data/fig12_spann.txt