echo "This script generates the data in Figure 15: compare in-memory index with on-disk index."

source $(dirname $0)/eval_f.sh

echo "PipeANN is already run in fig11, so we just need to run Vamana."
echo "If you are generating Figure 15 independently, please run fig11.sh."

function run_10_mem() {
    scripts/tests/drop_cache
    echo "[REPORT] K $1 BW $2 MODE $3 MEM_L $4 T 1 SIFT"
    build/tests/search_disk_index_mem uint8 $5 0 0 1 $2 /mnt/nvme/data/bigann/bigann_query.bbin /mnt/nvme/data/bigann/100M_gt.bin $1 /dev/shm/results l2 $3 $4 10 15 20 25 30 40 50 60 80 100 120 160 200
    scripts/tests/drop_cache
    echo "[REPORT] K $1 BW $2 MODE $3 MEM_L $4 T 1 DEEP" 
    build/tests/search_disk_index_mem float $7 0 0 1 $2 /mnt/nvme/data/deep/queries.fbin /mnt/nvme/data/deep/100M_gt.bin $1 /dev/shm/results l2 $3 $4 10 15 20 25 30 40 50 60 80 100 120 160 200
    scripts/tests/drop_cache
}

run_10_mem 10 8 5 0 /mnt/nvme2/indices/bigann/100m /mnt/nvme2/indices/spacev/100M /mnt/nvme2/indices/deep/100M | tee ./data/fig15_vamana.txt
