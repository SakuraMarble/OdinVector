# run top-10 evaluation.

function run_10_100M() {
    NTHREADS=$8
    # $1: top-K (10 in typical), $2: max I/O pipeline width, $3: search mode, $4: mem_L (0 for DiskANN, 10 for Starling and PipeANN)
    # $5 ~ $7: SIFT/SPACEV/DEEP index file prefix, $8: num threads
    echo "[REPORT] K $1 BW $2 MODE $3 MEM_L $4 T $8 SIFT" 
    build/tests/search_disk_index uint8 $5 $NTHREADS $2 /mnt/nvme/data/bigann/bigann_query.bbin /mnt/nvme/data/bigann/100M_gt.bin $1 l2 $3 $4 10 15 20 25 30 35 40 50 60 80 120 200 400
    echo "[REPORT] K $1 BW $2 MODE $3 MEM_L $4 T $8 SPACEV" 
    build/tests/search_disk_index int8 $6 $NTHREADS $2 /mnt/nvme/data/SPACEV1B/query.bin /mnt/nvme/data/SPACEV1B/100M_gt.bin $1 l2 $3 $4 10 15 20 25 30 35 40 50 60 80 120 200 400
    echo "[REPORT] K $1 BW $2 MODE $3 MEM_L $4 T $8 DEEP" 
    build/tests/search_disk_index float $7 $NTHREADS $2 /mnt/nvme/data/deep/queries.fbin /mnt/nvme/data/deep/100M_gt.bin $1 l2 $3 $4 10 15 20 25 30 35 40 50 60 80 120 200 400
}

function run_10_1B() {
    NTHREADS=$7
    # $1: top-K (10 in typical), $2: max I/O pipeline width, $3: search mode, $4: mem_L (0 for DiskANN, 10 for Starling and PipeANN)
    # $5 ~ $6: SIFT/SPACEV index file prefix, $7: num threads
    echo "[REPORT] K $1 BW $2 MODE $3 MEM_L $4 T $7 SIFT" 
    build/tests/search_disk_index uint8 $5 $NTHREADS $2 /mnt/nvme/data/bigann/bigann_query.bbin /mnt/nvme/data/bigann/truth.bin $1 l2 $3 $4 10 15 20 25 30 35 40 50 60 80 120 200 400
    echo "[REPORT] K $1 BW $2 MODE $3 MEM_L $4 T $7 SPACEV" 
    build/tests/search_disk_index int8 $6 $NTHREADS $2 /mnt/nvme/data/SPACEV1B/query.bin /mnt/nvme/data/SPACEV1B/truth.bin $1 l2 $3 $4 10 15 20 25 30 35 40 50 60 80 120 200 400
}
