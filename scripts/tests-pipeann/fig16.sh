echo "This script generates breakdown analysis."

function run_10_lat_tput_beam() {
    for i in 1 2 4 6 8 12 16 24 32 40 48 56
    do
        echo "[REPORT] T $i MODE $3 MEM_L $4 SIFT" 
        build/tests/search_disk_index uint8 $5 $i $2 /mnt/nvme/data/bigann/bigann_query.bbin /mnt/nvme/data/bigann/100M_gt.bin 10 l2 $3 $4 29
        echo "[REPORT] T $i MODE $3 MEM_L $4 SIFT" 
        build/tests/search_disk_index uint8 $5 $i $2 /mnt/nvme/data/bigann/bigann_query.bbin /mnt/nvme/data/bigann/100M_gt.bin 10 l2 $3 $4 115
    done
}

function run_10_lat_tput_pipe() {
    for i in 1 2 4 6 8 12 16 24 32 40 48 56
    do
        echo "[REPORT] T $i MODE $3 MEM_L $4 SIFT" 
        build/tests/search_disk_index uint8 $5 $i $2 /mnt/nvme/data/bigann/bigann_query.bbin /mnt/nvme/data/bigann/100M_gt.bin 10 l2 $3 $4 29
        echo "[REPORT] T $i MODE $3 MEM_L $4 SIFT" 
        build/tests/search_disk_index uint8 $5 $i $2 /mnt/nvme/data/bigann/bigann_query.bbin /mnt/nvme/data/bigann/100M_gt.bin 10 l2 $3 $4 118
    done
}

bash ./build.sh
run_10_lat_tput_beam 0 8 0 10 /mnt/nvme2/indices/bigann/100m | tee ./data/fig16_beam.txt

# re-compile to +Pipe.
if [ ! -f build_flags.sh ]; then
    echo "Please run the script in the root directory of PipeANN."
fi

echo "Building +Pipe version..."
bash ./build_flags.sh -DNAIVE_PIPE

run_10_lat_tput_pipe 0 8 2 10 /mnt/nvme2/indices/bigann/100m | tee ./data/fig16_pipe1.txt

echo "Building +AlgOpt version..."
bash ./build_flags.sh
run_10_lat_tput_pipe 0 8 2 10 /mnt/nvme2/indices/bigann/100m | tee ./data/fig16_pipe2.txt

echo "Building PipeANN..."
bash ./build.sh
run_10_lat_tput_pipe 0 32 2 10 /mnt/nvme2/indices/bigann/100m | tee ./data/fig16_pipe3.txt
