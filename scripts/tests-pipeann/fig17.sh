echo "Only static policy needs to be re-run in Figure 17. We could use the data in Figures 11 and 12 for dynamic policy."
echo "If you are generating Figure 17 independently, please run fig11.sh and fig12.sh."

source $(dirname $0)/eval_f.sh

echo "Building +Static version..."
bash ./build_flags.sh "-DDYN_PIPE_WIDTH -DSTATIC_POLICY"

echo "Running static policy..."
run_10_100M 10 32 2 10 /mnt/nvme2/indices/bigann/100m /mnt/nvme2/indices/spacev/100M /mnt/nvme2/indices/deep/100M 1 | tee ./data/fig17_static_lat.txt

echo "Running dynamic policy..."
run_10_100M 10 32 2 10 /mnt/nvme2/indices/bigann/100m /mnt/nvme2/indices/spacev/100M /mnt/nvme2/indices/deep/100M 56 | tee ./data/fig17_static_tput.txt

bash ./build.sh
