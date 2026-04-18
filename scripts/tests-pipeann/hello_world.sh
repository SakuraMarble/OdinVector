echo NOTE: this is a hello-world example for PipeANN. It searches top-10 on SIFT100M dataset.
build/tests/search_disk_index uint8 /mnt/nvme2/indices/bigann/100m 1 32 /mnt/nvme/data/bigann/bigann_query.bbin /mnt/nvme/data/bigann/100M_gt.bin 10 l2 2 10 10 20 30 40
echo Finished, you can check the latency and recall above.