echo "This is a hello-world example for OdinANN, it inserts 10K vectors into a 1M index."
build/tests/test_insert_search uint8 /mnt/nvme/data/bigann/bigann_2M.bbin 64 10000 1 10 32 0 /mnt/nvme/indices_upd/bigann/1M /mnt/nvme/data/bigann/bigann_query.bbin /mnt/nvme/indices_upd/bigann_gnd_insert/2M_topk 0 10 4 4 0 20

rm /mnt/nvme/indices_upd/bigann/1M_shadow*
