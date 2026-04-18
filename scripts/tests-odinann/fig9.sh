# breakdown analysis.

mkdir data
CWD=$(pwd)

rm /mnt/nvme/indices_upd/bigann/100M_shadow*
rm /mnt/nvme/indices_upd/bigann/100M_shadow1*
rm /mnt/nvme/indices_upd/bigann/100M_merge*

# build -DIN_PLACE_RECORD_UPDATE -DDIRECT_READ_CC

./build_flags.sh "-DIN_PLACE_RECORD_UPDATE -DDIRECT_READ_CC"
build/tests/test_insert_search uint8 /mnt/nvme/data/bigann/bigann_200M.bbin 128 100000 1 27 0 0 /mnt/nvme/indices_upd/bigann/100M /mnt/nvme/data/bigann/bigann_query.bbin /mnt/nvme/indices_upd/bigann_gnd_insert/500M_topk 0 10 4 4 0 10 |& tee $CWD/data/Baseline-insertonly-sift.txt

rm /mnt/nvme/indices_upd/bigann/100M_shadow*
rm /mnt/nvme/indices_upd/bigann/100M_shadow1*
rm /mnt/nvme/indices_upd/bigann/100M_merge*

./build_flags.sh "-DIN_PLACE_RECORD_UPDATE -DBG_IO_THREAD"
build/tests/test_insert_search uint8 /mnt/nvme/data/bigann/bigann_200M.bbin 128 100000 1 27 0 0 /mnt/nvme/indices_upd/bigann/100M /mnt/nvme/data/bigann/bigann_query.bbin /mnt/nvme/indices_upd/bigann_gnd_insert/500M_topk 0 10 4 4 0 10 |& tee $CWD/data/Async-insertonly-sift.txt

rm /mnt/nvme/indices_upd/bigann/100M_shadow*
rm /mnt/nvme/indices_upd/bigann/100M_shadow1*
rm /mnt/nvme/indices_upd/bigann/100M_merge*

./build_flags.sh "-DBG_IO_THREAD"
build/tests/test_insert_search uint8 /mnt/nvme/data/bigann/bigann_200M.bbin 128 100000 1 27 0 0 /mnt/nvme/indices_upd/bigann/100M /mnt/nvme/data/bigann/bigann_query.bbin /mnt/nvme/indices_upd/bigann_gnd_insert/500M_topk 0 10 4 4 0 10 |& tee $CWD/data/Op-insertonly-sift.txt

rm /mnt/nvme/indices_upd/bigann/100M_shadow*
rm /mnt/nvme/indices_upd/bigann/100M_shadow1*
rm /mnt/nvme/indices_upd/bigann/100M_merge*

./build.sh
build/tests/test_insert_search uint8 /mnt/nvme/data/bigann/bigann_200M.bbin 128 100000 1 27 0 0 /mnt/nvme/indices_upd/bigann/100M /mnt/nvme/data/bigann/bigann_query.bbin /mnt/nvme/indices_upd/bigann_gnd_insert/500M_topk 0 10 4 4 0 10 |& tee $CWD/data/Prune-insertonly-sift.txt
