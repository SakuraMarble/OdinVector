# run OdinANN first.
rm /mnt/nvme/indices_upd/bigann/100M_shadow*
rm /mnt/nvme/indices_upd/bigann/100M_shadow1*
rm /mnt/nvme/indices_upd/bigann/100M_merge*

mkdir data

CWD=$(pwd)

build/tests/test_insert_search uint8 /mnt/nvme/data/bigann/bigann_200M.bbin 128 1000000 100 10 32 0 /mnt/nvme/indices_upd/bigann/100M /mnt/nvme/data/bigann/bigann_query.bbin /mnt/nvme/indices_upd/bigann_gnd_insert/500M_topk 0 10 4 4 0 20 30 40 50 60 80 100 |& tee $CWD/data/OdinANN-insertonly-sift.txt

rm /mnt/nvme/indices_upd/bigann/100M_mem*
rm /mnt/nvme/indices_upd/bigann/100Mtemp0*
rm /mnt/nvme/indices_upd/bigann/100M_shadow*
rm /mnt/nvme/indices_upd/bigann/100M_shadow1*
rm /mnt/nvme/indices_upd/bigann/100M_merge*

# run DiskANN.
cd /mnt/nvme2/DiskANN/
scripts/moti_long.sh |& tee $CWD/data/DiskANN-insertonly-sift.txt

rm /mnt/nvme/indices_upd/bigann/100M_mem*
rm /mnt/nvme/indices_upd/bigann/100Mtemp0*
rm /mnt/nvme/indices_upd/bigann/100M_shadow*
rm /mnt/nvme/indices_upd/bigann/100M_shadow1*
rm /mnt/nvme/indices_upd/bigann/100M_merge*

# run SPFresh.
rm /mnt/nvme2/PipeANN/data/result-sift100m/*
cd /mnt/nvme2/SPFresh
bash Script/overall_sift_spfresh.sh |& tee $CWD/data/SPFresh-insertonly-sift.txt
