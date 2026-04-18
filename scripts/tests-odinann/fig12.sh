

# run OdinANN first.
rm /mnt/nvme/indices_upd/bigann/100M_shadow*
rm /mnt/nvme/indices_upd/bigann/100M_shadow1*
rm /mnt/nvme/indices_upd/bigann/100M_merge*

mkdir data

CWD=$(pwd)

build/tests/overall_performance uint8 /mnt/nvme/data/bigann/bigann_200M.bbin 128 /mnt/nvme/indices_upd/bigann/100M /mnt/nvme/data/bigann/bigann_query.bbin /mnt/nvme/indices_upd/bigann_gnd/500M_topk 10 4 100 20 30 |& tee $CWD/data/OdinANN-workload-change.txt

rm /mnt/nvme/indices_upd/bigann/100M_mem*
rm /mnt/nvme/indices_upd/bigann/100Mtemp0*
rm /mnt/nvme/indices_upd/bigann/100M_shadow*
rm /mnt/nvme/indices_upd/bigann/100M_shadow1*
rm /mnt/nvme/indices_upd/bigann/100M_merge*

# run DiskANN.
cd /mnt/nvme2/DiskANN/
scripts/overall_sift.sh |& tee $CWD/data/DiskANN-workload-change.txt

rm /mnt/nvme/indices_upd/bigann/100M_mem*
rm /mnt/nvme/indices_upd/bigann/100Mtemp0*
rm /mnt/nvme/indices_upd/bigann/100M_shadow*
rm /mnt/nvme/indices_upd/bigann/100M_shadow1*
rm /mnt/nvme/indices_upd/bigann/100M_merge*

# run SPFresh.
rm /mnt/nvme2/PipeANN/data/result-change/*
cd /mnt/nvme2/SPFresh
bash Script/overall_change_spfresh.sh |& tee $CWD/data/SPFresh-workload-change.txt
