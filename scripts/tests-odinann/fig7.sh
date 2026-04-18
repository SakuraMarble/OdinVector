# run OdinANN first.
rm /mnt/nvme/indices_upd/deep/100M_shadow*
rm /mnt/nvme/indices_upd/deep/100M_shadow1*
rm /mnt/nvme/indices_upd/deep/100M_merge*

mkdir data

CWD=$(pwd)

build/tests/test_insert_search float /mnt/nvme/data/deep/200M.fbin 128 1000000 100 10 32 0 /mnt/nvme/indices_upd/deep/100M /mnt/nvme/data/deep/queries.fbin /mnt/nvme/indices_upd/deep_gnd_insert/200M_topk 0 10 4 4 0 25 30 33 37 40 45 60 80 |& tee $CWD/data/OdinANN-insertonly-deep.txt

rm /mnt/nvme/indices_upd/deep/100M_mem*
rm /mnt/nvme/indices_upd/deep/100Mtemp0*
rm /mnt/nvme/indices_upd/deep/100M_shadow*
rm /mnt/nvme/indices_upd/deep/100M_shadow1*
rm /mnt/nvme/indices_upd/deep/100M_merge*

# run DiskANN.
cd /mnt/nvme2/DiskANN/
scripts/moti_long_deep.sh |& tee $CWD/data/DiskANN-insertonly-deep.txt

rm /mnt/nvme/indices_upd/deep/100M_mem*
rm /mnt/nvme/indices_upd/deep/100Mtemp0*
rm /mnt/nvme/indices_upd/deep/100M_shadow*
rm /mnt/nvme/indices_upd/deep/100M_shadow1*
rm /mnt/nvme/indices_upd/deep/100M_merge*

# run SPFresh.
rm /mnt/nvme2/PipeANN/data/result-deep100m/*
cd /mnt/nvme2/SPFresh
bash Script/overall_deep_spfresh.sh |& tee $CWD/data/SPFresh-insertonly-deep.txt
