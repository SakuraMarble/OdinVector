# run OdinANN first.
rm /mnt/nvme/indices_upd/sift1b/800M_shadow*
rm /mnt/nvme/indices_upd/sift1b/800M_shadow1*
rm /mnt/nvme/indices_upd/sift1b/800M_merge*

mkdir data

CWD=$(pwd)

build/tests/test_insert_search uint8 /mnt/nvme/data/bigann/bigann.bbin 160 1000000 200 12 32 0 /mnt/nvme/indices_upd/sift1b/800M /mnt/nvme/data/bigann/bigann_query.bbin /mnt/nvme/indices_upd/bigann_gnd_insert/1B_topk 700000000 10 4 4 0 25 |& tee $CWD/data/OdinANN-stress.txt

rm /mnt/nvme/indices_upd/sift1b/800M_mem*
rm /mnt/nvme/indices_upd/sift1b/800Mtemp0*
rm /mnt/nvme/indices_upd/sift1b/800M_shadow*
rm /mnt/nvme/indices_upd/sift1b/800M_shadow1*
rm /mnt/nvme/indices_upd/sift1b/800M_merge*

# run DiskANN.
cd /mnt/nvme2/DiskANN/
scripts/moti_stress.sh |& tee $CWD/data/DiskANN-stress.txt

rm /mnt/nvme/indices_upd/sift1b/800M_mem*
rm /mnt/nvme/indices_upd/sift1b/800Mtemp0*
rm /mnt/nvme/indices_upd/sift1b/800M_shadow*
rm /mnt/nvme/indices_upd/sift1b/800M_shadow1*
rm /mnt/nvme/indices_upd/sift1b/800M_merge*

rm /mnt/nvme2/PipeANN/data/result-1b/*
cd /mnt/nvme2/SPFresh
bash Script/overall_sift_spfresh_stress.sh |& tee $CWD/data/SPFresh-stress.txt
