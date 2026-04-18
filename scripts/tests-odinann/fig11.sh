# use different R.
# As we use a fixed location-to-ID table array size of 16 (see kMaxElemInAPage), 
# the memory footprint will be larger than reported in our paper.
# insert 5M vectors.

# R = 32, L = 64.
CWD=$(pwd)

for R in 32 64 96 128
do
    L=$[R+32]
    echo "REPORT R=$R, L=$L"
    build/tests/test_insert_search uint8 /mnt/nvme/data/bigann/bigann_200M.bbin $L 1000000 5 10 32 0 /mnt/nvme/indices_upd/bigann_varl/100m_l$R /mnt/nvme/data/bigann/bigann_query.bbin /mnt/nvme/indices_upd/bigann_gnd_insert/500M_topk 0 10 4 4 0 20 30 40 50 60 80 100 |& tee $CWD/data/varl_L$R.txt
    rm /mnt/nvme/indices_upd/bigann_varl/100m_l${R}_shadow*
    rm /mnt/nvme/indices_upd/bigann_varl/100m_l${R}_merge*
done

