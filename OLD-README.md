# [FAST'26 Artifact] OdinANN: Direct Insert for Consistently Stable Performance in Billion-Scale Graph-Based Vector Search

Welcome to the artifact repository of FAST'26 accepted paper: *OdinANN: Direct Insert for Consistently Stable Performance in Billion-Scale Graph-Based Vector Search*!
This repository contains the implementation of OdinANN, and scripts to reproduce the experiment results in our paper.

Should there be any questions, please **contact the authors in HotCRP** (or GitHub issue). The authors will respond to each question as soon as possible.

For AE reviewers, please directly refer to [Evaluate the Artifact](#for-reviewers-evaluate-the-artifact).
We have already set up the environment for AE reviewers on the provided platform.
Please see HotCRP for how to connect to it.


## Overview

We propose PipeANN, a **low-latency**, **billion-scale**, and **updatable** graph-based vector store (vector database) on SSD. Features:

* **Extremely low search latency**: <1ms in billion-scale vectors (top-10, 90% recall), only 1.14x-2.02x of in-memory graph-based index but **>10x** less memory usage (e.g., **40GB** for billion-scale datasets).

* **High search throughput**: 20K QPS in billion-scale vectors (top-10, 90% recall), higher than [DiskANN](https://github.com/microsoft/DiskANN) with `beam_width = 8` (latency-optimal) and [SPANN](https://github.com/microsoft/SPTAG).

* **Efficient vector updates**: `insert` and `delete` are supported with minimal interference with concurrent search (fluctuates only **1.07X**) and reduces memory usage (only **<90GB** for billion-scale datasets).

## Requirements

### Configurations

* CPU: X86 and ARM CPUs are tested. SIMD (e.g., AVX2, AVX512) will boost performance.

* DRAM: ~40GB (search-only) or ~90GB (search-update) per billion vectors, which may increase for larger product quantization (PQ) table size (Here we assume 32B per vector).

* SSD: ~700GB for SIFT with 1B vectors, ~900GB for SPACEV with 1.4B vectors.

* OS: Linux kernel supporting `io_uring` (e.g., >= 5.15) delivers best performance. Otherwise, enable `-DUSE_AIO` to use `libaio` instead.

* Vector dataset: less than 2B vectors to avoid integer overflow, each record size (`vector_size + 4 + 4 * num_neighbors`) is less than 4KB (>= 4KB is supported by search-only workloads but not well-tested, and not supported by updates).

### Software Dependencies

For Ubuntu >= 22.04, the command to install them:

```bash
sudo apt install make cmake g++ libaio-dev libgoogle-perftools-dev clang-format libboost-all-dev libmkl-full-dev libjemalloc-dev
```

The `libmkl` could be replaced by other `BLAS` libraries (e.g., OpenBlas).


### Build the Repository

First, build `liburing`. The compiled `liburing` library is in its `src` folder.

```bash
cd third_party/liburing
./configure
make -j
```


Then, build PipeANN.

```bash
bash ./build.sh
```

## Quick Start (Search-Only)

This section introduces how to build disk index and search using PipeANN.
To maximize search performance, `-DREAD_ONLY_TESTS` and `-DNO_MAPPING` definitions should be enabled in `CMakeLists.txt`.

### For DiskANN Users

For DiskANN users with on-disk indexes, enabling PipeANN only requires building an in-memory index (<10min for billion-scale datasets).
An example for SIFT100M (the hard-coded paths should be modified):

```bash
# Build in-memory index. Modify the INDEX_PREFIX and DATA_PATH beforehand.
export INDEX_PREFIX=/mnt/nvme2/indices/bigann/100m # on-disk index file name prefix.
export DATA_PATH=/mnt/nvme/data/bigann/100M.bbin
build/tests/utils/gen_random_slice uint8 ${DATA_PATH} ${INDEX_PREFIX}_SAMPLE_RATE_0.01 0.01
build/tests/build_memory_index uint8 ${INDEX_PREFIX}_SAMPLE_RATE_0.01_data.bin ${INDEX_PREFIX}_SAMPLE_RATE_0.01_ids.bin ${INDEX_PREFIX}_mem.index 0 0 32 64 1.2 24 l2

# Search the on-disk index. Modify the query and ground_truth path beforehand.
# build/tests/search_disk_index <data_type> <index_prefix> <nthreads> <I/O pipeline width (max for PipeANN)> <query file> <truth file> <top-K> <similarity> <search_mode (2 for PipeANN)> <L of in-memory index> <Ls for on-disk index> 
build/tests/search_disk_index uint8 ${INDEX_PREFIX} 1 32 /mnt/nvme/data/bigann/bigann_query.bbin /mnt/nvme/data/bigann/100M_gt.bin 10 l2 2 10 10 20 30 40
```

Then, you should see results like this:

```
Search parameters: #threads: 1,  beamwidth: 32
... some outputs during index loading ...
[search_disk_index.cpp:216:INFO] Use two ANNS for warming up...
[search_disk_index.cpp:219:INFO] Warming up finished.
     L   I/O Width         QPS    Mean Lat     P99 Lat   Mean Hops    Mean IOs   Recall@10
=========================================================================================
    10          32     1952.03      490.99     3346.00        0.00       22.28       67.11
    20          32     1717.53      547.84     1093.00        0.00       31.11       84.53
    30          32     1538.67      608.31     1231.00        0.00       41.02       91.04
    40          32     1420.46      655.24     1270.00        0.00       52.50       94.23
```

### For Others Starting from Scratch

This part introduces how to download the datasets, build the on-disk index, and then search on it using PipeANN.

#### Download the Datasets

* SIFT100M and SIFT1B from [BIGANN](http://corpus-texmex.irisa.fr/);
* DEEP1B using [deep1b_gt repository](https://github.com/matsui528/deep1b_gt) (Thanks, matsui528!);
* SPACEV100M and SPACEV1B from [SPTAG](https://github.com/microsoft/SPTAG).

If the datasets follow `ivecs` or `fvecs` format, you could transfer them into `bin` format using:
```bash
build/tests/utils/bvecs_to_bin bigann_base.bvecs bigann.bin # for byte vecs (SIFT), bigann_base.bvecs -> bigann.bin
build/tests/utils/fvecs_to_bin base.fvecs deep.bin # for float vecs (DEEP) base.fvecs -> deep.bin
build/tests/utils/ivecs_to_bin idx_1000M.ivecs idx_1000M.ibin # for int vecs (SIFT groundtruth) idx_1000M.ivecs -> idx_1000M.ibin
```

We need the `bin` files for `base`, `query`, and `groundtruth`.

The SPACEV1B dataset is divided into several sub-files in SPTAG. You could follow [SPTAG SPACEV1B](https://github.com/microsoft/SPTAG/tree/main/datasets/SPACEV1B) for how to read the dataset.
To concatenate them, just save the dataset's numpy `array` to `bin` format (the following Python code might be used).

```py
# bin format:
# | 4 bytes for num_vecs | 4 bytes for vector dimension (e.g., 100 for SPACEV) | flattened vectors |
def bin_write(vectors, filename):
    with open(filename, 'wb') as f:
        num_vecs, vector_dim = vectors.shape
        f.write(struct.pack('<i', num_vecs))
        f.write(struct.pack('<i', vector_dim))
        f.write(vectors.tobytes())

def bin_read(filename):
    with open(filename, 'rb') as f:
        num_vecs = struct.unpack('<i', f.read(4))[0]
        vector_dim = struct.unpack('<i', f.read(4))[0]
        data = f.read(num_vecs * vector_dim * 4)  # 4 bytes per float
        vectors = np.frombuffer(data, dtype=np.float32).reshape((num_vecs, vector_dim))
    return vectors
```

The dataset should contain a ground truth file for its full set.
Some datasets also contain the ground truth of subsets (first $k$ vectors). For example, SIFT100M's (the first 100M vectors of SIFT1B) ground truth could be found in `idx_100M.ivecs` of SIFT1B dataset.

#### Prepare the 100M Subsets

To generate the 100M subsets (SIFT100M, SPACEV100M, and DEEP100M) using the 1B datasets, use `change_pts` (for `bin`) and `pickup_vecs.py` (in the `deep1b_gt` repository, for `fvecs`).
```bash
# for SIFT, assume that the dataset is converted into bigann.bin
build/tests/change_pts uint8 /mnt/nvme/data/bigann/bigann.bin 100000000
mv /mnt/nvme/data/bigann/bigann.bin100000000 /mnt/nvme/data/bigann/100M.bbin

# for DEEP, assume that the dataset is in fvecs format.
python pickup_vecs.py --src ../base.fvecs --dst ../100M.fvecs --topk 100000000
# If DEEP is already in fbin format, change_pts also works.

# for SPACEV, assume that the dataset is concatenated into a huge file vectors.bin
build/tests/change_pts int8 /mnt/nvme/data/SPACEV1B/vectors.bin 100000000
mv /mnt/nvme/data/SPACEV1B/vectors.bin100000000 /mnt/nvme/data/SPACEV1B/100M.bin
```

Then, use `compute_groundtruth` to calculate the ground truths of 100M subsets. (DEEP100M's ground truth could be calculated using deep1b_gt).
If you want to evaluate **search-update workloads**, please **calculate top-1000 vectors** instead of top-100, from which we will pickup top-k vectors for different intervals.

Take SIFT100M as an example (in fact, its ground truth could also be found in SIFT1B):
```bash
# for SIFT100M, assume that the dataset is at /mnt/nvme/data/bigann/100M.bbin, the query is at /mnt/nvme/data/bigann/bigann_query.bbin 
# output: /mnt/nvme/data/bigann/100M_gt.bin
build/tests/utils/compute_groundtruth uint8 /mnt/nvme/data/bigann/100M.bbin /mnt/nvme/data/bigann/bigann_query.bbin 1000 /mnt/nvme/data/bigann/100M_gt.bin
```

#### Build On-Disk Index

PipeANN uses the same on-disk index as DiskANN.

```bash
# Usage:
# build/tests/build_disk_index <data_type (float/int8/uint8)> <data_file.bin> <index_prefix_path> <R>  <L>  <B>  <M>  <T> <similarity metric (cosine/l2) case sensitive>. <single_file_index (0/1)>
build/tests/build_disk_index uint8 /mnt/nvme/data/bigann/100M.bbin /mnt/nvme2/indices/bigann/100m 96 128 3.3 256 112 l2 0
```

The final index files will share a prefix of `/mnt/nvme2/indices/bigann/100m`.

Parameter explanation:

* R: maximum out-neighbors
* L: candidate pool size during build (build in fact conducts vector searches to optimize graph edges)
* B: in-memory PQ-compressed vector size. Our goal is to use 32 bytes per vector; higher-dimensional vectors might require more bytes.
* M: maximum memory used during build, 256GB is sufficient for the 100M index to be built totally in memory.
* T: number of threads used during build. Our machine has 112 threads.

We use the following parameters when building indexes:

| Dataset       | type | R  | L | B | M | T | similarity
|---------------|---|---|---|---|---| --- | --- 
| SIFT/DEEP/SPACEV100M | uint8/float/int8 | 96 | 128 | 3.3 | 256 | 112 | L2
| SIFT1B   | uint8 | 128 |  200 | 33 | 500 | 112 | L2
| SPACEV1B | int8 | 128 | 200  | 43 | 500 | 112 | L2

#### Build In-Memory Entry-Point Index (Optional)

An in-memory index is optional but could significantly improve performance by optimizing the entry point. By selecting `mem_L` to 0 in `search_disk_index`, the in-memory index is automatically skipped.

You could build it in the following way (take SIFT100M as an example):

```bash
# dataset: SIFT100M, {output prefix}: {INDEX_PREFIX}_SAMPLE_RATE_0.01
export INDEX_PREFIX=/mnt/nvme2/indices/bigann/100m # on-disk index file name prefix.
# first, generate random slice, sample rate is 0.01.
build/tests/utils/gen_random_slice uint8 /mnt/nvme/data/bigann/100M.bbin ${INDEX_PREFIX}_SAMPLE_RATE_0.01 0.01
# output files: {output prefix}_data.bin and {outputprefix}_ids.bin
# mem index: {INDEX_PREFIX}_mem.index
# All the in-memory indexes are built using R = 32, L = 64.
build/tests/build_memory_index uint8 ${INDEX_PREFIX}_SAMPLE_RATE_0.01_data.bin ${INDEX_PREFIX}_SAMPLE_RATE_0.01_ids.bin ${INDEX_PREFIX}_mem.index 0 0 32 64 1.2 24 l2
```

The output in-memory index should reside in three files: `100m_mem.index`, `100m_mem.index.data`, and `100m_mem.index.tags`.

## Quick Start (Search-Update)

Please prepare datasets and run PipeANN first, by referring to [Quick Start (Search-Only)](#quick-start-search-only).
Differently, `-DREAD_ONLY_TESTS` and `-DNO_MAPPING` flags should be disabled.
The in-memory index is optional.

### Prepare Tags (Optional)

Each vector corresponds to one tag, which does not change during updates (but its ID and location on disk may change).
PipeANN by default uses identity mapping (vector ID -> tag) for the initial vectors to their tags.
Use `gen_tags` to generate this mapping explicitly (optional, but necessary for FreshDiskANN).

```bash
# build/tests/gen_tags <type[int8/uint8/float]> <base_data_file> <index_file_prefix> <false>
build/tests/gen_tags uint8 /mnt/nvme/data/bigann/100M.bbin /mnt/nvme/indices_upd/bigann/100M false
```

You could use other tags if other mapping methods are preferred.

### Generate Ground-Truths

In our evaluation, we insert the second 100M vectors in SIFT1B/DEEP1B into the initial index built using its first 100M vectors.
We require ground truth for every `[0, 100M+iM)` vectors, where `0 <= i <= 100`.
The trivial approach is to calculate all of them, but it is costly.

We take a tricky approach: **select top-10 vectors for each interval** from the **top-1000** in SIFT1B (or the first 200M vectors in SIFT). 
Thus, only one top-1000 of the whole dataset should be calculated.
Amazingly, this approach is very likely to succeed.

```bash
# build/tests/gt_update <file> <tot_npts> <batch_npts> <target_topk> <target_dir> <insert_only>
# /mnt/nvme/data/bigann/truth.bin is the top-1000 for SIFT.
# insert 100M points in total, each batch contains 1M vectors.
build/tests/gt_update /mnt/nvme/data/bigann/truth.bin 100000000 1000000 10 /mnt/nvme/indices_upd/bigann_gnd/1B_topk 1
```

The output files will be stored in `/mnt/nvme/indices_upd/bigann_gnd/1B_topk/gt_{i * batch_npts}.bin`, each `bin` contains the ground truth for `[0, 100M + i*batch_npts)`.

For workload change (insert the second 100M, delete the first 100M), set the last parameter `insert_only` to `false`.

### Run The Benchmark

**Search-Insert Workload.** Please run `test_insert_search`. 
This benchmark first copies the original index to `_shadow.index`, to avoid polluting it.
Then, this benchmark executes `num_step` steps, each step inserts `vecs_per_step` vector (or, `batch_npts` above) and calculate recall.
Concurrent search with the `Ls` are executed.


An example to reproduce the results in OdinANN using SIFT100M dataset:

```bash
# build/tests/test_insert_search <type[int8/uint8/float]> <data_bin> <L_disk> <vecs_per_step> <num_steps> <insert_threads> <search_threads> <search_mode> <index_prefix> <query_file> <truthset_prefix> <truthset_l_offset> <recall@> <#beam_width> <search_beam_width> <mem_L> <Lsearch> <L2>
# 500M_topk stores the above-mentioned ground truth.
build/tests/test_insert_search uint8 /mnt/nvme/data/bigann/bigann_200M.bbin 128 1000000 100 10 32 0 /mnt/nvme/indices_upd/bigann/100M /mnt/nvme/data/bigann/bigann_query.bbin /mnt/nvme/indices_upd/bigann_gnd_insert/500M_topk 0 10 4 4 0 20 30 40 50 60 80 100
```

The `data_bin` should contain all the data (200M vectors in this setup), SIFT1B `bin` could also be used.
The `search_mode` is set to `0` for best-first search.
We use `L_disk = 128` for 100M-scale datasets and `160` for billion-scale datasets.
`mem_L` is set to 0, in order to skip using the in-memory index.

**Search-insert-delete workload.** Please run `overall_performance`.
This benchmark first copies the original index to `_shadow.index`, to avoid polluting it.
Then, this benchmark executes `step` steps, each step inserts and deletes `index_points / step` vector and calculate recall.
Concurrent search with the `Ls` are executed.

An example to reproduce the results in OdinANN using SIFT100M dataset:

```bash
# tests/overall_performance <type[int8/uint8/float]> <data_bin> <L_disk> <indice_path> <query_file> <truthset_prefix> <recall@> <#beam_width> <step> <Lsearch> <L2>
build/tests/overall_performance uint8 /mnt/nvme/data/bigann/bigann_200M.bbin 128 /mnt/nvme/indices_upd/bigann/100M /mnt/nvme/data/bigann/bigann_query.bbin /mnt/nvme/indices_upd/bigann_gnd/500M_topk 10 4 100 20 30
```

### Notes

* The index is **not crash-consistent** after updates currently, journaling could be adopted for it.

* To save a **consistent index snapshot** after updates, use `final_merge` similar to `test_insert_search`.

* For better performance, please select the `search_mode` to `2` (PipeANN) in `test_insert_search`, and set the `search_beam_width` to 32.
The in-memory index could also be used (but it is immutable during updates).


### Other Baselines

* For FreshDiskANN, please refer to [this repo](https://github.com/g4197/FreshDiskANN-baseline).

* For SPFresh, we directly use its original repo. We use the parameters in [SPFresh ini file](https://github.com/SPFresh/SPFresh/blob/main/Script_AE/iniFile/build_SPANN_spacev100m.ini) to build SPANN SIFT100M index. DEEP100M uses `PostingPageLimit=12`, while other configurations remain the same.


## For Reviewers: Evaluate the Artifact

We have prepared corresponding scripts to evaluate this artifact. 
Although this repo is named PipeANN, it is actually an integration of PipeANN and OdinANN.

### Hello-World Example (Artifact Functional, eta: 1 minute)

First, compile OdinANN.

```bash
bash ./build.sh
```

To verify that everything is prepared, you can run a hello-world example that verifies OdinANN's functionality, please run the following command:

```bash
bash scripts/tests-odinann/hello_world.sh
```

This script runs `test_insert_search`, which inserts 10K vectors into a index built using 1M vectors.
You could get a similar output as below. Note that the QPS, latency, and Recall might differ slightly.
```
This is a hello-world example for OdinANN, it inserts 10K vectors into a 1M index.
[test_insert_search.cpp:369:INFO] num insert threads: 10
... some outputs during index loading ...
[utils.h:291:INFO] Reading truthset file /mnt/nvme/indices_upd/bigann_gnd_insert/2M_topk/gt_0.bin...
[utils.h:300:INFO] Metadata: #pts = 10000, #dims = 10...
  Ls        QPS           Mean Lat      50 Lat      90 Lat      95 Lat      99 Lat    99.9 Lat   Recall@10    Disk IOs
==============================================================================
  20     2902.91            8.8294       6.733      18.721       22.44      30.237       39.81      92.136      34.269
... some other outputs during concurrent insert-search ...
[direct_insert.cpp:285:INFO] Processed 9075 tasks, throughput: 1814.97 tasks/sec.
... some other outputs during concurrent insert-search ...
[utils.h:291:INFO] Reading truthset file /mnt/nvme/indices_upd/bigann_gnd_insert/2M_topk/gt_10000.bin...
[utils.h:300:INFO] Metadata: #pts = 10000, #dims = 10...
  Ls        QPS           Mean Lat      50 Lat      90 Lat      95 Lat      99 Lat    99.9 Lat   Recall@10    Disk IOs
==============================================================================
  20     9213.54           1.34341       1.139       2.199       2.617       3.181       3.708      92.155     34.2034
... some other outputs for storing the index ...
[test_insert_search.cpp:342:INFO] Store finished.
```

If you can see this output, then everything is OK, and you can start running the artifact.

### Run All Experiments (Results Reproduced, eta: 30 days)

We have pre-compiled other baselines (FreshDiskANN and SPFresh).

* FreshDiskANN could be recompiled by executing `cmake ..` and `make -j` in its `build` directory.

* SPFresh is not advised to be recompiled, as it uses SPDK. We allow it to be run without `root` using the `setuid` bit. If you **really** want to recompile it, contact the authors via HotCRP.

Due to the limited time of AE, we assume that the reviewers cannot reproduce all the experiments.
However, we prepare scripts for all the experiments.
Reviewers could select some of them to reproduce. In general:

```bash
bash scripts/tests-odinann/figX.sh # X = 6, ..., 12
```
This generates the data used by Figure X. Here, we briefly introduce them:

* `fig6.sh` runs OdinANN, DiskANN and SPFresh using an insert-search workload on SIFT100M dataset. (eta: 4d)
* `fig7.sh` is similar to `fig6.sh`, but runs on DEEP100M dataset. (eta: 4d)
* `fig8.sh` runs the same workload using SIFT1B dataset (insert 200M vectors into the index with 800M vectors). (eta: 8d)
* `fig9.sh` inserts vectors using more insert threads. (eta: 1h)
* `fig10.sh` depends on `fig6.sh` and `fig7.sh`,without running additional experiments.
* `fig11.sh` runs the same workload using indexes with different out-neighbors (R). (eta: 5h)
* `fig12.sh` runs the insert-delete-search workload. (eta: 6d)

We place our experimental results inside the `data-example` folder for reference.
As we changed the logging code during reconstruction, the output might be slightly different.

### Plot the Figures

#### (Recommended) For Visual Studio Code users

Please install the Jupyter extension in VSCode. Then, please open `scripts/plotting.ipynb`.

Please use the Python 3.10.12 (`/usr/bin/python`) kernel.

Then, you can run each cell from top to bottom. The first cell contains prelude functions and definitions, so please run it first. Each other cell plots a figure.

In each cell, set the variable `USE_EXAMPLE` to `True` to use our provided example data for plotting, `False` to use the data generated by the scripts above.

### Appendix

#### Code Structure

We mainly introduce the `src` folder, which contains the main code.

```bash
src/
├── CMakeLists.txt
├── index.cpp # in-memory Vamana index
├── ssd_index.cpp # on-disk index (search-only)
├── search # search algorithms, details in README-PipeANN.md
│   ├── beam_search.cpp # best-first search
│   ├── coro_search.cpp # best-first search with inter-request scheduling
│   ├── page_search.cpp # search algorithm in Starling (SIGMOD '24)
│   └── pipe_search.cpp # our PipeANN search algorithm
├── update
│   ├── delete_merge.cpp # delete and merge implementation
│   ├── direct_insert.cpp # insert implementation
│   └── dynamic_index.cpp # on-disk index wrapper (search-insert)
└── utils # some utils (mainly for index building)
    ├── aux_utils.cpp
    ├── distance.cpp
    ├── linux_aligned_file_reader.cpp # io_uring and AIO support
    ├── math_utils.cpp
    ├── partition_and_pq.cpp
    ├── prune_neighbors.cpp
    └── utils.cpp
```

#### Indexes Used

We pre-built the indexes in our evaluation.
Here are the related files:

```bash
/mnt/nvme/data/
├── bigann
│   ├── 100M.bbin # SIFT100M dataset
│   ├── 100M_gt.bin # SIFT100M ground truth
│   ├── truth.bin # SIFT1B ground truth
│   ├── bigann.bbin # SIFT1B dataset
│   └── bigann_query.bbin # SIFT query
└── deep
    ├── 100M.fbin # DEEP100M dataset
    ├── 100M_gt.bin # DEEP100M ground truth
    └── queries.fbin # DEEP query

/mnt/nvme/indices_upd
├── bigann # SIFT100M index
│   ├── 100M_disk.index
│   ├── 100M_disk.index.tags
│   ├── 100M_pq_compressed.bin
│   └── 100M_pq_pivots.bin
├── bigann_gnd
│   └── 500M_topk # generated ground truth for insert-delete-search
├── bigann_gnd_insert
│   ├── 1B_topk # generated ground truth for insert-search
│   └── 500M_topk # generated ground truth for insert-search
├── bigann_varl # parameter study for different R (Figure 11)
├── deep_gnd_insert
│   └── 200M_topk # generated ground truth for insert-search
└── sift1b # SIFT1B index (800M vectors, insert 200M vectors)
    ├── 800M_disk.index
    ├── 800M_disk.index.tags
    ├── 800M_pq_compressed.bin
    └── 800M_pq_pivots.bin

/mnt/nvme2
├── indices_spann # Indexes for SPFresh
├── SPFresh # SPFresh code
├── DiskANN # FreshDiskANN code
└── PipeANN # source code of PipeANN+OdinANN

# For each index of OdinANN, important files:
/mnt/nvme/indices_upd/SIFT1B/
├── 1B_disk.index # full graph index
├── 1B_mem.index # in-memory graph
├── 1B_mem.index.tags # in-memory vector ID -> on-disk vector ID
├── 1B_pq_compressed.bin # PQ-compressed vectors
└── 1B_pq_pivots.bin # PQ pivots
```

## Cite Our Paper

If you use this repository in your research, please cite our papers:

Hao Guo and Youyou Lu. OdinANN: Direct Insert for Consistently Stable Performance in Billion-Scale Graph-Based Vector Search. To appear in the 24th USENIX Conference on File and Storage Technologies (FAST '26), Santa Clara CA USA, February 2026.

Hao Guo and Youyou Lu. Achieving Low-Latency Graph-Based Vector Search via Aligning Best-First Search Algorithm with SSD. To appear in the 19th USENIX Symposium on Operating Systems Design and Implementation (OSDI '25), Boston MA USA, July 2025.

## Acknowledgments

PipeANN is based on [DiskANN and FreshDiskANN](https://github.com/microsoft/DiskANN/tree/diskv2), we really appreciate it.
