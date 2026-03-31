## tests/insert_demo.cpp (compared to old test_insert_search.cpp)
```diff
--- C:\Users\dasua\Desktop\DiskANN\26FAST-PipeANN-main\tests\test_insert_search.cpp
+++ C:\Users\dasua\Desktop\DiskANN\ODinANN-Demo\tests\insert_demo.cpp
@@ -1,416 +1,836 @@
-#include "ssd_index.h"
-#include "v2/dynamic_index.h"
-
+﻿#include "v2/dynamic_index.h"
 #include <index.h>
-#include <cstddef>
-#include <future>
-#include <numeric>
 #include <omp.h>
-#include <string.h>
-#include <time.h>
-#include <timer.h>
-#include <cstring>
-#include <iomanip>
+#include <string>
+#include <vector>
 #include <iostream>
 #include <fstream>
-#include <dirent.h>
-#include <sys/stat.h>
-
-#include "aux_utils.h"
-#include "index.h"
-#include "math_utils.h"
-#include "partition_and_pq.h"
-#include "utils.h"
-
-#include <sys/mman.h>
+#include <atomic>
+#include <limits>
+#include <algorithm>
+#include <cctype>
+#include <sstream>
 #include <sys/stat.h>
 #include <unistd.h>
-
+#include <stdexcept>
+#include <iomanip>
+#include <cstdlib>
+#include "utils.h"         // Log macros
+#include "timer.h"         // pipeann::Timer
+
+
+
+template<typename T, typename TagT>
+static void verify_graph_links(pipeann::SSDIndex<T, TagT> *disk, const std::string &disk_index_path,
+                               uint64_t base_count, uint64_t total_after, uint64_t max_samples = 1000) {
+    if (disk == nullptr) {
+        LOG(WARNING) << "[Verify] Disk index is null, skip verification.";
+        return;
+    }
+    if (base_count == 0 || total_after <= base_count) {
+        LOG(INFO) << "[Verify] No new points to verify.";
+        return;
+    }
+    std::ifstream in(disk_index_path, std::ios::binary);
+    if (!in.is_open()) {
+        LOG(WARNING) << "[Verify] Failed to open disk index: " << disk_index_path;
+        return;
+    }
+
+    const uint64_t new_count = total_after - base_count;
+    const uint64_t sample_old = std::min<uint64_t>(max_samples, base_count);
+    const uint64_t sample_new = std::min<uint64_t>(max_samples, new_count);
+    const uint64_t stride_old = std::max<uint64_t>(1, base_count / sample_old);
+    const uint64_t stride_new = std::max<uint64_t>(1, new_count / sample_new);
+
+    std::vector<char> sector_buf(SECTOR_LEN);
+    uint64_t last_sector = std::numeric_limits<uint64_t>::max();
+
+    auto read_sector = [&](uint64_t sector_no) -> bool {
+        if (sector_no == last_sector) {
+            return true;
+        }
+        in.seekg((std::streamoff) sector_no * SECTOR_LEN, std::ios::beg);
+        if (!in.read(sector_buf.data(), SECTOR_LEN)) {
+            return false;
+        }
+        last_sector = sector_no;
+        return true;
+    };
+
+    uint64_t old_checked = 0, old_edges_to_new = 0, old_edges_total = 0;
+    for (uint64_t idx = 0; idx < base_count && old_checked < sample_old; idx += stride_old) {
+        uint64_t id = idx;
+        uint64_t sector = disk->node_sector_no((uint32_t) id);
+        if (!read_sector(sector)) {
+            LOG(WARNING) << "[Verify] Read failed at sector " << sector;
+            break;
+        }
+        char *node_buf = disk->offset_to_node(sector_buf.data(), (uint32_t) id);
+        unsigned *nbrs = disk->offset_to_node_nhood(node_buf);
+        unsigned nnbrs = *(nbrs - 1);
+        if (nnbrs > disk->max_degree) {
+            nnbrs = (unsigned) disk->max_degree;
+        }
+        for (unsigned i = 0; i < nnbrs; ++i) {
+            uint32_t nbr = nbrs[i];
+            if (nbr >= base_count && nbr < total_after) {
+                old_edges_to_new++;
+            }
+        }
+        old_edges_total += nnbrs;
+        old_checked++;
+    }
+
+    uint64_t new_checked = 0, new_edges_to_old = 0, new_edges_total = 0;
+    for (uint64_t idx = 0; idx < new_count && new_checked < sample_new; idx += stride_new) {
+        uint64_t id = base_count + idx;
+        uint64_t sector = disk->node_sector_no((uint32_t) id);
+        if (!read_sector(sector)) {
+            LOG(WARNING) << "[Verify] Read failed at sector " << sector;
+            break;
+        }
+        char *node_buf = disk->offset_to_node(sector_buf.data(), (uint32_t) id);
+        unsigned *nbrs = disk->offset_to_node_nhood(node_buf);
+        unsigned nnbrs = *(nbrs - 1);
+        if (nnbrs > disk->max_degree) {
+            nnbrs = (unsigned) disk->max_degree;
+        }
+        for (unsigned i = 0; i < nnbrs; ++i) {
+            uint32_t nbr = nbrs[i];
+            if (nbr < base_count) {
+                new_edges_to_old++;
+            }
+        }
+        new_edges_total += nnbrs;
+        new_checked++;
+    }
+
+    LOG(INFO) << "[Verify] Sample old nodes: " << old_checked
+              << ", edges->new: " << old_edges_to_new << "/" << old_edges_total;
+    LOG(INFO) << "[Verify] Sample new nodes: " << new_checked
+              << ", edges->old: " << new_edges_to_old << "/" << new_edges_total;
+
+    if (old_edges_to_new == 0) {
+        LOG(WARNING) << "[Verify] No old->new edges found in sample. Graph may be disconnected.";
+    }
+    if (new_edges_to_old == 0) {
+        LOG(WARNING) << "[Verify] No new->old edges found in sample. New nodes may be isolated.";
+    }
+}
+
+// ================= ???? =================
 int NUM_INSERT_THREADS = 10;
-int NUM_SEARCH_THREADS = 32;
-
-int search_mode = BEAM_SEARCH;
-
+pipeann::Timer globalTimer;
 int begin_time = 0;
-pipeann::Timer globalTimer;
-
-// acutually also shows disk size
-void ShowMemoryStatus(const std::string &filename) {
-  int current_time = globalTimer.elapsed() / 1.0e6f - begin_time;
-
... (diff truncated)
```
## src/ssd_index.cpp
```diff
--- C:\Users\dasua\Desktop\DiskANN\26FAST-PipeANN-main\src\ssd_index.cpp
+++ C:\Users\dasua\Desktop\DiskANN\ODinANN-Demo\src\ssd_index.cpp
@@ -3,6 +3,7 @@
 #include <malloc.h>
 
 #include <omp.h>
+#include <fstream>
 #include <cmath>
 #include "liburing/io_uring.h"
 #include "parameters.h"
@@ -11,10 +12,20 @@
 #include "utils.h"
 
 #include <unistd.h>
+#include <cstdlib>
 #include <sys/syscall.h>
 #include "tsl/robin_set.h"
 
 namespace pipeann {
+  static double get_rss_mb() {
+    std::ifstream f("/proc/self/statm");
+    long t = 0, r = 0;
+    if (f.is_open()) {
+      f >> t >> r;
+    }
+    long page_kb = sysconf(_SC_PAGE_SIZE) / 1024;
+    return (double) r * page_kb / 1024.0;
+  }
   template<typename T>
   DiskNode<T>::DiskNode(uint32_t id, T *coords, uint32_t *nhood) : id(id) {
     this->coords = coords;
@@ -222,8 +233,47 @@
     LOG(INFO) << "After single file index check, Tags offset: " << tags_offset
               << " PQ Pivots offset: " << pq_pivots_offset << " PQ Vectors offset: " << pq_vectors_offset;
 
-    size_t npts_u64, nchunks_u64;
-    pipeann::load_bin<_u8>(pq_compressed_vectors, data, npts_u64, nchunks_u64, pq_vectors_offset);
+    size_t npts_u64 = 0, nchunks_u64 = 0;
+    LOG(INFO) << "RSS before PQ load: " << get_rss_mb() << " MB";
+    {
+      std::ifstream pq_reader(pq_compressed_vectors, std::ios::binary);
+      if (!pq_reader.is_open()) {
+        LOG(ERROR) << "Failed to open PQ file: " << pq_compressed_vectors;
+        return -1;
+      }
+      pq_reader.seekg(pq_vectors_offset, pq_reader.beg);
+      int npts_i32 = 0, dim_i32 = 0;
+      pq_reader.read((char *) &npts_i32, sizeof(int));
+      pq_reader.read((char *) &dim_i32, sizeof(int));
+      if (!pq_reader.good()) {
+        LOG(ERROR) << "Failed to read PQ header from: " << pq_compressed_vectors;
+        return -1;
+      }
+      npts_u64 = (unsigned) npts_i32;
+      nchunks_u64 = (unsigned) dim_i32;
+      LOG(INFO) << "Metadata: #pts = " << npts_u64 << ", #dims = " << nchunks_u64 << "...";
+
+      uint64_t reserve_pts = npts_u64;
+      const char *expected_env = std::getenv("PIPEANN_PQ_EXPECTED_POINTS");
+      const char *extra_env = std::getenv("PIPEANN_PQ_EXTRA_POINTS");
+      if (expected_env && expected_env[0] != '\0') {
+        reserve_pts = std::max<uint64_t>(reserve_pts, std::strtoull(expected_env, nullptr, 10));
+      }
+      if (extra_env && extra_env[0] != '\0') {
+        uint64_t extra_pts = std::strtoull(extra_env, nullptr, 10);
+        reserve_pts = std::max<uint64_t>(reserve_pts, npts_u64 + extra_pts);
+      }
+      if (reserve_pts > npts_u64) {
+        LOG(INFO) << "PQ reserve points: " << reserve_pts << " (base " << npts_u64 << ")";
+      }
+      data.resize(reserve_pts * nchunks_u64);
+      pq_reader.read((char *) data.data(), (std::streamsize) (npts_u64 * nchunks_u64 * sizeof(_u8)));
+      if (!pq_reader.good()) {
+        LOG(ERROR) << "Failed to read PQ payload from: " << pq_compressed_vectors;
+        return -1;
+      }
+    }
+    LOG(INFO) << "RSS after PQ load: " << get_rss_mb() << " MB";
     this->num_points = this->init_num_pts = npts_u64;
     this->n_chunks = nchunks_u64;
 
@@ -256,13 +306,17 @@
 
     // load page layout and set cur_loc
     this->use_page_search_ = use_page_search;
+    LOG(INFO) << "RSS before page layout: " << get_rss_mb() << " MB";
     this->load_page_layout(index_prefix, nnodes_per_sector, num_points);
+    LOG(INFO) << "RSS after page layout: " << get_rss_mb() << " MB";
 
     // load tags
     if (this->enable_tags) {
       std::string tag_file = disk_index_file + ".tags";
+      LOG(INFO) << "RSS before tags load: " << get_rss_mb() << " MB";
       LOG(INFO) << "Loading tags from " << tag_file;
       this->load_tags(tag_file);
+      LOG(INFO) << "RSS after tags load: " << get_rss_mb() << " MB";
     }
 
     num_medoids = 1;
@@ -320,6 +374,20 @@
       LOG(INFO) << "Tags file not found. Using equal mapping";
       // Equal mapping are by default eliminated in tags map.
     } else {
+#ifdef TAGS_IDENTITY_ONLY
+      LOG(INFO) << "Tags identity mapping compiled-in. Skipping tag load.";
+      return;
+#endif
+      const char *skip_env = std::getenv("PIPEANN_SKIP_TAGS_LOAD");
+      if (skip_env && skip_env[0] != '\0' && skip_env[0] != '0') {
+        LOG(INFO) << "Skipping tags load due to PIPEANN_SKIP_TAGS_LOAD. Using equal mapping.";
+        return;
+      }
+      const char *identity_env = std::getenv("PIPEANN_TAGS_IDENTITY");
+      if (identity_env && identity_env[0] != '\0' && identity_env[0] != '0') {
+        LOG(INFO) << "Skipping tags load due to PIPEANN_TAGS_IDENTITY. Using equal mapping.";
+        return;
+      }
       LOG(INFO) << "Load tags from existing file: " << tag_file_name;
       pipeann::load_bin<TagT>(tag_file_name, tag_v, tag_num, tag_dim, offset);
       tags.reserve(tag_v.size());
... (diff truncated)
```

## src/update/delete_merge.cpp
```diff
--- C:\Users\dasua\Desktop\DiskANN\26FAST-PipeANN-main\src\update\delete_merge.cpp
+++ C:\Users\dasua\Desktop\DiskANN\ODinANN-Demo\src\update\delete_merge.cpp
@@ -11,6 +11,7 @@
 #include <cstdint>
 #include <limits>
 #include <tuple>
+#include <unordered_set>
 #include "timer.h"
 #include "tsl/robin_map.h"
 #include "utils.h"
@@ -31,10 +32,10 @@
       nthreads = this->max_nthreads;
     }
 
-    void *ctx = reader->get_ctx();
+    void *ctx = reader->get_ctx();// 从文件读取器获取上下文指针
 
     while (!bg_tasks.empty()) {
-      sleep(5);  // simple way to wait for background IO thread.
+      sleep(5);  // 循环检查后台任务队列是否为空 确保所有异步IO操作已完成 simple way to wait for background IO thread.
     }
     std::string disk_index_out = out_path_prefix + "_disk.index";
     // Note that the index is immutable currently.
@@ -45,15 +46,15 @@
     Timer delete_timer;
 
     char *rbuf = nullptr, *wbuf = nullptr;
-    alloc_aligned((void **) &rbuf, SECTORS_PER_MERGE * SECTOR_LEN, SECTOR_LEN);
-    alloc_aligned((void **) &wbuf, 2 * SECTORS_PER_MERGE * SECTOR_LEN, SECTOR_LEN);  // sliding window buffer.
-    uint64_t n_sectors = (cur_loc + nnodes_per_sector - 1) / nnodes_per_sector;
+    alloc_aligned((void **) &rbuf, SECTORS_PER_MERGE * SECTOR_LEN, SECTOR_LEN);// 读取缓冲区
+    alloc_aligned((void **) &wbuf, 2 * SECTORS_PER_MERGE * SECTOR_LEN, SECTOR_LEN);  // 写入缓冲区 sliding window buffer.
+    uint64_t n_sectors = (cur_loc + nnodes_per_sector - 1) / nnodes_per_sector;// 计算索引总扇区数
     LOG(INFO) << "Cur loc: " << cur_loc.load() << ", cur ID: " << cur_id << ", n_sectors: " << n_sectors
               << ", nnodes_per_sector: " << nnodes_per_sector;
 
-    constexpr int SECTORS_PER_POPULATE = 128;             // small to avoid blocking search threads.
-    uint32_t populate_nthreads = std::min(nthreads, 4u);  // restrict the flow.
-
+    constexpr int SECTORS_PER_POPULATE = 128;             // 每次处理的扇区数 small to avoid blocking search threads.
+    uint32_t populate_nthreads = std::min(nthreads, 4u);  // 实际用于填充操作的线程数 restrict the flow.
+    // 循环处理所有扇区
     for (uint64_t in_sector = 0; in_sector < n_sectors; in_sector += SECTORS_PER_POPULATE) {
       uint64_t st_sector = in_sector, ed_sector = std::min(in_sector + SECTORS_PER_POPULATE, n_sectors);
       uint64_t loc_st = st_sector * nnodes_per_sector, loc_ed = std::min(cur_loc.load(), ed_sector * nnodes_per_sector);
@@ -246,6 +247,445 @@
   }
 
   template<typename T, typename TagT>
+  void SSDIndex<T, TagT>::merge_deletes_stream_pq_tags(const std::string &in_path_prefix,
+                                                       const std::string &out_path_prefix,
+                                                       const std::vector<TagT> &deleted_nodes,
+                                                       const tsl::robin_set<TagT> &deleted_nodes_set,
+                                                       uint32_t nthreads, const uint32_t &n_sampled_nbrs) {
+    if (nthreads == 0) {
+      nthreads = this->max_nthreads;
+    }
+
+    void *ctx = reader->get_ctx();
+    while (!bg_tasks.empty()) {
+      sleep(5);
+    }
+    while (bg_tasks_inflight.load(std::memory_order_relaxed) != 0) {
+      sleep(1);
+    }
+    // ensure any cached updates are flushed before scan/write
+    this->flush_page_cache();
+    std::string disk_index_out = out_path_prefix + "_disk.index";
+
+    const bool no_deletes = deleted_nodes.empty() || deleted_nodes_set.empty();
+    if (no_deletes) {
+      if (thread_pq_bufs.empty()) {
+        LOG(ERROR) << "thread_pq_bufs is empty; cannot merge.";
+        return;
+      }
+
+      char *rbuf = nullptr, *wbuf = nullptr;
+      alloc_aligned((void **) &rbuf, SECTORS_PER_MERGE * SECTOR_LEN, SECTOR_LEN);
+      alloc_aligned((void **) &wbuf, 2 * SECTORS_PER_MERGE * SECTOR_LEN, SECTOR_LEN);
+
+      uint64_t n_sectors = (cur_loc + nnodes_per_sector - 1) / nnodes_per_sector;
+      const uint64_t new_npoints_val = cur_id;
+      LOG(INFO) << "Merge(no-deletes, stream PQ/tags): cur_loc=" << cur_loc.load()
+                << " cur_id=" << cur_id << " n_sectors=" << n_sectors
+                << " nnodes_per_sector=" << nnodes_per_sector;
+
+      int fd = open(disk_index_out.c_str(), O_DIRECT | O_LARGEFILE | O_RDWR | O_CREAT, 0755);
+      const uint64_t file_size =
+          SECTOR_LEN + ROUND_UP(new_npoints_val, nnodes_per_sector) / nnodes_per_sector * SECTOR_LEN;
+      ::ftruncate(fd, file_size);
+
+      std::fstream pq_io(out_path_prefix + "_pq_compressed.bin",
+                         std::ios::binary | std::ios::in | std::ios::out | std::ios::trunc);
+      std::fstream tags_io(out_path_prefix + "_disk.index.tags",
+                           std::ios::binary | std::ios::in | std::ios::out | std::ios::trunc);
+      if (!pq_io.is_open() || !tags_io.is_open()) {
+        LOG(ERROR) << "Failed to open PQ/tags output for streaming.";
+        close(fd);
+        aligned_free((void *) rbuf);
+        aligned_free((void *) wbuf);
+        return;
+      }
+      uint32_t npts_u32 = (uint32_t) new_npoints_val;
+      uint32_t ndims_u32 = (uint32_t) this->n_chunks;
+      pq_io.write((char *) &npts_u32, sizeof(uint32_t));
+      pq_io.write((char *) &ndims_u32, sizeof(uint32_t));
+      uint32_t tag_ndims_u32 = 1;
+      tags_io.write((char *) &npts_u32, sizeof(uint32_t));
+      tags_io.write((char *) &tag_ndims_u32, sizeof(uint32_t));
+
+      const uint64_t kVecInWBuf = 2 * SECTORS_PER_MERGE * nnodes_per_sector;
+      uint64_t wb_id = 0;
+      std::atomic<uint64_t> n_used_id = 0;
+      auto write_back = [&]() {
+        uint64_t buf_id = (wb_id % kVecInWBuf) / (nnodes_per_sector * SECTORS_PER_MERGE);
+        auto b = wbuf + buf_id * SECTORS_PER_MERGE * SECTOR_LEN;
+        std::vector<IORequest> write_reqs;
+        uint64_t id_delta = std::min((uint64_t) SECTORS_PER_MERGE * nnodes_per_sector, n_used_id - wb_id);
+        write_reqs.push_back(IORequest(loc_sector_no(wb_id) * SECTOR_LEN,
+                                       ROUND_UP(id_delta, nnodes_per_sector) / nnodes_per_sector * size_per_io, b, 0, 0));
... (diff truncated)
```

## src/update/direct_insert.cpp
```diff
--- C:\Users\dasua\Desktop\DiskANN\26FAST-PipeANN-main\src\update\direct_insert.cpp
+++ C:\Users\dasua\Desktop\DiskANN\ODinANN-Demo\src\update\direct_insert.cpp
@@ -4,6 +4,7 @@
 #include <malloc.h>
 #include <algorithm>
 #include <filesystem>
+#include <fstream>
 
 #include <omp.h>
 #include <chrono>
@@ -17,156 +18,247 @@
 #include "v2/page_cache.h"
 
 #include <unistd.h>
+#include <cstdlib>
 #include <sys/syscall.h>
 #include "linux_aligned_file_reader.h"
 
 namespace pipeann {
+  static double get_rss_mb() {
+    std::ifstream f("/proc/self/statm");
+    long t = 0, r = 0;
+    if (f.is_open()) {
+      f >> t >> r;
+    }
+    long page_kb = sysconf(_SC_PAGE_SIZE) / 1024;
+    return (double) r * page_kb / 1024.0;
+  }
+
+  static bool should_trace_insert(uint64_t trace_id) {
+    const char *env = std::getenv("PIPEANN_TRACE_INSERT");
+    if (!env) {
+      return false;
+    }
+    int limit = std::atoi(env);
+    if (limit <= 0) {
+      return false;
+    }
+    return trace_id < (uint64_t) limit;
+  }
+
+  static std::atomic<uint64_t> trace_insert_counter{0};
+  //point是待插入向量，tag是关联标签，deletion_set是删除标记集合（用于逻辑删除）
   template<typename T, typename TagT>
   int SSDIndex<T, TagT>::insert_in_place(const T *point, const TagT &tag, tsl::robin_set<uint32_t> *deletion_set) {
     QueryBuffer<T> *read_data = this->pop_query_buf(nullptr);
     void *ctx = reader->get_ctx();
 
-    uint32_t target_id = cur_id++;
+    uint32_t target_id = cur_id++;// 待插入向量的ID 全局唯一
+    const uint64_t trace_id = trace_insert_counter.fetch_add(1, std::memory_order_relaxed);
+    const bool trace = should_trace_insert(trace_id);
+    pipeann::Timer trace_timer;
+    if (trace) {
+      LOG(INFO) << "[TraceInsert] id=" << target_id << " start rss=" << get_rss_mb() << " MB";
+    }
     // write PQ.
 
-    std::vector<uint8_t> pq_coords = deflate_vector(point);
-    uint64_t pq_offset = target_id * n_chunks;
+    std::vector<uint8_t> pq_coords = deflate_vector(point);// 将原始向量压缩为PQ编码的uint8_t数组
+    uint64_t pq_offset = target_id * n_chunks;// 计算PQ坐标在 data 数组中的存储偏移量
     {
-      static std::mutex pq_mu;
+      static std::mutex pq_mu;// pq_mu 是静态互斥锁，保证多线程环境下的数据一致性
       std::lock_guard<std::mutex> lock(pq_mu);
-      if (this->data.size() < pq_offset + n_chunks) {
+      if (this->data.size() < pq_offset + n_chunks) { // 判断 data 容器是否足够存储新数据
         while (this->data.size() < pq_offset + n_chunks) {
-          this->data.resize(1.5 * this->data.size());
+          this->data.resize(1.5 * this->data.size());// 使用1.5倍扩容避免频繁重新分配内存
         }
       }
+      // 将压缩后的向量数据写入到data容器位置
+      /* 目的地址：this->data.data() + pq_offset 指向目标存储位置
+         源地址：pq_coords.data() 指向PQ压缩后的数据
+         复制字节数：n_chunks 个字节（即一个向量的PQ编码长度）*/
       memcpy(this->data.data() + pq_offset, pq_coords.data(), n_chunks);
     }
-
-    std::vector<Neighbor> exp_node_info;
-    tsl::robin_map<uint32_t, T *> coord_map;
-    coord_map.reserve(2 * this->l_index);
+    if (trace) {
+      LOG(INFO) << "[TraceInsert] id=" << target_id << " step=pq_write"
+                << " ms=" << (trace_timer.elapsed() / 1.0e3f)
+                << " rss=" << get_rss_mb() << " MB"
+                << " data_size=" << this->data.size();
+    }
+
+    std::vector<Neighbor> exp_node_info;// 存储束搜索过程中发现的节点及其相关信息
+    tsl::robin_map<uint32_t, T *> coord_map;// 以节点ID（uint32_t）为键，节点坐标指针（T*）为值
+    coord_map.reserve(2 * this->l_index);// 预分配 2 * l_index 个槽位
 
     std::vector<uint64_t> page_ref{};
     this->do_beam_search(point, 0, l_index, beam_width, exp_node_info, &coord_map, nullptr, deletion_set, false,
                          &page_ref);
-    std::vector<uint32_t> new_nhood;
-    prune_neighbors(coord_map, exp_node_info, new_nhood);
+    std::vector<uint32_t> new_nhood;// 存储经过剪枝后的新邻居节点ID
+    prune_neighbors(coord_map, exp_node_info, new_nhood);// 对搜索结果进行邻居剪枝
+    if (trace) {
+      LOG(INFO) << "[TraceInsert] id=" << target_id << " step=beam_prune"
+                << " ms=" << (trace_timer.elapsed() / 1.0e3f)
+                << " rss=" << get_rss_mb() << " MB"
+                << " nhood=" << new_nhood.size();
+    }
     // locs[new_nhood.size()] is the target, locs[0:new_nhood.size() - 1] are the neighbors.
     // lock the pages to write
-    std::set<uint64_t> pages_need_to_read;
+    std::set<uint64_t> pages_need_to_read;// 用于页面锁定和读取操作
 
 #ifdef IN_PLACE_RECORD_UPDATE
-    std::vector<uint64_t> locs;
-    for (auto &nbr : new_nhood) {
-      locs.emplace_back(id2loc(nbr));
-      pages_need_to_read.insert(node_sector_no(nbr));
-    }
-    locs.push_back(target_id);
-    pages_need_to_read.insert(loc_sector_no(target_id));
... (diff truncated)
```

## src/search/page_search.cpp
```diff
--- C:\Users\dasua\Desktop\DiskANN\26FAST-PipeANN-main\src\search\page_search.cpp
+++ C:\Users\dasua\Desktop\DiskANN\ODinANN-Demo\src\search\page_search.cpp
@@ -22,11 +22,36 @@
 #include "linux_aligned_file_reader.h"
 
 namespace pipeann {
+  static double get_rss_mb() {
+    std::ifstream f("/proc/self/statm");
+    long t = 0, r = 0;
+    if (f.is_open()) {
+      f >> t >> r;
+    }
+    long page_kb = sysconf(_SC_PAGE_SIZE) / 1024;
+    return (double) r * page_kb / 1024.0;
+  }
 
   template<typename T, typename TagT>
   void SSDIndex<T, TagT>::load_page_layout(const std::string &index_prefix, const _u64 nnodes_per_sector,
                                            const _u64 num_points) {
     std::string partition_file = index_prefix + "_partition.bin.aligned";
+#if defined(NO_MAPPING) || defined(IDENTITY_MAPPING_ONLY)
+    if (std::filesystem::exists(partition_file)) {
+      LOG(ERROR) << "Partition file exists but identity mapping is enabled: " << partition_file;
+      exit(-1);
+    }
+    // Identity mapping: loc == id. Avoid building id2loc_/page_layout to save memory.
+    this->cur_loc = num_points;
+    if (num_points % nnodes_per_sector != 0) {
+      cur_loc += nnodes_per_sector - (num_points % nnodes_per_sector);
+    }
+    LOG(INFO) << "Identity mapping enabled. Skipping page layout build.";
+    LOG(INFO) << "Cur location: " << this->cur_loc;
+    LOG(INFO) << "Page layout loaded.";
+    return;
+#endif
+    LOG(INFO) << "RSS before page layout build: " << get_rss_mb() << " MB";
     if (std::filesystem::exists(partition_file)) {
       LOG(INFO) << "Loading partition file " << partition_file;
       std::ifstream part(partition_file);
@@ -96,6 +121,7 @@
       }
 #endif
     }
+    LOG(INFO) << "RSS after page layout build: " << get_rss_mb() << " MB";
     LOG(INFO) << "Cur location: " << this->cur_loc;
     LOG(INFO) << "Page layout loaded.";
   }
```

## include/ssd_index.h
```diff
--- C:\Users\dasua\Desktop\DiskANN\26FAST-PipeANN-main\include\ssd_index.h
+++ C:\Users\dasua\Desktop\DiskANN\ODinANN-Demo\include\ssd_index.h
@@ -4,6 +4,7 @@
 #include <cstdint>
 #include <string>
 #include <set>
+#include <atomic>
 #include "v2/page_cache.h"
 #include <iostream>
 #include <iomanip>
@@ -295,6 +296,7 @@
     void delta_prune_neighbors_pq(std::vector<TriangleNeighbor> &pool, std::vector<uint32_t> &pruned_list,
                                   uint8_t *scratch, int tgt_idx);
     void reload(const char *index_prefix, uint32_t num_threads);
+    void flush_page_cache();
     // background I/O commit.
     struct BgTask {
       QueryBuffer<T> *thread_data;
@@ -304,6 +306,7 @@
     };
     // its concurrency should not be the bottleneck.
     ConcurrentQueue<BgTask *> bg_tasks = ConcurrentQueue<BgTask *>(nullptr);
+    std::atomic<uint64_t> bg_tasks_inflight{0};
     void bg_io_thread();
     static constexpr int kBgIOThreads = 1;
     std::thread *bg_io_thread_[kBgIOThreads]{nullptr};
@@ -319,7 +322,7 @@
     // if ID == tag, then it is not stored.
     libcuckoo::cuckoohash_map<uint32_t, TagT> tags;
     TagT id2tag(uint32_t id) {
-#ifdef NO_MAPPING
+#if defined(NO_MAPPING) || defined(TAGS_IDENTITY_ONLY)
       return id;  // use ID to replace tags.
 #else
       TagT ret;
@@ -448,7 +451,7 @@
 
     libcuckoo::cuckoohash_map<uint32_t, uint32_t> id2loc_;  // id -> loc (start from 0)
     uint32_t id2loc(uint32_t id) {
-#ifdef NO_MAPPING
+#if defined(NO_MAPPING) || defined(IDENTITY_MAPPING_ONLY)
       return id;
 #else
       uint32_t loc = 0;
@@ -476,14 +479,24 @@
 
     std::mutex alloc_lock;
     uint32_t loc2id(uint32_t loc) {
+#if defined(NO_MAPPING) || defined(IDENTITY_MAPPING_ONLY)
+      uint64_t cur = cur_id.load(std::memory_order_relaxed);
+      return (loc < cur) ? loc : kInvalidID;
+#else
       uint32_t page = loc_sector_no(loc);
       uint32_t offset = loc % nnodes_per_sector;
       uint32_t id = kInvalidID;
       page_layout.find_fn(page, [&](PageArr &v) { id = v[offset]; });
       return id;  // kInvalidID if fails.
+#endif
     }
 
     void set_loc2id(uint32_t loc, uint32_t id) {
+#if defined(NO_MAPPING) || defined(IDENTITY_MAPPING_ONLY)
+      (void) loc;
+      (void) id;
+      return;
+#else
       uint32_t page = loc_sector_no(loc);
       uint32_t offset = loc % nnodes_per_sector;
       page_layout.upsert(page, [&](PageArr &v, libcuckoo::UpsertContext ctx) {
@@ -494,9 +507,14 @@
         }
         v[offset] = id;
       });
+#endif
     }
 
     void erase_loc2id(uint32_t loc) {
+#if defined(NO_MAPPING) || defined(IDENTITY_MAPPING_ONLY)
+      (void) loc;
+      return;
+#else
       uint32_t page = loc_sector_no(loc);
       uint32_t offset = loc % nnodes_per_sector;
       page_layout.upsert(page, [&](PageArr &v) {
@@ -512,6 +530,7 @@
           empty_pages.push(page);
         }
       });
+#endif
     }
 
     ConcurrentQueue<uint32_t> empty_pages = ConcurrentQueue<uint32_t>(kInvalidID);
@@ -620,6 +639,10 @@
     }
 
     void verify_id2loc() {
+#if defined(NO_MAPPING) || defined(IDENTITY_MAPPING_ONLY)
+      LOG(INFO) << "ID2loc consistency check skipped (identity mapping).";
+      return;
+#endif
       // verify id -> loc -> id map.
       LOG(INFO) << "ID2loc size: " << id2loc_.size() << ", cur_loc: " << cur_loc.load() << ", cur_id: " << cur_id
                 << ", nnodes_per_sector: " << nnodes_per_sector;
@@ -657,6 +680,20 @@
                        const std::vector<TagT> &deleted_nodes, const tsl::robin_set<TagT> &deleted_nodes_set,
                        uint32_t nthreads, const uint32_t &n_sampled_nbrs);
 
+    // Full merge graph rewrite with streaming PQ/tags (no large PQ/tags buffers).
+    void merge_deletes_stream_pq_tags(const std::string &in_path_prefix, const std::string &out_path_prefix,
+                                      const std::vector<TagT> &deleted_nodes,
+                                      const tsl::robin_set<TagT> &deleted_nodes_set, uint32_t nthreads,
+                                      const uint32_t &n_sampled_nbrs);
+
+    // Graph-only rebuild: sequentially scan and re-prune neighbor lists in-place.
+    // Does NOT rebuild PQ or tags.
+    void rebuild_graph_in_place(uint32_t nthreads = 0);
+
+    // Graph-only merge to a new prefix. Rewrites disk graph, copies PQ/tags from input.
+    void merge_graph_only(const std::string &in_path_prefix, const std::string &out_path_prefix,
+                          uint32_t nthreads = 0);
... (diff truncated)
```

## tests/search_disk_index.cpp
```diff
--- C:\Users\dasua\Desktop\DiskANN\26FAST-PipeANN-main\tests\search_disk_index.cpp
+++ C:\Users\dasua\Desktop\DiskANN\ODinANN-Demo\tests\search_disk_index.cpp
@@ -4,6 +4,7 @@
 #include <string.h>
 #include <time.h>
 #include <iostream>
+#include <cstdlib>
 
 #include "log.h"
 #include "timer.h"
@@ -39,6 +40,16 @@
   uint32_t *tags = nullptr;
   size_t query_num, query_dim, gt_num, gt_dim;
   std::vector<_u64> Lvec;
+  uint64_t new_start_tag = 0;
+  bool report_new_hits = false;
+  if (const char *env = std::getenv("NEW_START_TAG")) {
+    char *endp = nullptr;
+    uint64_t v = std::strtoull(env, &endp, 10);
+    if (endp != env) {
+      new_start_tag = v;
+      report_new_hits = true;
+    }
+  }
 
   bool tags_flag = true;
 
@@ -192,6 +203,20 @@
     float mean_ios =
         (float) pipeann::get_mean_stats(stats, query_num, [](const pipeann::QueryStats &stats) { return stats.n_ios; });
 
+    double new_hit_pct = 0.0;
+    if (report_new_hits) {
+      const uint64_t total = (uint64_t) recall_at * (uint64_t) query_num;
+      uint64_t hits = 0;
+      for (uint64_t i = 0; i < total; ++i) {
+        if (query_result_tags_32[i] >= new_start_tag) {
+          ++hits;
+        }
+      }
+      if (total > 0) {
+        new_hit_pct = (double) hits * 100.0 / (double) total;
+      }
+    }
+
     delete[] stats;
 
     if (output) {
@@ -207,6 +232,9 @@
       std::cout << std::setw(6) << L << std::setw(12) << beamwidth << std::setw(12) << qps << std::setw(12)
                 << mean_latency << std::setw(12) << latency_999 << std::setw(12) << mean_hops << std::setw(12)
                 << mean_ios;
+      if (report_new_hits) {
+        std::cout << std::setw(12) << new_hit_pct;
+      }
       if (calc_recall_flag) {
         std::cout << std::setw(12) << recall << std::endl;
       }
@@ -228,6 +256,9 @@
   std::cout << std::setw(6) << "L" << std::setw(12) << "I/O Width" << std::setw(12) << "QPS" << std::setw(12)
             << "AvgLat(us)" << std::setw(12) << "P99 Lat" << std::setw(12) << "Mean Hops" << std::setw(12) << "Mean IOs"
             << std::setw(12);
+  if (report_new_hits) {
+    std::cout << std::setw(12) << "NewHit%";
+  }
   if (calc_recall_flag) {
     std::cout << std::setw(12) << recall_string << std::endl;
   } else
```

## CMakeLists.txt
```diff
--- C:\Users\dasua\Desktop\DiskANN\26FAST-PipeANN-main\CMakeLists.txt
+++ C:\Users\dasua\Desktop\DiskANN\ODinANN-Demo\CMakeLists.txt
@@ -29,7 +29,9 @@
 	$ENV{ADDITIONAL_DEFINITIONS}
 	-DNDEBUG
 	# -DUSE_AIO
-	# -DIN_PLACE_RECORD_UPDATE # disable design #1
+	-DIN_PLACE_RECORD_UPDATE # disable design #1
+	#-DTAGS_IDENTITY_ONLY # hardcode tag==id, skip loading tags map
+	-DIDENTITY_MAPPING_ONLY # id==loc, skip id2loc/page_layout build
 	# -DDIRECT_READ_CC # disable opt #1.1
 	# -DBG_IO_THREAD # enable opt #1.2
 	# -DDELTA_PRUNING # enable opt #2
@@ -126,3 +128,28 @@
 add_subdirectory(tests)
 add_subdirectory(tests/utils)
 # add_subdirectory(python)
+add_executable(test_insert tests/test_insert.cpp)
+set_target_properties(test_insert PROPERTIES
+    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests"
+)
+
+
+# 链接库保持不变 
+target_link_libraries(test_insert 
+    pipeann 
+    ${OpenMP_CXX_FLAGS}
+    ${BLAS_LIBRARIES} # 建议加上这个，防止缺少数学库链接
+)
+
+# add_subdirectory(python)
+add_executable(insert_demo tests/insert_demo.cpp)
+set_target_properties(insert_demo PROPERTIES
+    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests"
+)
+
+# 链接库保持不变 
+target_link_libraries(insert_demo 
+    pipeann 
+    ${OpenMP_CXX_FLAGS}
+    ${BLAS_LIBRARIES} # 建议加上这个，防止缺少数学库链接
+)
```
**Report‑Ready Core Diffs (Before → After)**

**1) Peak #1: PQ 扩容峰值（索引复制结束后 / 插入前）**

**Before (旧版：加载 PQ 只读原大小，插入时触发 1.5x 扩容拷贝)**
```cpp
// 26FAST-PipeANN-main/src/ssd_index.cpp
pipeann::load_bin<_u8>(pq_compressed_vectors, data, npts_u64, nchunks_u64, pq_vectors_offset);
this->num_points = this->init_num_pts = npts_u64;
this->n_chunks = nchunks_u64;
```
**功能**：仅加载原始 PQ 大小。  
**副作用**：插入时 data 容量不够 → 触发一次性 1.5x 扩容 + 拷贝（峰值）。

```cpp
// ODinANN-Demo/src/update/direct_insert.cpp
if (this->data.size() < pq_offset + n_chunks) {
  while (this->data.size() < pq_offset + n_chunks) {
    this->data.resize(1.5 * this->data.size());
  }
}
```

**After (新版：加载 PQ 时预留插入容量，避免扩容拷贝)**
```cpp
// ODinANN-Demo/src/ssd_index.cpp
// read PQ header → compute reserve_pts
data.resize(reserve_pts * nchunks_u64);
pq_reader.read((char*)data.data(), npts_u64 * nchunks_u64 * sizeof(_u8));
```
**优化点**：提前预留插入后容量，避免首次插入触发 1.5x 拷贝峰值。  
**功能不变**：PQ 内容一致，仅改变分配时机与峰值行为。

---

**2) Peak #2: Final merge 阶段 PQ/Tags 重建峰值**

**Before (旧版：构造完整 PQ/tags 新数组，再写盘)**
```cpp
// 26FAST-PipeANN-main/src/update/delete_merge.cpp
std::vector<uint8_t> pq_coords(new_npoints * n_chunks, 0);
std::vector<TagT> new_tags(new_npoints);

memcpy(pq_coords.data() + new_id * n_chunks,
       this->data.data() + id * n_chunks, n_chunks);
new_tags[new_id] = id2tag(id);

this->data = std::move(pq_coords);
pipeann::save_bin<TagT>(out_path_prefix + "_disk.index.tags", new_tags->data(), new_npoints, 1, 0);
pipeann::save_bin<uint8_t>(out_path_prefix + "_pq_compressed.bin", this->data.data(), new_npoints, n_chunks);
```
**功能**：删除/重映射时构建新 PQ/tags 并写盘。  
**副作用**：再申请一份完整 PQ/tags → 峰值≈旧 PQ + 新 PQ。

**After (新版：merge_stream 直接流式写 PQ/tags，不再建大数组)**
```cpp
// ODinANN-Demo/src/update/delete_merge.cpp (no-deletes fast path)
std::fstream pq_io(out_path_prefix + "_pq_compressed.bin", std::ios::trunc);
std::fstream tags_io(out_path_prefix + "_disk.index.tags", std::ios::trunc);

// per-node
tags_io.seekp(2*sizeof(uint32_t) + new_id*sizeof(TagT));
tags_io.write((char*)&tag, sizeof(TagT));
pq_io.seekp(2*sizeof(uint32_t) + new_id*n_chunks);
pq_io.write((char*)(this->data.data() + id*n_chunks), n_chunks);
```
**优化点**：不再分配完整 PQ/tags 缓冲区；边扫边写磁盘。  
**功能不变**：PQ/tags 仍完整落盘，但避免二次峰值。

---

**Summary (For PPT)**
Diff legend: lines starting with `+` are added, `-` are removed, and leading space means unchanged context.
Purpose: enable low‑memory incremental insert + merge flow while keeping recall, and provide observability for memory/IO bottlenecks.

Key Functional Changes
1. Added a dedicated insert tool `tests/insert_demo.cpp` for incremental insert + stream merge.
2. Added `merge_stream` path to perform full graph merge while streaming PQ/tags to disk (no extra full PQ rebuild).
3. Ensured insert writes are durable before merge by waiting for background IO and flushing page cache.
4. Pre‑extended disk index file for merge modes to prevent “write past EOF → missing edges” issues.
5. Reduced PQ expansion peak by pre‑reserving PQ capacity based on planned insert count.
6. Added identity‑mapping options to skip id2loc/page_layout and tags map when safe (reduces memory).
7. Added fine‑grained trace logging to pinpoint memory spikes and IO phases.
8. Added optional NewHit% metric to search to verify connectivity to new inserts.

Files and What Changed
1. `tests/insert_demo.cpp`
Added a full insert driver with `merge_stream` mode, pre‑extend, background IO flush before merge, PQ/tags header verification, and detailed progress logs.
2. `src/ssd_index.cpp`
Added PQ pre‑reserve logic using `PIPEANN_PQ_EXTRA_POINTS`, plus RSS logging around PQ/tags/page_layout loading.
3. `src/update/delete_merge.cpp`
Added `merge_deletes_stream_pq_tags` fast path for no‑delete scenario; streams PQ/tags, rewrites graph by sequential scan.
4. `src/update/direct_insert.cpp`
Added trace hooks (`PIPEANN_TRACE_INSERT`) to log per‑insert phases and memory; preserves in‑place update path.
5. `src/search/page_search.cpp`
Added identity mapping shortcut and RSS logging; skips page layout build when id==loc.
6. `include/ssd_index.h`
Added identity‑mapping helpers (`id2loc/loc2id/id2tag`) guarded by `IDENTITY_MAPPING_ONLY` and `TAGS_IDENTITY_ONLY`.
7. `tests/search_disk_index.cpp`
Added `NEW_START_TAG` support and NewHit% column to confirm reachability of newly inserted points.
8. `CMakeLists.txt`
Added build flags to enable in‑place updates, identity mapping, and tag identity options when applicable.

Behavioral Impact (What you can claim)
1. Insert remains graph‑aware (beamsearch + neighbor updates).
2. Merge becomes streaming for PQ/tags, reducing peak memory spikes from PQ rebuild.
3. Early IO flush + pre‑extend prevents graph loss that caused recall to stall at ~80%.
4. Optional identity mapping/tag identity significantly lowers memory on large datasets.

**Deeper Details (Requested Clarifications)**

**A) PQ pre‑reserve: how much is reserved, and what happens when it fills**

**Optimized logic (ODinANN‑Demo/src/ssd_index.cpp)**  
Reserve size is chosen at PQ load time:
```cpp
uint64_t reserve_pts = npts_u64;
const char* expected_env = std::getenv("PIPEANN_PQ_EXPECTED_POINTS");
const char* extra_env = std::getenv("PIPEANN_PQ_EXTRA_POINTS");
if (expected_env && expected_env[0] != '\0') {
  reserve_pts = std::max<uint64_t>(reserve_pts, std::strtoull(expected_env, nullptr, 10));
}
if (extra_env && extra_env[0] != '\0') {
  uint64_t extra_pts = std::strtoull(extra_env, nullptr, 10);
  reserve_pts = std::max<uint64_t>(reserve_pts, npts_u64 + extra_pts);
}
data.resize(reserve_pts * nchunks_u64);
```

**Interpretation**  
1. If no env var is set, reserve size equals the original PQ size (`reserve_pts = npts`).  
2. If you set `PIPEANN_PQ_EXPECTED_POINTS=TOTAL_AFTER_INSERT`, PQ is pre‑sized to that total.  
3. If you set `PIPEANN_PQ_EXTRA_POINTS=EXTRA`, PQ is pre‑sized to `npts + EXTRA`.  

**If PQ exceeds reserve**  
It still uses the existing dynamic growth in `direct_insert.cpp`, so it will expand by 1.5x when needed:
```cpp
if (this->data.size() < pq_offset + n_chunks) {
  while (this->data.size() < pq_offset + n_chunks) {
    this->data.resize(1.5 * this->data.size());
  }
}
```

**B) Final merge: what “streaming PQ/tags” actually writes**

In `merge_stream` (no deletes), PQ is already updated during `direct_insert`.  
Final merge does two things:
1. Sequentially scan the disk graph, re‑prune neighbor lists, and write back graph pages.
2. Stream PQ/tags to disk **by id** (no large in‑memory rebuild).

**Optimized streaming write (ODinANN‑Demo/src/update/delete_merge.cpp)**  
```cpp
std::fstream pq_io(out_path_prefix + "_pq_compressed.bin",
                   std::ios::binary | std::ios::in | std::ios::out | std::ios::trunc);
std::fstream tags_io(out_path_prefix + "_disk.index.tags",
                     std::ios::binary | std::ios::in | std::ios::out | std::ios::trunc);
pq_io.write((char*)&npts_u32, sizeof(uint32_t));
pq_io.write((char*)&ndims_u32, sizeof(uint32_t));
tags_io.write((char*)&npts_u32, sizeof(uint32_t));
tags_io.write((char*)&tag_ndims_u32, sizeof(uint32_t));

// inside per‑node loop:
TagT tag = (TagT) id2tag(id);
std::streamoff tag_off = 2 * sizeof(uint32_t) + (std::streamoff)new_id * sizeof(TagT);
tags_io.seekp(tag_off);
tags_io.write((char*)&tag, sizeof(TagT));

std::streamoff pq_off = 2 * sizeof(uint32_t) + (std::streamoff)new_id * n_chunks;
pq_io.seekp(pq_off);
pq_io.write((char*)(this->data.data() + id * n_chunks), n_chunks);
```

**Key point**  
Final merge does **not recompute PQ**. It only writes the already‑updated PQ codes and tags to disk while graph is rewritten.

**C) “No deletes” optimization vs original delete‑capable path**

Original code must handle deletions and ID remapping, so it builds full new PQ/tags buffers:
```cpp
// 26FAST-PipeANN-main/src/update/delete_merge.cpp (old)
std::vector<uint8_t> pq_coords(new_npoints * n_chunks, 0);
std::vector<TagT> new_tags(new_npoints);

// ... during scan
memcpy(pq_coords.data() + new_id * n_chunks,
       this->data.data() + id * n_chunks, n_chunks);
new_tags[new_id] = id2tag(id);

// after scan
this->data = std::move(pq_coords);
pipeann::save_bin<TagT>(out_path_prefix + "_disk.index.tags", new_tags->data(), new_npoints, 1, 0);
pipeann::save_bin<uint8_t>(out_path_prefix + "_pq_compressed.bin", this->data.data(), new_npoints, n_chunks);
```

Optimized “insert‑only” path skips deletion logic and avoids building full PQ/tags buffers:
```cpp
// ODinANN-Demo/src/update/delete_merge.cpp (new)
const bool no_deletes = deleted_nodes.empty() || deleted_nodes_set.empty();
if (no_deletes) {
  // stream PQ/tags directly to disk (no pq_coords/new_tags allocations)
  std::fstream pq_io(...); std::fstream tags_io(...);
  // ... per-node: write tag + pq bytes to correct offset
}
```

**D) test_insert_search → insert_demo: search logic removed**

Old test program includes query loading + search evaluation:
```cpp
// 26FAST-PipeANN-main/tests/test_insert_search.cpp (old)
load_aligned_bin(...)  // queries
search_index.search(...)  // evaluate recall/QPS
```

New insert tool focuses only on insert/merge workflow:
```cpp
// ODinANN-Demo/tests/insert_demo.cpp (new)
insert_in_place(...)
final_merge_stream_pq_tags(...)
// no search path in this tool
```

**E) Tags status: still preserved or removed?**

Tags are still supported and written to disk.  
They are **only skipped** when you explicitly enable identity mapping:
```cpp
#ifdef TAGS_IDENTITY_ONLY
  LOG(INFO) << "Tags identity mapping compiled-in. Skipping tag load.";
  return;
#endif
```

So by default, tags remain intact and are streamed in merge.  
Only when you enable `-DTAGS_IDENTITY_ONLY` (or set identity env flags) do tags become `tag == id` and are not loaded to save memory.
