#include "aligned_file_reader.h"
#include "libcuckoo/cuckoohash_map.hh"
#include "ssd_index.h"
#include <malloc.h>
#include <algorithm>
#include <filesystem>

#include <omp.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <tuple>
#include <unordered_set>
#include "timer.h"
#include "tsl/robin_map.h"
#include "utils.h"
#include "v2/page_cache.h"

#include <unistd.h>
#include <sys/syscall.h>
#include "linux_aligned_file_reader.h"

namespace pipeann {
#define SECTORS_PER_MERGE 65536
  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::merge_deletes(const std::string &in_path_prefix, const std::string &out_path_prefix,
                                        const std::vector<TagT> &deleted_nodes,
                                        const tsl::robin_set<TagT> &deleted_nodes_set, uint32_t nthreads,
                                        const uint32_t &n_sampled_nbrs) {
    if (nthreads == 0) {
      nthreads = this->max_nthreads;
    }

    void *ctx = reader->get_ctx();// 从文件读取器获取上下文指针

    while (!bg_tasks.empty()) {
      sleep(5);  // 循环检查后台任务队列是否为空 确保所有异步IO操作已完成 simple way to wait for background IO thread.
    }
    std::string disk_index_out = out_path_prefix + "_disk.index";
    // Note that the index is immutable currently.
    // Step 1: populate neighborhoods, allocate IDs.
    libcuckoo::cuckoohash_map<uint32_t, uint32_t> id_map;                       // old_id -> new_id
    libcuckoo::cuckoohash_map<uint32_t, std::vector<uint32_t>> deleted_nhoods;  // id -> nhood
    std::atomic<uint64_t> new_npoints = 0;
    Timer delete_timer;

    char *rbuf = nullptr, *wbuf = nullptr;
    alloc_aligned((void **) &rbuf, SECTORS_PER_MERGE * SECTOR_LEN, SECTOR_LEN);// 读取缓冲区
    alloc_aligned((void **) &wbuf, 2 * SECTORS_PER_MERGE * SECTOR_LEN, SECTOR_LEN);  // 写入缓冲区 sliding window buffer.
    uint64_t n_sectors = (cur_loc + nnodes_per_sector - 1) / nnodes_per_sector;// 计算索引总扇区数
    LOG(INFO) << "Cur loc: " << cur_loc.load() << ", cur ID: " << cur_id << ", n_sectors: " << n_sectors
              << ", nnodes_per_sector: " << nnodes_per_sector;

    constexpr int SECTORS_PER_POPULATE = 128;             // 每次处理的扇区数 small to avoid blocking search threads.
    uint32_t populate_nthreads = std::min(nthreads, 4u);  // 实际用于填充操作的线程数 restrict the flow.
    // 循环处理所有扇区
    for (uint64_t in_sector = 0; in_sector < n_sectors; in_sector += SECTORS_PER_POPULATE) {
      uint64_t st_sector = in_sector, ed_sector = std::min(in_sector + SECTORS_PER_POPULATE, n_sectors);
      uint64_t loc_st = st_sector * nnodes_per_sector, loc_ed = std::min(cur_loc.load(), ed_sector * nnodes_per_sector);
      uint64_t n_sectors_to_read = ed_sector - st_sector;
      std::vector<IORequest> read_reqs;
      read_reqs.push_back(IORequest(loc_sector_no(loc_st) * SECTOR_LEN, n_sectors_to_read * size_per_io, rbuf, 0, 0));
      reader->read(read_reqs, ctx, false);

#pragma omp parallel for num_threads(populate_nthreads)
      for (uint64_t loc = loc_st; loc < loc_ed; ++loc) {
        // populate nhood.
        uint64_t id = loc2id(loc);
        if (id == kInvalidID || id == kAllocatedID) {
          continue;
        }

        uint64_t tag = id2tag(id);
        if (deleted_nodes_set.find(tag) == deleted_nodes_set.end()) {  // 2. not deleted, alloc ID.
          // allocate ID.
          uint64_t new_id = new_npoints.fetch_add(1);
          id_map.insert(id, new_id);
          continue;
        }

        // 3. deleted, populate nhoods.
        auto page_rbuf = rbuf + (loc / nnodes_per_sector - st_sector) * SECTOR_LEN;
        auto node_rbuf = offset_to_loc(page_rbuf, loc);
        DiskNode<T> node(id, offset_to_node_coords(node_rbuf), offset_to_node_nhood(node_rbuf));
        std::vector<uint32_t> nhood;
        for (uint32_t i = 0; i < node.nnbrs; ++i) {
          uint32_t nbr_tag = id2tag(node.nbrs[i]);
          if (deleted_nodes_set.find(nbr_tag) == deleted_nodes_set.end()) {
            nhood.push_back(node.nbrs[i]);  // filtered neighborhoods.
          }
        }
        // sample for less space consumption.
        if (nhood.size() > n_sampled_nbrs) {
          // std::shuffle(nhood.begin(), nhood.end(), std::default_random_engine());
          nhood.resize(n_sampled_nbrs);  // nearest.
        }
        deleted_nhoods.insert(id, nhood);
      }
    }
    LOG(INFO) << "Finished populating neighborhoods, totally elapsed: " << delete_timer.elapsed() / 1e3
              << "ms, new npoints: " << new_npoints.load() << " " << "id_map size: " << id_map.size();

    // Step 2: prune neighbors, populate PQ and tags.
    int fd = open(disk_index_out.c_str(), O_DIRECT | O_LARGEFILE | O_RDWR | O_CREAT, 0755);
    const uint64_t kVecInWBuf = 2 * SECTORS_PER_MERGE * nnodes_per_sector;
    uint64_t wb_id = 0;
    std::atomic<uint64_t> n_used_id = 0;
    auto write_back = [&]() {
      // write one buffer.
      uint64_t buf_id = (wb_id % kVecInWBuf) / (nnodes_per_sector * SECTORS_PER_MERGE);
      auto b = wbuf + buf_id * SECTORS_PER_MERGE * SECTOR_LEN;
      std::vector<IORequest> write_reqs;
      uint64_t id_delta = std::min((uint64_t) SECTORS_PER_MERGE * nnodes_per_sector, n_used_id - wb_id);
      write_reqs.push_back(IORequest(loc_sector_no(wb_id) * SECTOR_LEN,
                                     ROUND_UP(id_delta, nnodes_per_sector) / nnodes_per_sector * size_per_io, b, 0, 0));
      reader->write_fd(fd, write_reqs, ctx);
      wb_id += id_delta;
      LOG(INFO) << "Write back " << wb_id << "/" << n_used_id << " IDs.";
    };

    std::vector<uint8_t> pq_coords(new_npoints * n_chunks, 0);
    std::vector<TagT> new_tags(new_npoints);

    for (uint64_t in_sector = 0; in_sector < n_sectors; in_sector += SECTORS_PER_MERGE) {
      uint64_t st_sector = in_sector, ed_sector = std::min(in_sector + SECTORS_PER_MERGE, n_sectors);
      uint64_t loc_st = st_sector * nnodes_per_sector, loc_ed = std::min(cur_loc.load(), ed_sector * nnodes_per_sector);
      uint64_t n_sectors_to_read = ed_sector - st_sector;
      std::vector<IORequest> read_reqs;
      read_reqs.push_back(IORequest(loc_sector_no(loc_st) * SECTOR_LEN, n_sectors_to_read * size_per_io, rbuf, 0, 0));
      reader->read(read_reqs, ctx, false);  // read in fd

#pragma omp parallel for num_threads(nthreads)
      for (uint64_t loc = loc_st; loc < loc_ed; ++loc) {
          uint64_t id = loc2id(loc);
          if (id == kInvalidID || id == kAllocatedID) {
            continue;
          }

        uint64_t tag = id2tag(id);
        if (deleted_nodes_set.find(tag) != deleted_nodes_set.end()) {  // deleted.
          continue;
        }

        auto page_rbuf = rbuf + (loc / nnodes_per_sector - st_sector) * SECTOR_LEN;
        auto loc_rbuf = offset_to_loc(page_rbuf, loc);
        DiskNode<T> node(id, offset_to_node_coords(loc_rbuf), offset_to_node_nhood(loc_rbuf));
        // prune neighbors.
        std::unordered_set<uint32_t> nhood_set;
        for (uint32_t i = 0; i < node.nnbrs; ++i) {
          uint32_t nbr_tag = id2tag(node.nbrs[i]);
          if (deleted_nodes_set.find(nbr_tag) != deleted_nodes_set.end()) {
            // deleted, insert neighbors.
            const auto &nhoods = deleted_nhoods.find(node.nbrs[i]);
            nhood_set.insert(nhoods.begin(), nhoods.end());
          } else {
            nhood_set.insert(node.nbrs[i]);
            // LOG(INFO) << id << " insert " << node.nbrs[i];
          }
        }
        nhood_set.erase(id);  // remove self.
        std::vector<uint32_t> nhood(nhood_set.begin(), nhood_set.end());

        if (nhood.size() > this->range) {
          std::vector<float> dists(nhood.size(), 0.0f);
          std::vector<Neighbor> pool(nhood.size());
          auto &thread_pq_buf = thread_pq_bufs[omp_get_thread_num()];
          compute_pq_dists(id, nhood.data(), dists.data(), (_u32) nhood.size(), thread_pq_buf);

          for (uint32_t k = 0; k < nhood.size(); k++) {
            pool[k].id = nhood[k];
            pool[k].distance = dists[k];
          }
          std::sort(pool.begin(), pool.end());
          if (pool.size() > this->maxc) {
            pool.resize(this->maxc);
          }
          nhood.clear();
          this->prune_neighbors_pq(pool, nhood, thread_pq_buf);
        }

        // map to new IDs.
        for (auto &nbr : nhood) {
          nbr = id_map.find(nbr);
        }

        // write neighbors.
        uint64_t new_id = id_map.find(id);
        uint64_t off = new_id % kVecInWBuf;
        auto page_wbuf = wbuf + (off / nnodes_per_sector) * SECTOR_LEN;
        auto loc_wbuf = offset_to_loc(page_wbuf, off);
        DiskNode<T> w_node(new_id, offset_to_node_coords(loc_wbuf), offset_to_node_nhood(loc_wbuf));
        memcpy(w_node.coords, node.coords, data_dim * sizeof(T));
        w_node.nnbrs = nhood.size();
        *(w_node.nbrs - 1) = w_node.nnbrs;
        memcpy(w_node.nbrs, nhood.data(), w_node.nnbrs * sizeof(uint32_t));
        ++n_used_id;
        // copy PQ and tags.
        memcpy(pq_coords.data() + new_id * n_chunks, this->data.data() + id * n_chunks, n_chunks);
        new_tags[new_id] = id2tag(id);
      }

      LOG(INFO) << "Processed " << ed_sector << "/" << n_sectors << " sectors, n_used_id: " << n_used_id << ".";
      if (n_used_id - wb_id >= SECTORS_PER_MERGE * nnodes_per_sector) {
        write_back();
      }
    }

    while (wb_id < n_used_id) {
      write_back();
    }
    LOG(INFO) << "Write nhoods finished, totally elapsed " << delete_timer.elapsed() / 1e3 << "ms.";

    uint32_t medoid = this->medoids[0];
    while (deleted_nodes_set.find(id2tag(medoid)) != deleted_nodes_set.end()) {
      LOG(INFO) << "Medoid deleted. Choosing another start node.";
      const auto &nhoods = deleted_nhoods.find(medoid);
      medoid = nhoods[0];
    }
    close(fd);
    // free buf
    aligned_free((void *) rbuf);
    aligned_free((void *) wbuf);

    // set metadata, PQ and tags.
    merge_lock.lock();  // unlock in reload().
    // metadata.
    this->num_points = new_npoints;
    this->medoids[0] = id_map.find(medoid);
    // PQ.
    this->data = std::move(pq_coords);
    // tags.
    tags.clear();
    id2loc_.clear();
    page_layout.clear();
#pragma omp parallel for num_threads(nthreads)
    for (size_t i = 0; i < new_tags.size(); ++i) {
      tags.insert_or_assign(i, new_tags[i]);
      // TODO(gh): use partition data to init id2loc_ and page_layout.
      id2loc_.insert_or_assign(i, i);
      set_loc2id(i, i);
    }

    this->write_metadata_and_pq(in_path_prefix, out_path_prefix, new_npoints, id_map.find(medoid), &new_tags);
    LOG(INFO) << "Write metadata and PQ finished, totally elapsed " << delete_timer.elapsed() / 1e3 << "ms.";
    LOG(INFO) << "Write metadata finished, totally elapsed " << delete_timer.elapsed() / 1e3 << "ms.";
  }

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::merge_deletes_stream_pq_tags(const std::string &in_path_prefix,
                                                       const std::string &out_path_prefix,
                                                       const std::vector<TagT> &deleted_nodes,
                                                       const tsl::robin_set<TagT> &deleted_nodes_set,
                                                       uint32_t nthreads, const uint32_t &n_sampled_nbrs) {
    if (nthreads == 0) {
      nthreads = this->max_nthreads;
    }

    void *ctx = reader->get_ctx();
    while (!bg_tasks.empty()) {
      sleep(5);
    }
    while (bg_tasks_inflight.load(std::memory_order_relaxed) != 0) {
      sleep(1);
    }
    // ensure any cached updates are flushed before scan/write
    this->flush_page_cache();
    std::string disk_index_out = out_path_prefix + "_disk.index";

    libcuckoo::cuckoohash_map<uint32_t, uint32_t> id_map;                       // old_id -> new_id
    libcuckoo::cuckoohash_map<uint32_t, std::vector<uint32_t>> deleted_nhoods;  // id -> nhood
    std::atomic<uint64_t> new_npoints = 0;
    Timer delete_timer;

    char *rbuf = nullptr, *wbuf = nullptr;
    alloc_aligned((void **) &rbuf, SECTORS_PER_MERGE * SECTOR_LEN, SECTOR_LEN);
    alloc_aligned((void **) &wbuf, 2 * SECTORS_PER_MERGE * SECTOR_LEN, SECTOR_LEN);  // sliding window buffer.
    uint64_t n_sectors = (cur_loc + nnodes_per_sector - 1) / nnodes_per_sector;
    LOG(INFO) << "Cur loc: " << cur_loc.load() << ", cur ID: " << cur_id << ", n_sectors: " << n_sectors
              << ", nnodes_per_sector: " << nnodes_per_sector;

    constexpr int SECTORS_PER_POPULATE = 128;
    uint32_t populate_nthreads = std::min(nthreads, 4u);
    for (uint64_t in_sector = 0; in_sector < n_sectors; in_sector += SECTORS_PER_POPULATE) {
      uint64_t st_sector = in_sector, ed_sector = std::min(in_sector + SECTORS_PER_POPULATE, n_sectors);
      uint64_t loc_st = st_sector * nnodes_per_sector, loc_ed = std::min(cur_loc.load(), ed_sector * nnodes_per_sector);
      uint64_t n_sectors_to_read = ed_sector - st_sector;
      std::vector<IORequest> read_reqs;
      read_reqs.push_back(IORequest(loc_sector_no(loc_st) * SECTOR_LEN, n_sectors_to_read * size_per_io, rbuf, 0, 0));
      reader->read(read_reqs, ctx, false);

#pragma omp parallel for num_threads(populate_nthreads)
      for (uint64_t loc = loc_st; loc < loc_ed; ++loc) {
        uint64_t id = loc2id(loc);
        if (id == kInvalidID) {
          continue;
        }

        uint64_t tag = id2tag(id);
        if (deleted_nodes_set.find(tag) == deleted_nodes_set.end()) {
          uint64_t new_id = new_npoints.fetch_add(1);
          id_map.insert(id, new_id);
          continue;
        }

        auto page_rbuf = rbuf + (loc / nnodes_per_sector - st_sector) * SECTOR_LEN;
        auto node_rbuf = offset_to_loc(page_rbuf, loc);
        DiskNode<T> node(id, offset_to_node_coords(node_rbuf), offset_to_node_nhood(node_rbuf));
        std::vector<uint32_t> nhood;
        for (uint32_t i = 0; i < node.nnbrs; ++i) {
          uint32_t nbr_tag = id2tag(node.nbrs[i]);
          if (deleted_nodes_set.find(nbr_tag) == deleted_nodes_set.end()) {
            nhood.push_back(node.nbrs[i]);
          }
        }
        if (nhood.size() > n_sampled_nbrs) {
          nhood.resize(n_sampled_nbrs);
        }
        deleted_nhoods.insert(id, nhood);
      }
    }
    LOG(INFO) << "Finished populating neighborhoods, totally elapsed: " << delete_timer.elapsed() / 1e3
              << "ms, new npoints: " << new_npoints.load() << " " << "id_map size: " << id_map.size();

    int fd = open(disk_index_out.c_str(), O_DIRECT | O_LARGEFILE | O_RDWR | O_CREAT, 0755);
    const uint64_t kVecInWBuf = 2 * SECTORS_PER_MERGE * nnodes_per_sector;
    uint64_t wb_id = 0;
    std::atomic<uint64_t> n_used_id = 0;
    auto write_back = [&]() {
      uint64_t buf_id = (wb_id % kVecInWBuf) / (nnodes_per_sector * SECTORS_PER_MERGE);
      auto b = wbuf + buf_id * SECTORS_PER_MERGE * SECTOR_LEN;
      std::vector<IORequest> write_reqs;
      uint64_t id_delta = std::min((uint64_t) SECTORS_PER_MERGE * nnodes_per_sector, n_used_id - wb_id);
      write_reqs.push_back(IORequest(loc_sector_no(wb_id) * SECTOR_LEN,
                                     ROUND_UP(id_delta, nnodes_per_sector) / nnodes_per_sector * size_per_io, b, 0, 0));
      reader->write_fd(fd, write_reqs, ctx);
      wb_id += id_delta;
      LOG(INFO) << "Write back " << wb_id << "/" << n_used_id << " IDs.";
    };

    const uint64_t new_npoints_val = new_npoints.load();
    const uint64_t file_size =
        SECTOR_LEN + ROUND_UP(new_npoints_val, nnodes_per_sector) / nnodes_per_sector * SECTOR_LEN;
    ::ftruncate(fd, file_size);

    // prepare PQ/tags output streams (streaming, no large buffers)
    std::fstream pq_io(out_path_prefix + "_pq_compressed.bin",
                       std::ios::binary | std::ios::in | std::ios::out | std::ios::trunc);
    std::fstream tags_io(out_path_prefix + "_disk.index.tags",
                         std::ios::binary | std::ios::in | std::ios::out | std::ios::trunc);
    if (!pq_io.is_open() || !tags_io.is_open()) {
      LOG(ERROR) << "Failed to open PQ/tags output for streaming.";
      close(fd);
      aligned_free((void *) rbuf);
      aligned_free((void *) wbuf);
      return;
    }
    uint32_t npts_u32 = (uint32_t) new_npoints_val;
    uint32_t ndims_u32 = (uint32_t) this->n_chunks;
    pq_io.write((char *) &npts_u32, sizeof(uint32_t));
    pq_io.write((char *) &ndims_u32, sizeof(uint32_t));
    uint32_t tag_ndims_u32 = 1;
    tags_io.write((char *) &npts_u32, sizeof(uint32_t));
    tags_io.write((char *) &tag_ndims_u32, sizeof(uint32_t));

    for (uint64_t in_sector = 0; in_sector < n_sectors; in_sector += SECTORS_PER_MERGE) {
      uint64_t st_sector = in_sector, ed_sector = std::min(in_sector + SECTORS_PER_MERGE, n_sectors);
      uint64_t loc_st = st_sector * nnodes_per_sector, loc_ed = std::min(cur_loc.load(), ed_sector * nnodes_per_sector);
      uint64_t n_sectors_to_read = ed_sector - st_sector;
      std::vector<IORequest> read_reqs;
      read_reqs.push_back(IORequest(loc_sector_no(loc_st) * SECTOR_LEN, n_sectors_to_read * size_per_io, rbuf, 0, 0));
      reader->read(read_reqs, ctx, false);

#pragma omp parallel for num_threads(nthreads)
      for (uint64_t loc = loc_st; loc < loc_ed; ++loc) {
        uint64_t id = loc2id(loc);
        if (id == kInvalidID) {
          continue;
        }

        uint64_t tag = id2tag(id);
        if (deleted_nodes_set.find(tag) != deleted_nodes_set.end()) {
          continue;
        }

        auto page_rbuf = rbuf + (loc / nnodes_per_sector - st_sector) * SECTOR_LEN;
        auto loc_rbuf = offset_to_loc(page_rbuf, loc);
        DiskNode<T> node(id, offset_to_node_coords(loc_rbuf), offset_to_node_nhood(loc_rbuf));

        std::unordered_set<uint32_t> nhood_set;
        nhood_set.reserve(node.nnbrs * 2 + 1);
        for (uint32_t i = 0; i < node.nnbrs; ++i) {
          uint32_t nbr_tag = id2tag(node.nbrs[i]);
          if (deleted_nodes_set.find(nbr_tag) != deleted_nodes_set.end()) {
            const auto &nhoods = deleted_nhoods.find(node.nbrs[i]);
            nhood_set.insert(nhoods.begin(), nhoods.end());
          } else {
            nhood_set.insert(node.nbrs[i]);
          }
        }
        nhood_set.erase(id);
        std::vector<uint32_t> nhood(nhood_set.begin(), nhood_set.end());

        if (nhood.size() > this->range) {
          std::vector<float> dists(nhood.size(), 0.0f);
          std::vector<Neighbor> pool(nhood.size());
          auto &thread_pq_buf = thread_pq_bufs[omp_get_thread_num()];
          compute_pq_dists(id, nhood.data(), dists.data(), (_u32) nhood.size(), thread_pq_buf);

          for (uint32_t k = 0; k < nhood.size(); k++) {
            pool[k].id = nhood[k];
            pool[k].distance = dists[k];
          }
          std::sort(pool.begin(), pool.end());
          if (pool.size() > this->maxc) {
            pool.resize(this->maxc);
          }
          nhood.clear();
          this->prune_neighbors_pq(pool, nhood, thread_pq_buf);
        }

        for (auto &nbr : nhood) {
          nbr = id_map.find(nbr);
        }

        uint64_t new_id = id_map.find(id);
        uint64_t off = new_id % kVecInWBuf;
        auto page_wbuf = wbuf + (off / nnodes_per_sector) * SECTOR_LEN;
        auto loc_wbuf = offset_to_loc(page_wbuf, off);
        DiskNode<T> w_node(new_id, offset_to_node_coords(loc_wbuf), offset_to_node_nhood(loc_wbuf));
        memcpy(w_node.coords, node.coords, data_dim * sizeof(T));
        w_node.nnbrs = nhood.size();
        *(w_node.nbrs - 1) = w_node.nnbrs;
        memcpy(w_node.nbrs, nhood.data(), w_node.nnbrs * sizeof(uint32_t));
        ++n_used_id;

        // stream PQ + tag
#pragma omp critical
        {
          std::streamoff tag_off = (std::streamoff) (2 * sizeof(uint32_t)) +
                                   (std::streamoff) new_id * (std::streamoff) sizeof(TagT);
          tags_io.seekp(tag_off, std::ios::beg);
          tags_io.write((char *) &tag, sizeof(TagT));
          std::streamoff pq_off = (std::streamoff) (2 * sizeof(uint32_t)) +
                                  (std::streamoff) new_id * (std::streamoff) n_chunks;
          pq_io.seekp(pq_off, std::ios::beg);
          pq_io.write((char *) (this->data.data() + id * n_chunks), n_chunks);
        }
      }

      LOG(INFO) << "Processed " << ed_sector << "/" << n_sectors << " sectors, n_used_id: " << n_used_id << ".";
      if (n_used_id - wb_id >= SECTORS_PER_MERGE * nnodes_per_sector) {
        write_back();
      }
    }

    while (wb_id < n_used_id) {
      write_back();
    }
    LOG(INFO) << "Write nhoods finished, totally elapsed " << delete_timer.elapsed() / 1e3 << "ms.";

    uint32_t medoid = this->medoids[0];
    while (deleted_nodes_set.find(id2tag(medoid)) != deleted_nodes_set.end()) {
      LOG(INFO) << "Medoid deleted. Choosing another start node.";
      const auto &nhoods = deleted_nhoods.find(medoid);
      medoid = nhoods[0];
    }
    close(fd);
    aligned_free((void *) rbuf);
    aligned_free((void *) wbuf);

    // update metadata header (do NOT truncate, graph already written)
    {
      std::vector<uint64_t> output_metadata;
      output_metadata.push_back(new_npoints_val);
      output_metadata.push_back((uint64_t) this->data_dim);
      output_metadata.push_back(medoid);
      output_metadata.push_back(this->max_node_len);
      output_metadata.push_back(nnodes_per_sector);
      output_metadata.push_back(this->num_frozen_points);
      output_metadata.push_back(this->frozen_location);
      output_metadata.push_back(file_size);

      std::fstream io(disk_index_out, std::ios::binary | std::ios::in | std::ios::out);
      if (!io.is_open()) {
        LOG(ERROR) << "Failed to open disk index for header update: " << disk_index_out;
      } else {
        uint32_t npts_hdr = (uint32_t) output_metadata.size();
        uint32_t ndims_hdr = 1;
        io.seekp(0, std::ios::beg);
        io.write((char *) &npts_hdr, sizeof(uint32_t));
        io.write((char *) &ndims_hdr, sizeof(uint32_t));
        io.write((char *) output_metadata.data(),
                 (std::streamsize) (output_metadata.size() * sizeof(uint64_t)));
        io.flush();
      }
    }

    pq_io.close();
    tags_io.close();

    if (in_path_prefix != out_path_prefix) {
      std::filesystem::copy(in_path_prefix + "_pq_pivots.bin", out_path_prefix + "_pq_pivots.bin",
                            std::filesystem::copy_options::overwrite_existing);
    }
  }


  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::write_metadata_and_pq(const std::string &in_path_prefix, const std::string &out_path_prefix,
                                                const uint64_t &new_npoints, const uint64_t &new_medoid,
                                                std::vector<TagT> *new_tags) {
    uint64_t file_size = SECTOR_LEN + ROUND_UP(new_npoints, nnodes_per_sector) / nnodes_per_sector * SECTOR_LEN;
    std::vector<uint64_t> output_metadata;
    output_metadata.push_back(new_npoints);
    output_metadata.push_back((uint64_t) this->data_dim);

    output_metadata.push_back(new_medoid);  // mapped medoid
    output_metadata.push_back(this->max_node_len);
    output_metadata.push_back(nnodes_per_sector);
    output_metadata.push_back(this->num_frozen_points);
    output_metadata.push_back(this->frozen_location);
    output_metadata.push_back(file_size);
    LOG(INFO) << "New metadata: " << "num points: " << new_npoints << " data dim: " << this->data_dim
              << " medoid: " << new_medoid << " max node len: " << this->max_node_len;
    LOG(INFO) << "Nnodes per sector: " << nnodes_per_sector << " num frozen points: " << this->num_frozen_points
              << " frozen location: " << this->frozen_location << " file size: " << file_size / 1024 / 1024 << "MB";

    std::string disk_index_out = out_path_prefix + "_disk.index";
    pipeann::save_bin<uint64_t>(disk_index_out, output_metadata.data(), output_metadata.size(), 1, 0);
    std::ignore = truncate(disk_index_out.c_str(), file_size);

    // Step 3. Write tags and PQ.
    std::vector<TagT> tags_vec;
    if (new_tags == nullptr) {
      tags_vec.resize(new_npoints);
      for (uint64_t i = 0; i < new_npoints; ++i) {
        tags_vec[i] = id2tag(i);
      }
      new_tags = &tags_vec;
    }
    pipeann::save_bin<TagT>(out_path_prefix + "_disk.index.tags", new_tags->data(), new_npoints, 1, 0);

    // write PQ pivots.
    std::string pq_out = out_path_prefix + "_pq_compressed.bin";
    pipeann::save_bin<uint8_t>(pq_out, this->data.data(), new_npoints, n_chunks);

    if (in_path_prefix != out_path_prefix) {
      std::filesystem::copy(in_path_prefix + "_pq_pivots.bin", out_path_prefix + "_pq_pivots.bin",
                            std::filesystem::copy_options::overwrite_existing);
    }
  }

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::rebuild_graph_in_place(uint32_t nthreads) {
    if (nthreads == 0) {
      nthreads = this->max_nthreads;
    }

    void *ctx = reader->get_ctx();
    while (!bg_tasks.empty()) {
      sleep(1);
    }
    while (bg_tasks_inflight.load(std::memory_order_relaxed) != 0) {
      sleep(1);
    }
    // Ensure cache is clean before scan/write.
    this->flush_page_cache();

    const uint64_t n_sectors = (cur_loc + nnodes_per_sector - 1) / nnodes_per_sector;
    if (thread_pq_bufs.empty()) {
      LOG(ERROR) << "thread_pq_bufs is empty; cannot rebuild graph.";
      return;
    }
    LOG(INFO) << "Graph-only merge (in-place): cur_loc=" << cur_loc.load()
              << " cur_id=" << cur_id << " n_sectors=" << n_sectors
              << " nnodes_per_sector=" << nnodes_per_sector;

    char *buf = nullptr;
    pipeann::alloc_aligned((void **) &buf, SECTORS_PER_MERGE * SECTOR_LEN, SECTOR_LEN);

    for (uint64_t in_sector = 0; in_sector < n_sectors; in_sector += SECTORS_PER_MERGE) {
      uint64_t st_sector = in_sector;
      uint64_t ed_sector = std::min(in_sector + SECTORS_PER_MERGE, n_sectors);
      uint64_t loc_st = st_sector * nnodes_per_sector;
      uint64_t loc_ed = std::min(cur_loc.load(), ed_sector * nnodes_per_sector);
      uint64_t n_sectors_to_rw = ed_sector - st_sector;

      std::vector<IORequest> read_reqs;
      read_reqs.push_back(IORequest(loc_sector_no(loc_st) * SECTOR_LEN,
                                    n_sectors_to_rw * size_per_io, buf, 0, 0));
      reader->read(read_reqs, ctx, false);

#pragma omp parallel for num_threads(nthreads)
      for (uint64_t loc = loc_st; loc < loc_ed; ++loc) {
        uint64_t id = loc2id(loc);
        if (id == kInvalidID || id == kAllocatedID) {
          continue;
        }

        auto page_buf = buf + (loc / nnodes_per_sector - st_sector) * SECTOR_LEN;
        auto loc_buf = offset_to_loc(page_buf, loc);
        DiskNode<T> node(id, offset_to_node_coords(loc_buf), offset_to_node_nhood(loc_buf));

        std::unordered_set<uint32_t> nhood_set;
        nhood_set.reserve(node.nnbrs * 2 + 1);
        for (uint32_t i = 0; i < node.nnbrs; ++i) {
          uint32_t nbr = node.nbrs[i];
          if (nbr != id) {
            nhood_set.insert(nbr);
          }
        }
        std::vector<uint32_t> nhood(nhood_set.begin(), nhood_set.end());

        if (nhood.size() > this->range) {
          std::vector<float> dists(nhood.size(), 0.0f);
          std::vector<Neighbor> pool(nhood.size());
          auto &thread_pq_buf = thread_pq_bufs[omp_get_thread_num() % thread_pq_bufs.size()];
          compute_pq_dists(id, nhood.data(), dists.data(), (_u32) nhood.size(), thread_pq_buf);

          for (uint32_t k = 0; k < nhood.size(); k++) {
            pool[k].id = nhood[k];
            pool[k].distance = dists[k];
          }
          std::sort(pool.begin(), pool.end());
          if (pool.size() > this->maxc) {
            pool.resize(this->maxc);
          }
          nhood.clear();
          this->prune_neighbors_pq(pool, nhood, thread_pq_buf);
        }

        node.nnbrs = nhood.size();
        *(node.nbrs - 1) = node.nnbrs;
        if (!nhood.empty()) {
          memcpy(node.nbrs, nhood.data(), node.nnbrs * sizeof(uint32_t));
        }
      }

      std::vector<IORequest> write_reqs;
      write_reqs.push_back(IORequest(loc_sector_no(loc_st) * SECTOR_LEN,
                                     n_sectors_to_rw * size_per_io, buf, 0, 0));
      reader->write(write_reqs, ctx, false);

      LOG(INFO) << "Graph-only merge: processed " << ed_sector << "/" << n_sectors << " sectors.";
    }

    pipeann::aligned_free((void *) buf);
    LOG(INFO) << "Graph-only merge (in-place) finished.";
  }

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::merge_graph_only(const std::string &in_path_prefix, const std::string &out_path_prefix,
                                           uint32_t nthreads) {
    if (nthreads == 0) {
      nthreads = this->max_nthreads;
    }

    void *ctx = reader->get_ctx();
    while (!bg_tasks.empty()) {
      sleep(1);
    }
    while (bg_tasks_inflight.load(std::memory_order_relaxed) != 0) {
      sleep(1);
    }
    this->flush_page_cache();

    if (thread_pq_bufs.empty()) {
      LOG(ERROR) << "thread_pq_bufs is empty; cannot merge graph.";
      return;
    }

    std::string disk_index_out = out_path_prefix + "_disk.index";
    int fd = open(disk_index_out.c_str(), O_DIRECT | O_LARGEFILE | O_RDWR | O_CREAT, 0755);

    const uint64_t new_npoints = this->cur_id;
    const uint64_t file_size =
        SECTOR_LEN + ROUND_UP(new_npoints, nnodes_per_sector) / nnodes_per_sector * SECTOR_LEN;
    ::ftruncate(fd, file_size);

    const uint64_t n_sectors = (cur_loc + nnodes_per_sector - 1) / nnodes_per_sector;
    LOG(INFO) << "Graph-only merge (out-prefix): cur_loc=" << cur_loc.load() << " cur_id=" << cur_id
              << " n_sectors=" << n_sectors << " nnodes_per_sector=" << nnodes_per_sector;

    char *buf = nullptr;
    pipeann::alloc_aligned((void **) &buf, SECTORS_PER_MERGE * SECTOR_LEN, SECTOR_LEN);

    for (uint64_t in_sector = 0; in_sector < n_sectors; in_sector += SECTORS_PER_MERGE) {
      uint64_t st_sector = in_sector, ed_sector = std::min(in_sector + SECTORS_PER_MERGE, n_sectors);
      uint64_t loc_st = st_sector * nnodes_per_sector;
      uint64_t loc_ed = std::min(cur_loc.load(), ed_sector * nnodes_per_sector);
      uint64_t n_sectors_to_rw = ed_sector - st_sector;

      std::vector<IORequest> read_reqs;
      read_reqs.push_back(IORequest(loc_sector_no(loc_st) * SECTOR_LEN, n_sectors_to_rw * size_per_io, buf, 0, 0));
      reader->read(read_reqs, ctx, false);

#pragma omp parallel for num_threads(nthreads)
      for (uint64_t loc = loc_st; loc < loc_ed; ++loc) {
        uint64_t id = loc2id(loc);
        if (id == kInvalidID || id == kAllocatedID) {
          continue;
        }

        auto page_buf = buf + (loc / nnodes_per_sector - st_sector) * SECTOR_LEN;
        auto loc_buf = offset_to_loc(page_buf, loc);
        DiskNode<T> node(id, offset_to_node_coords(loc_buf), offset_to_node_nhood(loc_buf));

        std::unordered_set<uint32_t> nhood_set;
        nhood_set.reserve(node.nnbrs * 2 + 1);
        for (uint32_t i = 0; i < node.nnbrs; ++i) {
          uint32_t nbr = node.nbrs[i];
          if (nbr != id) {
            nhood_set.insert(nbr);
          }
        }
        std::vector<uint32_t> nhood(nhood_set.begin(), nhood_set.end());

        if (nhood.size() > this->range) {
          std::vector<float> dists(nhood.size(), 0.0f);
          std::vector<Neighbor> pool(nhood.size());
          auto &thread_pq_buf = thread_pq_bufs[omp_get_thread_num() % thread_pq_bufs.size()];
          compute_pq_dists(id, nhood.data(), dists.data(), (_u32) nhood.size(), thread_pq_buf);

          for (uint32_t k = 0; k < nhood.size(); k++) {
            pool[k].id = nhood[k];
            pool[k].distance = dists[k];
          }
          std::sort(pool.begin(), pool.end());
          if (pool.size() > this->maxc) {
            pool.resize(this->maxc);
          }
          nhood.clear();
          this->prune_neighbors_pq(pool, nhood, thread_pq_buf);
        }

        node.nnbrs = nhood.size();
        *(node.nbrs - 1) = node.nnbrs;
        if (!nhood.empty()) {
          memcpy(node.nbrs, nhood.data(), node.nnbrs * sizeof(uint32_t));
        }
      }

      std::vector<IORequest> write_reqs;
      write_reqs.push_back(
          IORequest(loc_sector_no(loc_st) * SECTOR_LEN, n_sectors_to_rw * size_per_io, buf, 0, 0));
      reader->write_fd(fd, write_reqs, ctx);

      LOG(INFO) << "Graph-only merge(out): processed " << ed_sector << "/" << n_sectors << " sectors.";
    }

    close(fd);
    pipeann::aligned_free((void *) buf);

    // write metadata header in-place (do NOT truncate, or we will wipe the graph)
    {
      uint32_t medoid = this->medoids[0];
      std::vector<uint64_t> output_metadata;
      output_metadata.push_back(new_npoints);
      output_metadata.push_back((uint64_t) this->data_dim);
      output_metadata.push_back(medoid);
      output_metadata.push_back(this->max_node_len);
      output_metadata.push_back(nnodes_per_sector);
      output_metadata.push_back(this->num_frozen_points);
      output_metadata.push_back(this->frozen_location);
      output_metadata.push_back(file_size);

      std::fstream io(disk_index_out, std::ios::binary | std::ios::in | std::ios::out);
      if (!io.is_open()) {
        LOG(ERROR) << "Failed to open disk index for header update: " << disk_index_out;
      } else {
        uint32_t npts_u32 = (uint32_t) output_metadata.size();
        uint32_t ndims_u32 = 1;
        io.seekp(0, std::ios::beg);
        io.write((char *) &npts_u32, sizeof(uint32_t));
        io.write((char *) &ndims_u32, sizeof(uint32_t));
        io.write((char *) output_metadata.data(),
                 (std::streamsize) (output_metadata.size() * sizeof(uint64_t)));
        io.flush();
      }
    }

    // copy PQ/tags without rebuilding in memory
    std::filesystem::copy(in_path_prefix + "_pq_compressed.bin", out_path_prefix + "_pq_compressed.bin",
                          std::filesystem::copy_options::overwrite_existing);
    std::filesystem::copy(in_path_prefix + "_pq_pivots.bin", out_path_prefix + "_pq_pivots.bin",
                          std::filesystem::copy_options::overwrite_existing);
    const std::string tag_in = in_path_prefix + "_disk.index.tags";
    if (std::filesystem::exists(tag_in)) {
      std::filesystem::copy(tag_in, out_path_prefix + "_disk.index.tags",
                            std::filesystem::copy_options::overwrite_existing);
    }

    LOG(INFO) << "Graph-only merge (out-prefix) finished.";
  }

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::reload(const char *index_prefix, uint32_t num_threads) {
    std::string iprefix = std::string(index_prefix);
    std::string pq_compressed_vectors = iprefix + "_pq_compressed.bin";
    std::string disk_index_file = iprefix + "_disk.index";
    this->_disk_index_file = disk_index_file;
    this->max_nthreads = num_threads;

    reader->close();
    reader->open(disk_index_file, true, false);

    LOG(INFO) << "Reloading, num_points " << this->num_points << " n_chunks: " << this->n_chunks;
    this->cur_id = this->cur_loc = this->num_points;
    if (this->num_points % nnodes_per_sector != 0) {
      this->cur_loc += nnodes_per_sector - (num_points % nnodes_per_sector);
    }

    while (!this->empty_pages.empty()) {
      this->empty_pages.pop();
    }
    merge_lock.unlock();
    return;
  }

  template class SSDIndex<float>;
  template class SSDIndex<_s8>;
  template class SSDIndex<_u8>;
}  // namespace pipeann
