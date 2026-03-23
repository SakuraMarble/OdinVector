#include "aligned_file_reader.h"
#include "libcuckoo/cuckoohash_map.hh"
#include "ssd_index.h"
#include <malloc.h>
#include <algorithm>
#include <filesystem>
#include <queue>

#include <omp.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <tuple>
#include "timer.h"
#include "tsl/robin_map.h"
#include "utils.h"
#include "v2/page_cache.h"

#include <unistd.h>
#include <sys/syscall.h>
#include "linux_aligned_file_reader.h"

namespace pipeann {
  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::do_beam_search(const T *query1, uint32_t mem_L, uint32_t l_search, const uint32_t beam_width,
                                         std::vector<Neighbor> &expanded_nodes_info,
                                         tsl::robin_map<uint32_t, T *> *coord_map, QueryStats *stats,
                                         tsl::robin_set<uint32_t> *exclude_nodes /* tags */, bool dyn_search_l,
                                         std::vector<uint64_t> *passthrough_page_ref) {
    uint32_t original_l_search = l_search;
    auto diskSearchBegin = std::chrono::high_resolution_clock::now();

    auto query_buf = pop_query_buf(query1);
    void *ctx = reader->get_ctx();

    const T *query = query_buf->aligned_query_T;

    // reset query
    query_buf->reset();

    // pointers to buffers for data
    T *data_buf = query_buf->coord_scratch;
    _u64 &data_buf_idx = query_buf->coord_idx;
    _mm_prefetch((char *) data_buf, _MM_HINT_T1);

    // sector scratch
    char *sector_scratch = query_buf->sector_scratch;
    _u64 &sector_scratch_idx = query_buf->sector_idx;

    // query <-> PQ chunk centers distances
    float *pq_dists = query_buf->aligned_pqtable_dist_scratch;
    pq_table.populate_chunk_distances(query, pq_dists);

    // query <-> neighbor list
    float *dist_scratch = query_buf->aligned_dist_scratch;
    _u8 *pq_coord_scratch = query_buf->aligned_pq_coord_scratch;

    // lambda to batch compute query<-> node distances in PQ space
    auto compute_dists = [this, pq_coord_scratch, pq_dists](const unsigned *ids, const _u64 n_ids, float *dists_out) {
      ::aggregate_coords(ids, n_ids, this->data.data(), this->n_chunks, pq_coord_scratch);
      ::pq_dist_lookup(pq_coord_scratch, n_ids, this->n_chunks, pq_dists, dists_out);
    };

    Timer query_timer, io_timer, cpu_timer;
    std::vector<Neighbor> retset;
    retset.resize(mem_L + 10 * l_search);
    tsl::robin_set<_u64> visited(4096);

    // re-naming `expanded_nodes_info` to not change rest of the code
    std::vector<Neighbor> &full_retset = expanded_nodes_info;
    full_retset.reserve(10 * l_search);
    _u32 best_medoid = medoids[0];

    unsigned cur_list_size = 0;
    auto compute_and_add_to_retset = [&](const unsigned *node_ids, const _u64 n_ids) {
      compute_dists(node_ids, n_ids, dist_scratch);
      for (_u64 i = 0; i < n_ids; ++i) {
        retset[cur_list_size].id = node_ids[i];
        retset[cur_list_size].distance = dist_scratch[i];
        retset[cur_list_size++].flag = true;
        visited.insert(node_ids[i]);
      }
    };

    if (mem_L) {
      std::vector<unsigned> mem_tags(mem_L);
      std::vector<float> mem_dists(mem_L);
      mem_index_->search_with_tags(query, mem_L, mem_L, mem_tags.data(), mem_dists.data());
      compute_and_add_to_retset(mem_tags.data(), std::min((unsigned) mem_L, (unsigned) l_search));
    } else {
      // Do not use optimized start point.
      compute_and_add_to_retset(&best_medoid, 1);
    }

    std::sort(retset.begin(), retset.begin() + cur_list_size);

    unsigned cmps = 0;
    unsigned hops = 0;
    unsigned num_ios = 0;
    unsigned k = 0;

    // cleared every iteration
    std::vector<unsigned> frontier;
    using fnhood_t = std::tuple<unsigned, unsigned, char *>;
    std::vector<fnhood_t> frontier_nhoods;
    std::vector<IORequest> frontier_read_reqs;
    std::vector<uint32_t> vec_rdlocks;

    std::vector<uint64_t> new_page_ref{};
    std::vector<uint64_t> &page_ref = passthrough_page_ref ? *passthrough_page_ref : new_page_ref;

    while (k < cur_list_size) {
      auto nk = cur_list_size;
      // clear iteration state
      frontier.clear();
      frontier_nhoods.clear();
      frontier_read_reqs.clear();
      vec_rdlocks.clear();
      sector_scratch_idx = 0;
      // find new beam
      // WAS: _u64 marker = k - 1;
      _u32 marker = k;
      _u32 num_seen = 0;
      while (marker < cur_list_size && frontier.size() < beam_width && num_seen < beam_width) {
        if (retset[marker].flag) {
          num_seen++;
          frontier.push_back(retset[marker].id);
          retset[marker].flag = false;
        }
        marker++;
      }

      // read nhoods of frontier ids
      std::vector<uint32_t> locked;
      if (!frontier.empty()) {
        if (stats != nullptr)
          stats->n_hops++;
        locked = this->lock_idx(idx_lock_table, kInvalidID, frontier, true);
        for (_u64 i = 0; i < frontier.size(); i++) {
          uint32_t id = frontier[i];
          uint32_t loc = this->id2loc(id);
          uint64_t offset = loc_sector_no(loc) * SECTOR_LEN;
          auto sector_buf = sector_scratch + sector_scratch_idx * size_per_io;
          fnhood_t fnhood = std::make_tuple(id, loc, sector_buf);
          sector_scratch_idx++;
          frontier_nhoods.push_back(fnhood);
          frontier_read_reqs.emplace_back(IORequest(offset, size_per_io, sector_buf, u_loc_offset(loc), max_node_len));
          if (stats != nullptr) {
            stats->n_4k++;
            stats->n_ios++;
          }
          num_ios++;
        }
        io_timer.reset();
#ifdef DIRECT_READ_CC
        reader->read(frontier_read_reqs, ctx);
#else
        reader->read_alloc(frontier_read_reqs, ctx, &page_ref);
#endif

        if (stats != nullptr) {
          stats->io_us += (double) io_timer.elapsed();
        }
        this->unlock_idx(idx_lock_table, locked);
      }

      for (auto &frontier_nhood : frontier_nhoods) {
        auto [id, loc, sector_buf] = frontier_nhood;
        char *node_disk_buf = offset_to_loc(sector_buf, loc);
        unsigned *node_buf = offset_to_node_nhood(node_disk_buf);
        _u64 nnbrs = (_u64) (*node_buf);
        T *node_fp_coords = offset_to_node_coords(node_disk_buf);
        assert(data_buf_idx < MAX_N_CMPS);

        T *node_fp_coords_copy = data_buf + (data_buf_idx * aligned_dim);
        data_buf_idx++;
        memcpy(node_fp_coords_copy, node_fp_coords, data_dim * sizeof(T));
        float cur_expanded_dist = dist_cmp->compare(query, node_fp_coords_copy, (unsigned) aligned_dim);

        if (coord_map != nullptr) {
          coord_map->insert(std::make_pair(id, node_fp_coords_copy));
        }
        full_retset.push_back(Neighbor(id, cur_expanded_dist, true));

        unsigned *node_nbrs = (node_buf + 1);

        // compute node_nbrs <-> query dist in PQ space
        cpu_timer.reset();
        compute_dists(node_nbrs, nnbrs, dist_scratch);
        if (stats != nullptr) {
          stats->n_cmps += (double) nnbrs;
          stats->cpu_us += (double) cpu_timer.elapsed();
        }

        cpu_timer.reset();
        // process prefetch-ed nhood
        for (_u64 m = 0; m < nnbrs; ++m) {
          unsigned id = node_nbrs[m];
          if (unlikely(id > this->cur_id)) {
            LOG(ERROR) << "ID is larger than current ID, " << id << " vs " << this->cur_id;
            crash();
          }
          if (visited.find(id) != visited.end()) {
            continue;
          } else {
            visited.insert(id);
            cmps++;
            float dist = dist_scratch[m];
            if (stats != nullptr) {
              stats->n_cmps++;
            }
            if (dist >= retset[cur_list_size - 1].distance && (cur_list_size == l_search))
              continue;
            Neighbor nn(id, dist, true);
            // variable search_L for deleted nodes.
            // Return position in sorted list where nn inserted.

            auto r = InsertIntoPool(retset.data(), cur_list_size, nn);

            if (cur_list_size < l_search) {
              ++cur_list_size;
              if (unlikely(cur_list_size >= retset.size())) {
                retset.resize(2 * cur_list_size);
              }
            }

            if (r < nk)
              nk = r;  // nk logs the best position in the retset that was
                       // updated due to neighbors of n.
          }
        }

        if (dyn_search_l) {
          // TODO(gh): contention still exists in id2tag(x)
          // O(n), but it is not slow as L is typically smaller than 300.
          // l_search monotonically increases to handle deleted nodes.
          _u32 tot = 0, cur = 0;
          for (cur = 0; cur < cur_list_size; ++cur) {
            uint32_t tag = id2tag(retset[cur].id);
            if (exclude_nodes->find(tag) == exclude_nodes->end()) {
              ++tot;
              if (tot == original_l_search) {
                break;
              }
            }
          }
          // cur is the stopped index (cur + 1 is the length it should be)
          l_search = std::max(original_l_search, cur + 1);
        }

        if (stats != nullptr) {
          stats->cpu_us += (double) cpu_timer.elapsed();
        }
      }

      // update best inserted position
      //

      if (nk <= k)
        k = nk;  // k is the best position in retset updated in this round.
      else
        ++k;

      hops++;
      if (stats != nullptr && stats->n_current_used != 0) {
        auto diskSearchEnd = std::chrono::high_resolution_clock::now();
        double elapsedSeconds =
            std::chrono::duration_cast<std::chrono::milliseconds>(diskSearchEnd - diskSearchBegin).count();
        if (elapsedSeconds >= stats->n_current_used)
          break;
      }
    }
    // re-sort by distance
    std::sort(full_retset.begin(), full_retset.end(),
              [](const Neighbor &left, const Neighbor &right) { return left < right; });

    if (passthrough_page_ref == nullptr) {
      reader->deref(&page_ref, ctx);
    }

    push_query_buf(query_buf);

    if (stats != nullptr) {
      stats->total_us = (double) query_timer.elapsed();
    }
  }

  template<typename T, typename TagT>
  size_t SSDIndex<T, TagT>::beam_search(const T *query, const _u64 k_search, const _u32 mem_L, const _u64 l_search,
                                        TagT *res_tags, float *distances, const _u64 beam_width, QueryStats *stats,
                                        tsl::robin_set<uint32_t> *deleted_nodes, bool dyn_search_l) {
    // iterate to fixed point
    std::shared_lock lk(merge_lock);
    std::vector<Neighbor> expanded_nodes_info;
    this->do_beam_search(query, mem_L, (_u32) l_search, (_u32) beam_width, expanded_nodes_info, nullptr, stats,
                         deleted_nodes, dyn_search_l);
    _u64 res_count = 0;
    for (uint32_t i = 0; i < l_search && res_count < k_search && i < expanded_nodes_info.size(); i++) {
      res_tags[res_count] = id2tag(expanded_nodes_info[i].id);
      distances[res_count] = expanded_nodes_info[i].distance;
      res_count++;
    }
    return res_count;
  }

  template class SSDIndex<float>;
  template class SSDIndex<_s8>;
  template class SSDIndex<_u8>;
}  // namespace pipeann

// namespace pipeann {

//   template<typename T, typename TagT>
//   size_t SSDIndex<T, TagT>::filter_beam_search(const T *query1, const _u64 k_search, const _u32 mem_L, const _u64 l_search,
//                                                TagT *res_tags, float *distances, const _u64 beam_width,
//                                                const std::vector<uint32_t> &candidate_ids, QueryStats *stats) {
//     if (candidate_ids.empty()) {
//       std::cout << "[DEBUG] candidate_ids is empty, returning early." << std::endl;
//       for (uint64_t i = 0; i < k_search; i++) {
//         res_tags[i] = 0;
//         distances[i] = std::numeric_limits<float>::max();
//       }
//       return 0;
//     }

//     std::shared_lock lk(merge_lock);
//     auto diskSearchBegin = std::chrono::high_resolution_clock::now();

//     auto query_buf = pop_query_buf(query1);
//     void *ctx = reader->get_ctx();
//     const T *query = query_buf->aligned_query_T;
//     query_buf->reset();

//     // Data buffers
//     T *data_buf = query_buf->coord_scratch;
//     _u64 &data_buf_idx = query_buf->coord_idx;
//     char *sector_scratch = query_buf->sector_scratch;
//     _u64 &sector_scratch_idx = query_buf->sector_idx;

//     // PQ Setup
//     float *pq_dists = query_buf->aligned_pqtable_dist_scratch;
//     pq_table.populate_chunk_distances(query, pq_dists);
//     float *dist_scratch = query_buf->aligned_dist_scratch;
//     _u8 *pq_coord_scratch = query_buf->aligned_pq_coord_scratch;

//     auto compute_dists = [this, pq_coord_scratch, pq_dists](const unsigned *ids, const _u64 n_ids, float *dists_out) {
//       ::aggregate_coords(ids, n_ids, this->data.data(), this->n_chunks, pq_coord_scratch);
//       ::pq_dist_lookup(pq_coord_scratch, n_ids, this->n_chunks, pq_dists, dists_out);
//     };

//     Timer query_timer, io_timer, cpu_timer;
    
//     // search_ctx: Navigation pool (bounded by l_search)
//     std::vector<Neighbor> retset;
//     retset.resize(mem_L + 10 * l_search);
//     unsigned cur_list_size = 0;

//     // result_heap: Max-heap for collecting valid filtered results
//     std::priority_queue<Neighbor> result_heap; 
    
//     tsl::robin_set<_u64> visited(4096);
//     _u32 best_medoid = medoids[0];

//     // Initialize entry points
//     auto compute_and_add_to_retset = [&](const unsigned *node_ids, const _u64 n_ids) {
//       compute_dists(node_ids, n_ids, dist_scratch);
//       for (_u64 i = 0; i < n_ids; ++i) {
//         // [修复]：不要在这里判断 candidate_ids 并加入 result_heap！
//         // 因为这些点接下来一定会被放到 frontier 里去读盘，读完盘再算精确距离。
//         retset[cur_list_size].id = node_ids[i];
//         retset[cur_list_size].distance = dist_scratch[i];
//         retset[cur_list_size++].flag = true;
//         visited.insert(node_ids[i]);
//       }
//     };

//     // PipeANN requires using mem_index or medoids since candidate_ids are external tags, not internal IDs
//     if (mem_L) {
//       std::vector<unsigned> mem_tags(mem_L);
//       std::vector<float> mem_dists(mem_L);
//       mem_index_->search_with_tags(query, mem_L, mem_L, mem_tags.data(), mem_dists.data());
//       compute_and_add_to_retset(mem_tags.data(), std::min((unsigned) mem_L, (unsigned) l_search));
//     } else {
//       compute_and_add_to_retset(&best_medoid, 1);
//     }

//     std::sort(retset.begin(), retset.begin() + cur_list_size);

//     unsigned cmps = 0;
//     unsigned hops = 0;
//     unsigned num_ios = 0;
//     unsigned k = 0;

//     std::vector<unsigned> frontier;
//     using fnhood_t = std::tuple<unsigned, unsigned, char *>;
//     std::vector<fnhood_t> frontier_nhoods;
//     std::vector<IORequest> frontier_read_reqs;
//     std::vector<uint64_t> page_ref{};

//     // Main Search Loop
//     while (k < cur_list_size) {
//       auto nk = cur_list_size;
//       frontier.clear();
//       frontier_nhoods.clear();
//       frontier_read_reqs.clear();
//       sector_scratch_idx = 0;

//       // Extract unexpanded nodes for the beam
//       _u32 marker = k;
//       _u32 num_seen = 0;
//       while (marker < cur_list_size && frontier.size() < beam_width && num_seen < beam_width) {
//         if (retset[marker].flag) {
//           num_seen++;
//           frontier.push_back(retset[marker].id);
//           retset[marker].flag = false;
//         }
//         marker++;
//       }

//       std::vector<uint32_t> locked;
//       if (!frontier.empty()) {
//         if (stats != nullptr) stats->n_hops++;
//         locked = this->lock_idx(idx_lock_table, kInvalidID, frontier, true);
        
//         for (_u64 i = 0; i < frontier.size(); i++) {
//           uint32_t id = frontier[i];
//           uint32_t loc = this->id2loc(id);
//           uint64_t offset = loc_sector_no(loc) * SECTOR_LEN;
//           auto sector_buf = sector_scratch + sector_scratch_idx * size_per_io;
          
//           frontier_nhoods.push_back(std::make_tuple(id, loc, sector_buf));
//           frontier_read_reqs.emplace_back(IORequest(offset, size_per_io, sector_buf, u_loc_offset(loc), max_node_len));
//           sector_scratch_idx++;
          
//           if (stats != nullptr) { stats->n_4k++; stats->n_ios++; }
//           num_ios++;
//         }
        
//         io_timer.reset();
// #ifdef DIRECT_READ_CC
//         reader->read(frontier_read_reqs, ctx);
// #else
//         reader->read_alloc(frontier_read_reqs, ctx, &page_ref);
// #endif
//         if (stats != nullptr) stats->io_us += (double) io_timer.elapsed();
//         this->unlock_idx(idx_lock_table, locked);
//       }

//       // [DEBUG] 4. 统计在这轮中，到底碰到了多少个有效的 candidate
//       _u32 candidates_found_this_hop = 0; 

//       // Process Neighborhoods
//       for (auto &frontier_nhood : frontier_nhoods) {
//         auto [id, loc, sector_buf] = frontier_nhood;
//         char *node_disk_buf = offset_to_loc(sector_buf, loc);

//         // ==========================================
//         // 【核心修复】：恢复节点自身的精确距离计算
//         // ==========================================
//         T *node_fp_coords = offset_to_node_coords(node_disk_buf);
//         T *node_fp_coords_copy = data_buf + (data_buf_idx * aligned_dim);
//         memcpy(node_fp_coords_copy, node_fp_coords, data_dim * sizeof(T));
//         data_buf_idx++; // 必须累加，保证 AVX 内存对齐不会互相覆盖
        
//         float exact_dist = dist_cmp->compare(query, node_fp_coords_copy, (unsigned) aligned_dim);

//         // 检查过滤条件，并使用【精确距离】更新最终的结果堆
//         uint32_t tag = id2tag(id);
//         if (std::binary_search(candidate_ids.begin(), candidate_ids.end(), tag)) {
//           if (result_heap.size() < k_search) {
//             result_heap.push(Neighbor(id, exact_dist, true));
//           } else if (exact_dist < result_heap.top().distance) {
//             result_heap.pop();
//             result_heap.push(Neighbor(id, exact_dist, true));
//           }
//         }
//         // ==========================================

//         // 接下来处理它的邻居，仅用于图的路由寻路 (PQ距离)
//         unsigned *node_buf = offset_to_node_nhood(node_disk_buf);
//         _u64 nnbrs = (_u64) (*node_buf);
//         unsigned *node_nbrs = (node_buf + 1);

//         cpu_timer.reset();
//         compute_dists(node_nbrs, nnbrs, dist_scratch);
//         if (stats != nullptr) {
//           stats->n_cmps += (double) nnbrs;
//           stats->cpu_us += (double) cpu_timer.elapsed();
//         }

//         cpu_timer.reset();
//         for (_u64 m = 0; m < nnbrs; ++m) {
//           unsigned nbr_id = node_nbrs[m];

//           if (unlikely(nbr_id > this->cur_id)) continue;
//           if (visited.find(nbr_id) != visited.end()) continue;
//           visited.insert(nbr_id);

//           float pq_dist = dist_scratch[m];
//           if (stats != nullptr) stats->n_cmps++;

//           // Branch B: Maintain search navigation retset (仅用 PQ 距离判断是否加入路由池)
//           if (cur_list_size < l_search || pq_dist < retset[cur_list_size - 1].distance) {
//             Neighbor nn(nbr_id, pq_dist, true);
//             auto r = InsertIntoPool(retset.data(), cur_list_size, nn);
//             if (cur_list_size < l_search) {
//               ++cur_list_size;
//               if (unlikely(cur_list_size >= retset.size())) {
//                 retset.resize(2 * cur_list_size);
//               }
//             }
//             if (r < nk) nk = r;
//           }
//         }
//         if (stats != nullptr) stats->cpu_us += (double) cpu_timer.elapsed();
//       }
      
//       if (nk <= k) k = nk; 
//       else ++k;

//       hops++;
//     }

//     reader->deref(&page_ref, ctx);
//     push_query_buf(query_buf);

//     // --- Extract and Format Results ---
//     std::vector<Neighbor> final_results;
//     final_results.reserve(result_heap.size());
//     while (!result_heap.empty()) {
//       final_results.push_back(result_heap.top());
//       result_heap.pop();
//     }

//     // Sort ascending (priority queue pops largest first)
//     std::sort(final_results.begin(), final_results.end());

//     size_t result_count = std::min((size_t)k_search, final_results.size());
//     for (size_t i = 0; i < result_count; i++) {
//       res_tags[i] = id2tag(final_results[i].id);
//       distances[i] = final_results[i].distance;
//     }

//     // Pad remaining slots
//     for (size_t i = result_count; i < k_search; i++) {
//       res_tags[i] = 0;
//       distances[i] = std::numeric_limits<float>::max();
//     }

//     if (stats != nullptr) stats->total_us = (double) query_timer.elapsed();

//     return result_count;
//   }
// } // namespace pipeann



namespace pipeann {

  // [修改点 1]: 在函数签名中新增了 `uint32_t k_search` 参数
  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::do_filter_beam_search(const T *query1, uint32_t k_search, uint32_t mem_L, uint32_t l_search, const uint32_t beam_width,
                                         std::vector<Neighbor> &expanded_nodes_info, tsl::robin_set<uint32_t> *candidate_set,
                                         tsl::robin_map<uint32_t, T *> *coord_map, QueryStats *stats,
                                         tsl::robin_set<uint32_t> *exclude_nodes /* tags */, bool dyn_search_l,
                                         std::vector<uint64_t> *passthrough_page_ref) {
    uint32_t original_l_search = l_search;
    auto diskSearchBegin = std::chrono::high_resolution_clock::now();

    auto query_buf = pop_query_buf(query1);
    void *ctx = reader->get_ctx();

    const T *query = query_buf->aligned_query_T;
    query_buf->reset();

    T *data_buf = query_buf->coord_scratch;
    _u64 &data_buf_idx = query_buf->coord_idx;
    _mm_prefetch((char *) data_buf, _MM_HINT_T1);

    char *sector_scratch = query_buf->sector_scratch;
    _u64 &sector_scratch_idx = query_buf->sector_idx;

    float *pq_dists = query_buf->aligned_pqtable_dist_scratch;
    pq_table.populate_chunk_distances(query, pq_dists);
    // --- [上帝视角 Debug：验证真值的 PQ 误差] ---
    // 假设我们要查的真值 ID 是 740504
    uint32_t gt_id = 740504; 
    
    // 1. 获取真值的压缩向量计算 PQ 距离
    _u8 gt_pq_coords[n_chunks];
    ::aggregate_coords(&gt_id, 1, this->data.data(), this->n_chunks, gt_pq_coords);
    float gt_pq_dist = 0;
    ::pq_dist_lookup(gt_pq_coords, 1, this->n_chunks, pq_dists, &gt_pq_dist);

    // 2. 发起一次同步 IO 读取真值的精确向量 (临时读取，仅供 Debug)
    uint32_t gt_loc = this->id2loc(gt_id);
    uint64_t gt_offset = loc_sector_no(gt_loc) * SECTOR_LEN;
    char debug_sector[SECTOR_LEN];
    std::vector<IORequest> debug_req = {IORequest(gt_offset, SECTOR_LEN, debug_sector, u_loc_offset(gt_loc), max_node_len)};
    reader->read(debug_req, ctx);
    
    char* gt_node_disk_buf = offset_to_loc(debug_sector, gt_loc);
    T* gt_exact_coords = offset_to_node_coords(gt_node_disk_buf);
    float gt_exact_dist = dist_cmp->compare(query, gt_exact_coords, (unsigned)aligned_dim);

    std::cout << "[GOD MODE DEBUG] GT ID: " << gt_id << std::endl;
    std::cout << "  -> Exact Distance : " << gt_exact_dist << std::endl;
    std::cout << "  -> PQ Distance    : " << gt_pq_dist << std::endl;
    // ---------------------------------------------
    float *dist_scratch = query_buf->aligned_dist_scratch;
    _u8 *pq_coord_scratch = query_buf->aligned_pq_coord_scratch;

    auto compute_dists = [this, pq_coord_scratch, pq_dists](const unsigned *ids, const _u64 n_ids, float *dists_out) {
      ::aggregate_coords(ids, n_ids, this->data.data(), this->n_chunks, pq_coord_scratch);
      ::pq_dist_lookup(pq_coord_scratch, n_ids, this->n_chunks, pq_dists, dists_out);
    };

    Timer query_timer, io_timer, cpu_timer;
    std::vector<Neighbor> retset;
    retset.resize(mem_L + 10 * l_search);
    tsl::robin_set<_u64> visited(4096);
    _u32 best_medoid = medoids[0];
    
    std::vector<Neighbor> &full_retset = expanded_nodes_info;
    full_retset.reserve(10 * l_search);

    // [修改点 2]: 引入双堆结构与 Exact 集合
    std::priority_queue<Neighbor> exact_result_heap; // 用于存放计算过**精确距离**且符合条件的节点
    std::priority_queue<Neighbor> pq_candidate_heap; // 用于存放内层循环用**PQ距离**海选的潜力节点
    tsl::robin_set<uint32_t> exact_computed_set(4096); // 记录哪些节点已经计算过精确距离，防止重复 IO

    // 辅助 Lambda：维护 Max-Heap 大小为 max_size
    auto push_to_heap = [](std::priority_queue<Neighbor>& heap, const Neighbor& n, size_t max_size) {
        if (heap.size() < max_size) {
            heap.push(n);
        } else if (n.distance < heap.top().distance) {
            heap.pop();
            heap.push(n);
        }
    };

    // mbj: 容错系数，PQ候选集收集更多以消除误差 IMPORTANT! 过小会导致过滤条件过于严格，过大则增加不必要的 IO 和计算开销
    uint32_t refine_capacity = std::max((uint32_t)3000, k_search * 1000); 

    unsigned cur_list_size = 0;
    auto compute_and_add_to_retset = [&](const unsigned *node_ids, const _u64 n_ids) {
      compute_dists(node_ids, n_ids, dist_scratch);
      for (_u64 i = 0; i < n_ids; ++i) {
        retset[cur_list_size].id = node_ids[i];
        retset[cur_list_size].distance = dist_scratch[i];
        retset[cur_list_size++].flag = true;
        visited.insert(node_ids[i]);

        // [修改点 3]: 入口点虽然只有 PQ 距离，但如果符合条件，也要丢进海选池
        uint32_t tag = id2tag(node_ids[i]);
        if (candidate_set->find(tag) != candidate_set->end()) {
            push_to_heap(pq_candidate_heap, Neighbor(node_ids[i], dist_scratch[i], true), refine_capacity);
        }
      }
    };

    if (mem_L) {
      std::vector<unsigned> mem_tags(mem_L);
      std::vector<float> mem_dists(mem_L);
      mem_index_->search_with_tags(query, mem_L, mem_L, mem_tags.data(), mem_dists.data());
      compute_and_add_to_retset(mem_tags.data(), std::min((unsigned) mem_L, (unsigned) l_search));
    } else {
      // std::vector<uint32_t> entry_tags;
      // unsigned num_entry = std::min((unsigned) candidate_set->size(), (unsigned) k_search); // mbj: 入口点也受 k_search 限制
      // auto it = candidate_set->begin();
      // for(unsigned i = 0; i < num_entry && it != candidate_set->end(); ++i, ++it) {
      //   entry_tags.push_back(*it);
      // }
      // compute_and_add_to_retset(entry_tags.data(), num_entry);
      // Do not use optimized start point.
      compute_and_add_to_retset(&best_medoid, 1);
    }

    std::sort(retset.begin(), retset.begin() + cur_list_size);

    unsigned cmps = 0, hops = 0, num_ios = 0, k = 0;
    std::vector<unsigned> frontier;
    using fnhood_t = std::tuple<unsigned, unsigned, char *>;
    std::vector<fnhood_t> frontier_nhoods;
    std::vector<IORequest> frontier_read_reqs;
    std::vector<uint32_t> vec_rdlocks;

    std::vector<uint64_t> new_page_ref{};
    std::vector<uint64_t> &page_ref = passthrough_page_ref ? *passthrough_page_ref : new_page_ref;

    while (k < cur_list_size) {
      auto nk = cur_list_size;
      frontier.clear();
      frontier_nhoods.clear();
      frontier_read_reqs.clear();
      vec_rdlocks.clear();
      sector_scratch_idx = 0;
      
      _u32 marker = k;
      _u32 num_seen = 0;
      while (marker < cur_list_size && frontier.size() < beam_width && num_seen < beam_width) {
        if (retset[marker].flag) {
          num_seen++;
          frontier.push_back(retset[marker].id);
          retset[marker].flag = false;
        }
        marker++;
      }

      std::vector<uint32_t> locked;
      if (!frontier.empty()) {
        if (stats != nullptr) stats->n_hops++;
        locked = this->lock_idx(idx_lock_table, kInvalidID, frontier, true);
        for (_u64 i = 0; i < frontier.size(); i++) {
          uint32_t id = frontier[i];
          uint32_t loc = this->id2loc(id);
          uint64_t offset = loc_sector_no(loc) * SECTOR_LEN;
          auto sector_buf = sector_scratch + sector_scratch_idx * size_per_io;
          sector_scratch_idx++;
          frontier_nhoods.push_back(std::make_tuple(id, loc, sector_buf));
          frontier_read_reqs.emplace_back(IORequest(offset, size_per_io, sector_buf, u_loc_offset(loc), max_node_len));
          if (stats != nullptr) { stats->n_4k++; stats->n_ios++; }
          num_ios++;
        }
        io_timer.reset();
#ifdef DIRECT_READ_CC
        reader->read(frontier_read_reqs, ctx);
#else
        reader->read_alloc(frontier_read_reqs, ctx, &page_ref);
#endif
        if (stats != nullptr) stats->io_us += (double) io_timer.elapsed();
        this->unlock_idx(idx_lock_table, locked);
      }

      for (auto &frontier_nhood : frontier_nhoods) {
        auto [id, loc, sector_buf] = frontier_nhood;
        char *node_disk_buf = offset_to_loc(sector_buf, loc);
        unsigned *node_buf = offset_to_node_nhood(node_disk_buf);
        _u64 nnbrs = (_u64) (*node_buf);
        T *node_fp_coords = offset_to_node_coords(node_disk_buf);
        assert(data_buf_idx < MAX_N_CMPS);

        T *node_fp_coords_copy = data_buf + (data_buf_idx * aligned_dim);
        data_buf_idx++;
        memcpy(node_fp_coords_copy, node_fp_coords, data_dim * sizeof(T));
        float cur_expanded_dist = dist_cmp->compare(query, node_fp_coords_copy, (unsigned) aligned_dim);

        if (coord_map != nullptr) {
          coord_map->insert(std::make_pair(id, node_fp_coords_copy));
        }

        // [修改点 4]: 在此处收集精确结果！并且打上 "已计算" 的标记
        exact_computed_set.insert(id);
        uint32_t tag = id2tag(id);
        if (candidate_set->find(tag) != candidate_set->end()) {
            push_to_heap(exact_result_heap, Neighbor(id, cur_expanded_dist, true), k_search);
        }

        unsigned *node_nbrs = (node_buf + 1);

        cpu_timer.reset();
        compute_dists(node_nbrs, nnbrs, dist_scratch);
        if (stats != nullptr) {
          stats->n_cmps += (double) nnbrs;
          stats->cpu_us += (double) cpu_timer.elapsed();
        }

        cpu_timer.reset();
        for (_u64 m = 0; m < nnbrs; ++m) {
          unsigned nbr_id = node_nbrs[m];
          if (unlikely(nbr_id > this->cur_id)) {
            LOG(ERROR) << "ID is larger than current ID, " << nbr_id << " vs " << this->cur_id;
            crash();
          }
          if (visited.find(nbr_id) != visited.end()) continue;

          visited.insert(nbr_id);
          cmps++;
          float dist = dist_scratch[m];
          if (stats != nullptr) stats->n_cmps++;

          // [修改点 5]: 这里只使用 PQ 距离收集候选集，不再干扰精确对比池
          uint32_t nbr_tag = id2tag(nbr_id);
          if (candidate_set->find(nbr_tag) != candidate_set->end()) {
              push_to_heap(pq_candidate_heap, Neighbor(nbr_id, dist, true), refine_capacity);
          }

          if (dist >= retset[cur_list_size - 1].distance && (cur_list_size == l_search))
            continue;
          
          Neighbor nn(nbr_id, dist, true);
          auto r = InsertIntoPool(retset.data(), cur_list_size, nn);

          if (cur_list_size < l_search) {
            ++cur_list_size;
            if (unlikely(cur_list_size >= retset.size())) {
              retset.resize(2 * cur_list_size);
            }
          }
          if (r < nk) nk = r;
        }

        if (dyn_search_l) {
          _u32 tot = 0, cur = 0;
          for (cur = 0; cur < cur_list_size; ++cur) {
            uint32_t tag_dyn = id2tag(retset[cur].id);
            if (exclude_nodes->find(tag_dyn) == exclude_nodes->end() && candidate_set->find(tag_dyn) != candidate_set->end()) { // mbj: 动态调整时也要考虑过滤条件，尽可能增大探测范围
              ++tot;
              if (tot == original_l_search) break;
            }
          }
          l_search = std::max(original_l_search, cur + 1);
        }

        if (stats != nullptr) stats->cpu_us += (double) cpu_timer.elapsed();
      }

      if (nk <= k) k = nk;
      else ++k;

      hops++;
      if (stats != nullptr && stats->n_current_used != 0) {
        auto diskSearchEnd = std::chrono::high_resolution_clock::now();
        double elapsedSeconds = std::chrono::duration_cast<std::chrono::milliseconds>(diskSearchEnd - diskSearchBegin).count();
        if (elapsedSeconds >= stats->n_current_used) break;
      }
    }

    // =========================================================================
    // [修改点 6]: 新增 Refinement (终面) 阶段
    // =========================================================================
    std::vector<uint32_t> refine_ids;
    while (!pq_candidate_heap.empty()) {
        uint32_t cid = pq_candidate_heap.top().id;
        pq_candidate_heap.pop();
        // 只有没被免费 IO 过精确坐标的节点，才进入这一步
        if (exact_computed_set.find(cid) == exact_computed_set.end()) {
            refine_ids.push_back(cid);
            exact_computed_set.insert(cid); // 防止在 Batch 中重复
        }
    }

    // 分批次读取，防止超出 sector_scratch 容量（每次读取量设定为 beam_width）
    size_t processed = 0;
    while (processed < refine_ids.size()) {
        size_t batch_size = std::min((size_t)beam_width, refine_ids.size() - processed);
        std::vector<uint32_t> batch_ids(refine_ids.begin() + processed, refine_ids.begin() + processed + batch_size);
        processed += batch_size;

        frontier_nhoods.clear();
        frontier_read_reqs.clear();
        sector_scratch_idx = 0; // 重用之前申请好的 buffer 区间

        auto locked = this->lock_idx(idx_lock_table, kInvalidID, batch_ids, true);
        for (size_t i = 0; i < batch_ids.size(); i++) {
            uint32_t id = batch_ids[i];
            uint32_t loc = this->id2loc(id);
            uint64_t offset = loc_sector_no(loc) * SECTOR_LEN;
            auto sector_buf = sector_scratch + sector_scratch_idx * size_per_io;
            sector_scratch_idx++;
            frontier_nhoods.push_back(std::make_tuple(id, loc, sector_buf));
            frontier_read_reqs.emplace_back(IORequest(offset, size_per_io, sector_buf, u_loc_offset(loc), max_node_len));
            if (stats != nullptr) { stats->n_4k++; stats->n_ios++; }
        }

        io_timer.reset();
#ifdef DIRECT_READ_CC
        reader->read(frontier_read_reqs, ctx);
#else
        reader->read_alloc(frontier_read_reqs, ctx, &page_ref);
#endif
        if (stats != nullptr) stats->io_us += (double) io_timer.elapsed();
        this->unlock_idx(idx_lock_table, locked);

        cpu_timer.reset();
        for (auto &frontier_nhood : frontier_nhoods) {
            auto [id, loc, sector_buf] = frontier_nhood;
            char *node_disk_buf = offset_to_loc(sector_buf, loc);
            T *node_fp_coords = offset_to_node_coords(node_disk_buf);

            // 如果空间够，保存 copy 用于 coord_map，否则直接比对原位置
            float exact_dist = 0.0f;
            if (data_buf_idx < MAX_N_CMPS) {
                T *node_fp_coords_copy = data_buf + (data_buf_idx * aligned_dim);
                data_buf_idx++;
                memcpy(node_fp_coords_copy, node_fp_coords, data_dim * sizeof(T));
                exact_dist = dist_cmp->compare(query, node_fp_coords_copy, (unsigned) aligned_dim);
                if (coord_map != nullptr) coord_map->insert(std::make_pair(id, node_fp_coords_copy));
            } else {
                exact_dist = dist_cmp->compare(query, node_fp_coords, (unsigned) aligned_dim);
            }
            // 使用精确距离纳入最终结果池！
            push_to_heap(exact_result_heap, Neighbor(id, exact_dist, true), k_search);
        }
        if (stats != nullptr) stats->cpu_us += (double) cpu_timer.elapsed();
    }
    // =========================================================================

    // 从精确结果集中弹出最终数据
    while (!exact_result_heap.empty()) {
        full_retset.push_back(exact_result_heap.top());
        exact_result_heap.pop();
    }
    // 优先队列是大顶堆，弹出的最大值在最前面，所以需要倒序一下，让距离最小的在开头
    std::reverse(full_retset.begin(), full_retset.end());

    if (passthrough_page_ref == nullptr) {
      reader->deref(&page_ref, ctx);
    }

    push_query_buf(query_buf);

    if (stats != nullptr) {
      stats->total_us = (double) query_timer.elapsed();
    }
  }

  template<typename T, typename TagT>
  size_t SSDIndex<T, TagT>::filter_beam_search(const T *query, const _u64 k_search, const _u32 mem_L, const _u64 l_search,
                                        TagT *res_tags, float *distances, const _u64 beam_width, const std::vector<uint32_t>& candidate_ids, QueryStats *stats,
                                        tsl::robin_set<uint32_t> *deleted_nodes, bool dyn_search_l) {
    std::shared_lock lk(merge_lock);
    std::vector<Neighbor> expanded_nodes_info;
    tsl::robin_set<uint32_t> candidate_set(candidate_ids.begin(), candidate_ids.end()); 

    // [修改点 7]: 传递了 `k_search` 给 `do_filter_beam_search`
    this->do_filter_beam_search(query, (_u32)k_search, mem_L, (_u32) l_search, (_u32) beam_width, expanded_nodes_info, &candidate_set, nullptr, stats,
                         deleted_nodes, dyn_search_l);
                         
    _u64 res_count = 0;
    for (uint32_t i = 0; i < k_search && i < expanded_nodes_info.size(); i++) {
      res_tags[res_count] = id2tag(expanded_nodes_info[i].id);
      distances[res_count] = expanded_nodes_info[i].distance;
      res_count++;
    }
    return res_count;
  }
}  // namespace pipeann