#include "aligned_file_reader.h"
#include "libcuckoo/cuckoohash_map.hh"
#include "ssd_index.h"
#include <malloc.h>
#include <algorithm>
#include <filesystem>
#include <fstream>

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
#include <cstdlib>
#include <sys/syscall.h>
#include "linux_aligned_file_reader.h"

namespace pipeann {
  static double get_rss_mb() {
    std::ifstream f("/proc/self/statm");
    long t = 0, r = 0;
    if (f.is_open()) {
      f >> t >> r;
    }
    long page_kb = sysconf(_SC_PAGE_SIZE) / 1024;
    return (double) r * page_kb / 1024.0;
  }

  static bool should_trace_insert(uint64_t trace_id) {
    const char *env = std::getenv("PIPEANN_TRACE_INSERT");
    if (!env) {
      return false;
    }
    int limit = std::atoi(env);
    if (limit <= 0) {
      return false;
    }
    return trace_id < (uint64_t) limit;
  }

  static std::atomic<uint64_t> trace_insert_counter{0};
  //point是待插入向量，tag是关联标签，deletion_set是删除标记集合（用于逻辑删除）
  template<typename T, typename TagT>
  int SSDIndex<T, TagT>::insert_in_place(const T *point, const TagT &tag, tsl::robin_set<uint32_t> *deletion_set) {
    QueryBuffer<T> *read_data = this->pop_query_buf(nullptr);
    void *ctx = reader->get_ctx();

    uint32_t target_id = cur_id++;// 待插入向量的ID 全局唯一
    const uint64_t trace_id = trace_insert_counter.fetch_add(1, std::memory_order_relaxed);
    const bool trace = should_trace_insert(trace_id);
    pipeann::Timer trace_timer;
    if (trace) {
      LOG(INFO) << "[TraceInsert] id=" << target_id << " start rss=" << get_rss_mb() << " MB";
    }
    // write PQ.

    std::vector<uint8_t> pq_coords = deflate_vector(point);// 将原始向量压缩为PQ编码的uint8_t数组
    uint64_t pq_offset = target_id * n_chunks;// 计算PQ坐标在 data 数组中的存储偏移量
    {
      static std::mutex pq_mu;// pq_mu 是静态互斥锁，保证多线程环境下的数据一致性
      std::lock_guard<std::mutex> lock(pq_mu);
      if (this->data.size() < pq_offset + n_chunks) { // 判断 data 容器是否足够存储新数据
        while (this->data.size() < pq_offset + n_chunks) {
          this->data.resize(1.5 * this->data.size());// 使用1.5倍扩容避免频繁重新分配内存
        }
      }
      // 将压缩后的向量数据写入到data容器位置
      /* 目的地址：this->data.data() + pq_offset 指向目标存储位置
         源地址：pq_coords.data() 指向PQ压缩后的数据
         复制字节数：n_chunks 个字节（即一个向量的PQ编码长度）*/
      memcpy(this->data.data() + pq_offset, pq_coords.data(), n_chunks);
    }
    if (trace) {
      LOG(INFO) << "[TraceInsert] id=" << target_id << " step=pq_write"
                << " ms=" << (trace_timer.elapsed() / 1.0e3f)
                << " rss=" << get_rss_mb() << " MB"
                << " data_size=" << this->data.size();
    }

    std::vector<Neighbor> exp_node_info;// 存储束搜索过程中发现的节点及其相关信息
    tsl::robin_map<uint32_t, T *> coord_map;// 以节点ID（uint32_t）为键，节点坐标指针（T*）为值
    coord_map.reserve(2 * this->l_index);// 预分配 2 * l_index 个槽位

    std::vector<uint64_t> page_ref{};
    this->do_beam_search(point, 0, l_index, beam_width, exp_node_info, &coord_map, nullptr, deletion_set, false,
                         &page_ref);
    std::vector<uint32_t> new_nhood;// 存储经过剪枝后的新邻居节点ID
    prune_neighbors(coord_map, exp_node_info, new_nhood);// 对搜索结果进行邻居剪枝
    if (trace) {
      LOG(INFO) << "[TraceInsert] id=" << target_id << " step=beam_prune"
                << " ms=" << (trace_timer.elapsed() / 1.0e3f)
                << " rss=" << get_rss_mb() << " MB"
                << " nhood=" << new_nhood.size();
    }
    // locs[new_nhood.size()] is the target, locs[0:new_nhood.size() - 1] are the neighbors.
    // lock the pages to write
    std::set<uint64_t> pages_need_to_read;// 用于页面锁定和读取操作

#ifdef IN_PLACE_RECORD_UPDATE
    // 直接更新模式：在原位置修改
    std::vector<uint64_t> locs;// 存储新节点的ID
    for (auto &nbr : new_nhood) { // 遍历 new_nhood 中的每个邻居节点
      locs.emplace_back(id2loc(nbr));// 使用 id2loc(nbr) 将节点ID转换为位置ID
      pages_need_to_read.insert(node_sector_no(nbr));//将节点所在扇区号加入 pages_need_to_read 集合
    }
    locs.push_back(target_id);// 将目标节点ID加入位置向量
    pages_need_to_read.insert(loc_sector_no(target_id));//将目标节点所在扇区加入读取集合
    id2loc_.insert_or_assign(target_id, target_id);// 将目标ID到位置的映射设置为一对一关系

    // update loc2id, target_id <-> target_id. 更新位置到ID的映射
    cur_loc++;  // for target ID, atomic update. 增加位置计数器 cur_loc
    set_loc2id(target_id, target_id);
#else
    // 间接更新模式：为新节点及其邻分配新位置
    auto locs = this->alloc_loc(new_nhood.size() + 1, page_ref, pages_need_to_read);
#endif
    // 获取需要读取/修改/写入(RMW)的所有页面
    std::set<uint64_t> pages_to_rmw_set;
    for (auto &loc : locs) {
      pages_to_rmw_set.insert(loc_sector_no(loc));// 对每个位置计算其扇区号并添加到集合
    }
    std::vector<IORequest> pages_to_rmw;
    // ordered because of std::set 为每个页面创建IO请求对象
    for (auto &page_no : pages_to_rmw_set) {
      pages_to_rmw.push_back(IORequest(page_no * SECTOR_LEN, size_per_io, nullptr, 0, 0));
    }
    // lock the target and the neighbor ids (ensure that sector_no does not change).
    // 锁定目标和邻居ID，确保扇区号不会改变
    auto pages_locked = v2::lockReqs(this->page_lock_table, pages_to_rmw);
    lock_vec(vec_lock_table, target_id, new_nhood);

    // 重新读取候选页面 re-read the candidate pages (mostly in the cache).
    std::unordered_map<uint32_t, char *> page_buf_map;

    auto &update_buf = read_data->update_buf;
    // 存储读取请求,存储4KB写入请求（待合并）,存储合并后的写入请求
    std::vector<IORequest> reads, writes_4k, writes;
    assert(new_nhood.size() < MAX_N_EDGES);
    // 为每个邻居节点创建读取请求
    for (uint32_t i = 0; i < new_nhood.size(); ++i) {
      reads.push_back(
          // 构建扇区级别的IO请求
          IORequest(node_sector_no(new_nhood[i]) * SECTOR_LEN, size_per_io, update_buf + i * size_per_io, 0, 0));
      // 扇区号映射到对应的缓冲区位置    
      page_buf_map[node_sector_no(new_nhood[i])] = update_buf + i * size_per_io;
    }

    // 为需要RMW的页面创建写入请求
    for (uint32_t i = new_nhood.size(); i < new_nhood.size() + pages_to_rmw.size(); ++i) {
      auto off = pages_to_rmw[i - new_nhood.size()].offset;// 从 pages_to_rmw 获取偏移量
      writes_4k.push_back(IORequest(off, size_per_io, update_buf + i * size_per_io, 0, 0));
      // LOG(INFO) << off / SECTOR_LEN;
      uint64_t page = off / SECTOR_LEN;
      if (pages_need_to_read.find(page) != pages_need_to_read.end()) {
        reads.push_back(IORequest(off, size_per_io, update_buf + i * size_per_io, 0, 0));
      }
      // 将页面号映射到缓冲区位置
      page_buf_map[off / SECTOR_LEN] = update_buf + i * size_per_io;
    }

    // generate continuous writes from 4k writes.
    // dummy one.
    // 添加哨兵：在 writes_4k 末尾添加一个极大偏移的虚拟请求作为结束标志
    // 遍历4K写入请求，将连续的页面合并成大块写入
    writes_4k.push_back(IORequest(std::numeric_limits<uint64_t>::max(), 0, nullptr, 0, 0));
    uint64_t start_idx = 0;
    uint64_t cur_off = writes_4k[0].offset;
    // 遍历4K写入请求，将连续的页面合并成大块写入
    for (uint32_t i = 1; i < writes_4k.size(); ++i) {
      // 检查当前请求的偏移量是否紧接在上一个请求之后
      // 不相等：表示连续段在此中断，需要合并前面的连续请求
      if (writes_4k[i].offset != cur_off + size_per_io) {
        // 将 start_idx 到 i-1 的所有4KB请求合并成一个大请求
        writes.push_back(
            IORequest(writes_4k[start_idx].offset, size_per_io * (i - start_idx), writes_4k[start_idx].buf, 0, 0));
        // start_idx 指向新的连续段起点    
        start_idx = i;
      }
      // cur_off 更新为当前请求的偏移量
      cur_off = writes_4k[i].offset;
    }
    writes_4k.pop_back();// 移除哨兵：删除之前添加的虚拟请求

#ifdef DIRECT_READ_CC
    reader->read(reads, ctx);// 使用读取器直接执行读取请求
#else
    reader->read_alloc(reads, ctx, &page_ref);// 执行带页面引用分配的读取
#endif
    if (trace) {
      LOG(INFO) << "[TraceInsert] id=" << target_id << " step=read_pages"
                << " ms=" << (trace_timer.elapsed() / 1.0e3f)
                << " rss=" << get_rss_mb() << " MB"
                << " reads=" << reads.size()
                << " writes4k=" << writes_4k.size()
                << " writes=" << writes.size();
    }

    // update the target node.
    // 获取目标节点所在扇区号
    auto sector = loc_sector_no(locs[new_nhood.size()]);
    // 从页面缓冲区映射中找到目标节点的缓冲区位置
    auto node_buf = offset_to_loc(page_buf_map[sector], locs[new_nhood.size()]);
    // 构造一个可修改的目标节点对象
    DiskNode<T> target_node(target_id, offset_to_node_coords(node_buf), offset_to_node_nhood(node_buf));
    // 将新插入的向量数据复制到目标节点的坐标区域
    memcpy(target_node.coords, point, data_dim * sizeof(T));
    target_node.nnbrs = new_nhood.size();// 更新目标节点的邻居数量
    // 在邻居数组前一个位置存储邻居数量（约定俗成的存储方式）
    *(target_node.nbrs - 1) = target_node.nnbrs;
    // 将新的邻居ID列表复制到目标节点的邻居区域
    memcpy(target_node.nbrs, new_nhood.data(), new_nhood.size() * sizeof(uint32_t));
    tags.insert_or_assign(target_id, tag);// 标签映射中插入或更新目标节点的标签

    // update the neighbors
    for (uint32_t i = 0; i < new_nhood.size(); ++i) {
      // 获取当前邻居节点的扇区号
      auto r_sector = node_sector_no(new_nhood[i]);
      // 获取当前邻居节点的缓冲区位置
      if (page_buf_map.find(r_sector) == page_buf_map.end()) {
        LOG(ERROR) << new_nhood[i] << " " << "Sector " << r_sector << " not found in page_buf_map";
        exit(-1);
      }
      // 从页面缓冲区中定位到具体节点
      auto r_node_buf = offset_to_node(page_buf_map[r_sector], new_nhood[i]);
      // 创建一个可修改的邻居节点对象
      DiskNode<T> r_nbr_node(new_nhood[i], offset_to_node_coords(r_node_buf), offset_to_node_nhood(r_node_buf));
      // 为邻居节点创建新的邻居列表（容量+1用于添加新节点）
      std::vector<uint32_t> nhood(r_nbr_node.nnbrs + 1);
      // 将邻居节点的当前邻居列表复制到新向量
      nhood.assign(r_nbr_node.nbrs, r_nbr_node.nbrs + r_nbr_node.nnbrs);
      // 将目标节点ID添加到邻居节点的邻居列表中
      nhood.emplace_back(target_id);  // attention: we do not reuse IDs.

      if (nhood.size() > this->range) {  // prune neighbors
#ifdef DELTA_PRUNING
        auto &thread_pq_buf = read_data->aligned_pq_coord_scratch;// 获取线程专用的PQ缓冲区
        // tgt_dists：目标节点到邻居的距离;nbr_dists：当前邻居节点到邻居的距离
        std::vector<float> tgt_dists(nhood.size(), 0.0f), nbr_dists(nhood.size(), 0.0f);
        // 计算目标节点到所有邻居的距离
        compute_pq_dists(target_id, nhood.data(), tgt_dists.data(), (_u32) nhood.size(), thread_pq_buf);
        // 计算当前邻居节点到所有邻居的距离
        compute_pq_dists(r_nbr_node.id, nhood.data(), nbr_dists.data(), (_u32) nhood.size(), thread_pq_buf);
        // 三角形不等式相关的邻居信息
        std::vector<TriangleNeighbor> tri_pool(nhood.size());

        // 填充 tri_pool 向量
        for (uint32_t k = 0; k < nhood.size(); k++) {
          tri_pool[k].id = nhood[k];
          tri_pool[k].tgt_dis = tgt_dists[k];
          tri_pool[k].distance = nbr_dists[k];
        }
        std::sort(tri_pool.begin(), tri_pool.end());//根据距离对三角形邻居池进行排序

        // 获取目标节点在三角形邻居池中的索引
        int tgt_idx = -1;
        for (int k = 0; k < (int) nhood.size(); ++k) {
          if (tri_pool[k].id == target_id) {
            tgt_idx = k;
            break;
          }
        }
        if (unlikely(tgt_idx == -1)) {
          LOG(ERROR) << "Target ID " << target_id << " not found in tri_pool";
          exit(-1);
        }
        // 使用三角不等式对邻居剪枝
        this->delta_prune_neighbors_pq(tri_pool, nhood, thread_pq_buf, tgt_idx);
#else   // 标准剪枝
        std::vector<float> dists(nhood.size(), 0.0f);//存储到邻居节点的距离
        std::vector<Neighbor> pool(nhood.size());// 存储邻居节点及其距离信息
        auto &thread_pq_buf = read_data->aligned_pq_coord_scratch;// 获取线程专用的PQ缓冲区
        // 计算当前邻居节点到所有邻居的距离
        compute_pq_dists(r_nbr_node.id, nhood.data(), dists.data(), (_u32) nhood.size(), thread_pq_buf);
        // 邻居ID和距离填充邻居池
        for (uint32_t k = 0; k < nhood.size(); k++) {
          pool[k].id = nhood[k];
          pool[k].distance = dists[k];
        }
        nhood.clear();// 清空邻居向量
        std::sort(pool.begin(), pool.end());// 根据距离对邻居池进行排序
        this->prune_neighbors_pq(pool, nhood, thread_pq_buf);// 执行PQ剪枝，更新邻居列表
#endif
      }

      auto w_sector = loc_sector_no(locs[i]);//获取写入位置所在的扇区号
      auto w_node_buf = offset_to_loc(page_buf_map[w_sector], locs[i]);//从页面缓冲区映射中找到写入位置
      /*
      // DiskNode<T> 内存布局示意
  [coords][nnbrs存储位置][neighbors array]
  ↓           ↓              ↓
  |           |              |
  |           |              +-- 实际邻居ID数组 (uint32_t[])
  |           +-- 存储邻居数量(nnbrs)的位置
  +-- 坐标数据 (T[])
  假设有一个邻居节点B，现在要把新节点A加进来：
  读取B节点：B的坐标 + B的邻居列表
  添加A到B的邻居列表：B的邻居列表变成 [..., A]
  如果邻居太多进行剪枝：B的邻居列表变成 [..., A, ...]（更少）
  写回B节点：
  B的坐标 = B的原始坐标（不变）
  B的邻居数量 = 剪枝后的新数量
  B的邻居列表 = 剪枝后的新邻居列表
      */
      // 创建一个可修改的邻居节点对象用于写入
      DiskNode<T> w_nbr_node(new_nhood[i], offset_to_node_coords(w_node_buf), offset_to_node_nhood(w_node_buf));
      // 更新剪枝后邻居节点的邻居数量
      w_nbr_node.nnbrs = (_u32) nhood.size();
      // 在邻居数组前一个位置存储数量
      *(w_nbr_node.nbrs - 1) = (_u32) nhood.size();  // write to buf
      // 将原始邻居节点的坐标数据复制到写入节点
      // 虽然我们更新了邻居列表（剪枝），但节点本身的向量坐标保持不变
      memcpy(w_nbr_node.coords, r_nbr_node.coords, data_dim * sizeof(T));
      // 将剪枝后的neighbors array复制到写入节点
      memcpy(w_nbr_node.nbrs, nhood.data(), w_nbr_node.nnbrs * sizeof(uint32_t));
    }

    std::vector<uint64_t> write_page_ref;
    reader->wbc_write(writes, ctx, &write_page_ref);//使用写回缓存（write-back cache）执行批量写入操作
    if (trace) {
      LOG(INFO) << "[TraceInsert] id=" << target_id << " step=write_back"
                << " ms=" << (trace_timer.elapsed() / 1.0e3f)
                << " rss=" << get_rss_mb() << " MB";
    }

#ifndef IN_PLACE_RECORD_UPDATE
    // update locs
    // no concurrency issue for target_id (as it can be only inserted).
    // 目标ID映射到其分配的新位置
    id2loc_.insert_or_assign(target_id, locs[new_nhood.size()]);
    //锁定目标ID和邻居ID的索引访问\锁定相关页面的索引访问
    auto locked = lock_idx(idx_lock_table, target_id, new_nhood);
    auto page_locked = lock_page_idx(page_idx_lock_table, target_id, new_nhood);
    std::vector<uint64_t> orig_locs;// 存储邻居节点的原始位置
    for (uint32_t i = 0; i < new_nhood.size(); ++i) {
      // 遍历所有邻居节点 获取邻居节点的原始位置并存储
      orig_locs.emplace_back(id2loc(new_nhood[i]));
      // 更新邻居ID到新位置的映射
      id2loc_.insert_or_assign(new_nhood[i], locs[i]);
    }

    // with lock, for simple concurrency with alloc_loc.
    // Only for convenience, note that locs[new_nhood.size()] -> target.
    new_nhood.push_back(target_id);// 将目标ID添加到邻居列表中
    // 更新位置到ID的映射，确保新位置正确映射回对应的节点ID
    erase_and_set_loc(orig_locs, locs, new_nhood);
    unlock_page_idx(page_idx_lock_table, page_locked);
    unlock_idx(idx_lock_table, locked);
#endif
    unlock_vec(vec_lock_table, target_id, new_nhood);
/*
  假设：
  节点A（ID=100）要插入
  选择了节点B（ID=200）、C（ID=300）作为邻居
  alloc_loc 分配了新位置：B→pos500, C→pos501, A→pos502
  映射更新过程：
  1 最初 id2loc_: {200→pos200, 300→pos300}   loc2id: {pos200→200, pos300→300}
  2 分配新位置后 locs = [pos500, pos501, pos502]  // [B的新位置, C的新位置, A的位置]
  3 orig_locs = [pos200, pos300];  // B和C的原始位置
    locs = [pos500, pos501, pos502];  // 新位置
    new_nhood = [200, 300, 100];  // 节点ID列表
  4 更新映射：id2loc_: {100→pos502, 200→pos500, 300→pos501}
             loc2id:  {pos500→200, pos501→300, pos502→100}
  */


    // LOG(INFO) << "ID " << target_id << " Target loc " << id2loc(target_id);

    // commit writes (in the background thread.)
#ifdef BG_IO_THREAD
    if (!page_ref.empty()) {// 如果有页面引用需要处理
      auto bg_task = new BgTask{// 创建一个新的后台任务对象
          .thread_data = read_data,// 传递查询缓冲区数据
          .writes = std::move(writes),// 移动写入请求到后台任务
          .pages_to_unlock = std::move(pages_locked),// 移动需要解锁的页面列表
          .pages_to_deref = std::move(write_page_ref),// 移动需要取消引用的页面列表
      };
      bg_tasks_inflight.fetch_add(1, std::memory_order_relaxed);
      bg_tasks.push(bg_task);// 将后台任务推送到任务队列
      bg_tasks.push_notify_all();// 通知后台线程有新任务可处理
    } else {
      // 直接解锁页面请求
      v2::unlockReqs(this->page_lock_table, pages_locked);
    }
    reader->deref(&page_ref, ctx);// 取消对读取页面的引用
#else
    reader->write(writes, ctx);// 直接执行写入请求
    v2::unlockReqs(this->page_lock_table, pages_locked);// 解锁页面请求
    reader->deref(&write_page_ref, ctx);// 取消对写入页面的引用

    reader->deref(&page_ref, ctx);// 取消对读取页面的引用
    this->push_query_buf(read_data);// 将查询缓冲区返回池中
#endif
    return target_id;//返回新插入节点的ID
  }

  template<class T, class TagT>
  void SSDIndex<T, TagT>::bg_io_thread() {
    auto ctx = reader->get_ctx();//从读取器获取I/O操作上下文
    uint8_t *buf = nullptr;
    pipeann::alloc_aligned((void **) &buf, (MAX_N_EDGES + 1) * SECTOR_LEN, SECTOR_LEN);
    auto timer = pipeann::Timer();
    uint64_t n_tasks = 0;

    while (true) {
      auto task = bg_tasks.pop();
      while (task == nullptr) {
        this->bg_tasks.wait_for_push_notify();
        task = bg_tasks.pop();
      }

      reader->write(task->writes, ctx);
      v2::unlockReqs(this->page_lock_table, task->pages_to_unlock);
      reader->deref(&task->pages_to_deref, ctx);
      this->push_query_buf(task->thread_data);// 将查询缓冲区返回池中
      delete task;
      bg_tasks_inflight.fetch_sub(1, std::memory_order_relaxed);
      ++n_tasks;

      if (timer.elapsed() >= 5000000) {
        LOG(INFO) << "Processed " << n_tasks << " tasks, throughput: " << (double) n_tasks * 1e6 / timer.elapsed()
                  << " tasks/sec.";
        timer.reset();
        n_tasks = 0;
      }
    }
  }

  template class SSDIndex<float>;
  template class SSDIndex<_s8>;
  template class SSDIndex<_u8>;
}  // namespace pipeann
