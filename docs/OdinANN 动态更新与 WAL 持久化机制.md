## OdinANN 动态更新与 WAL 持久化机制

### 基于 `lazy_wal_hot_service.cpp` 的动态更新与崩溃恢复分析

#### 概览

基于 `lazy_wal_hot_service.cpp` 的源码实现，并结合 NVMe 服务器上的实际运行日志，本文分析 OdinANN 热更新服务中的以下机制：

* 在线插入与在线删除的执行路径
* WAL 的记录格式、刷盘方式与 group commit 机制
* 插入后立即可搜索的实现方式
* 服务异常退出后的 WAL replay 恢复流程
* `delete + insert + verify + SIGKILL + replay + verify` 实验结果解读

实验中服务器使用的命令如下：

```bash
/home/mabojing/ODinANN-NEW/build/tests/lazy_wal_hot_service uint8 \
  /home/mabojing/ODinANN-NEW/index/R32_L100_B33_M16/bigann_09B_R32_L100_B33_M16_merge \
  /home/mabojing/ODinANN-NEW/data/BIGANN_0.1B_mutated_100k.u8bin \
  100 28 4 0 18083
```

该程序是一个长期驻留的 TCP 服务。服务在加载基础磁盘索引后，对外提供插入、删除、搜索、WAL 管理、merge 和 shutdown 等命令接口，并在每次更新时先写入 WAL，再修改动态索引状态。

从机制上看，这个程序实现的是：

* 基础磁盘索引常驻
* 增量更新以 lazy 方式作用于动态状态
* 增量操作通过 WAL 持久化
* 重启后通过 replay 恢复未 merge 的更新

本实验关注的重点是未执行 merge 时，动态更新状态能否仅依赖 WAL 完成崩溃恢复，并保持搜索可用。

---

#### 服务功能与命令接口

`lazy_wal_hot_service.cpp` 中的命令分发表明，该程序是一个完整的在线服务，而不是单轮测试程序。

```cpp
return "OK commands: HELP STATUS STATS INSERT_RANDOM n [threads] INSERT_ROWS csv_rows [threads] "
       "INSERT_RANDOM_ASYNC n [threads] INSERT_ROWS_ASYNC csv_rows [threads] ASYNC_WAIT [timeout_sec] "
       "DELETE_BASE_RANDOM n DELETE_INSERTED n DELETE_TAGS csv_tags SEARCH_ROW row k [L] [beam] "
       "SEARCH_TAG tag k [L] [beam] SEARCH_RANDOM nq k [L] [beam] WAL_INFO WAL_FLUSH MERGE [threads] QUIT SHUTDOWN";
```

从这个接口可以看出，程序支持以下几类操作：

* 更新类：`INSERT_RANDOM`、`INSERT_ROWS`、`INSERT_RANDOM_ASYNC`、`INSERT_ROWS_ASYNC`
* 删除类：`DELETE_BASE_RANDOM`、`DELETE_INSERTED`、`DELETE_TAGS`
* 检索类：`SEARCH_ROW`、`SEARCH_TAG`、`SEARCH_RANDOM`
* WAL 管理类：`WAL_INFO`、`WAL_FLUSH`
* 状态维护类：`STATUS`、`STATS`、`MERGE`

其中，本次实验实际主要使用的是：

* `DELETE_BASE_RANDOM`（随机删除基座 tag）
* `DELETE_TAGS`（删除随机抽样的已插入 tag）
* `INSERT_ROWS` / `INSERT_RANDOM`
* `SEARCH_TAG`

下文重点分析这几条路径及其与 WAL 的配合关系。

---

#### 核心对象与状态组织

源码中的核心服务类为：

```cpp
template<typename T, typename TagT>
class LazyWalHotService
```

该类内部维护了三类核心状态：

* 动态索引对象：`std::unique_ptr<pipeann::DynamicSSDIndex<T, TagT>> index_`
* 新插入 tag 到原始数据行号的映射：`std::unordered_map<TagT, uint64_t> tag_to_row_`
* WAL 管理对象：`LazyWal wal_`

此外还维护了：

* `next_tag_`：下一个可分配 tag
* `wal_pending_`：尚未刷盘的 WAL 缓冲
* `async_queue_`：异步插入任务队列
* 多组统计计数器：插入数、删除数、查询数、replay 数等

这一状态组织方式决定了程序的基本工作模式：

* 静态数据仍由底层磁盘索引负责
* 新插入数据通过 `index_->insert(...)` 加入动态结构
* 新插入 tag 的查询辅助信息由 `tag_to_row_` 维护
* 更新操作的持久化由 `LazyWal` 负责

---

#### WAL记录格式与落盘方式

源码中 WAL 记录结构如下：

```cpp
struct WalEntry {
  // I <tag> <row>
  // D <tag>
  // M
  char op = '\0';
  uint64_t tag = 0;
  uint64_t row = 0;
};
```

这里定义了三种 WAL 操作：

* `I`：插入记录，保存新 tag 及其来源 row
* `D`：删除记录，仅保存 tag
* `M`：merge 标记，不带额外参数

WAL 写入逻辑如下：

```cpp
void append_batch(const std::vector<WalEntry> &entries) const {
  if (entries.empty()) return;
  int fd = ::open(path_.c_str(), O_WRONLY | O_CREAT | O_APPEND, 0644);
  ...
  for (const auto &e : entries) {
    std::string line;
    if (e.op == 'I') {
      line = "I " + std::to_string(e.tag) + " " + std::to_string(e.row) + "\n";
    } else if (e.op == 'D') {
      line = "D " + std::to_string(e.tag) + "\n";
    } else if (e.op == 'M') {
      line = "M\n";
    }
    ...
    ::write(fd, p, left);
  }
  if (::fsync(fd) != 0) {
    throw std::runtime_error("fsync wal failed ...");
  }
  ::close(fd);
}
```

由上述实现可以看出：

* 当前实现使用文本 WAL，而非二进制 WAL
* 每条记录按追加方式写入文件尾部
* 每次 batch 完成后显式调用 `fsync`

这意味着当前实现采用了较强的 WAL 持久化语义。只要 `append_batch` 返回成功，对应 batch 即已完成落盘；即使服务随后被 `SIGKILL`，这些操作仍可在下次启动时重新读取并回放。

---

#### group commit机制

程序没有强制每条操作单独写盘，而是提供了简单的 group commit 参数：

```cpp
wal_group_commit_ops_ = std::max<uint64_t>(1, get_env_u64("PIPEANN_LAZY_WAL_GROUP_COMMIT_OPS", 1));
wal_group_commit_ms_ = get_env_u64("PIPEANN_LAZY_WAL_GROUP_COMMIT_MS", 0);
```

更新操作先进入内存缓冲 `wal_pending_`：

```cpp
double append_wal_entries(const std::vector<WalEntry> &entries, bool force_flush) {
  std::lock_guard<std::mutex> g(wal_mu_);
  wal_pending_.insert(wal_pending_.end(), entries.begin(), entries.end());
  wal_pending_ops_.store((uint64_t) wal_pending_.size(), std::memory_order_relaxed);
  const bool reach_ops = wal_pending_.size() >= wal_group_commit_ops_;
  bool reach_time = false;
  if (!wal_pending_.empty() && wal_group_commit_ms_ > 0) {
    ...
    reach_time = ms >= (int64_t) wal_group_commit_ms_;
  }
  if (force_flush || reach_ops || reach_time) {
    flush_wal_pending_locked();
  }
  ...
}
```

真正的刷盘动作由 `flush_wal_pending_locked()` 执行：

```cpp
void flush_wal_pending_locked() {
  if (wal_pending_.empty()) return;
  wal_.append_batch(wal_pending_);
  wal_pending_.clear();
  wal_last_flush_tp_ = std::chrono::steady_clock::now();
  wal_pending_ops_.store(0, std::memory_order_relaxed);
  wal_flush_count_.fetch_add(1, std::memory_order_relaxed);
}
```

本次实验日志中：

```text
GROUP_COMMIT_OPS=1 GROUP_COMMIT_MS=0
```

因此，本次实验的实际行为等价于：

* 操作一旦进入 WAL 缓冲，就立即达到刷盘阈值
* 不依赖时间窗口攒批
* `wal_ms` 统计值可以近似看成“同步写 WAL”代价

因此，本实验更适合用于验证崩溃恢复正确性，而不是评估极限吞吐。

---

#### 服务启动阶段

服务启动的核心逻辑位于 `LazyWalHotService` 构造函数中：

```cpp
LazyWalHotService(...)
    : data_bin_(data_bin),
      nthreads_(nthreads),
      beamwidth_(beamwidth),
      L_disk_(L_disk),
      rng_(get_env_u64("PIPEANN_SERVICE_SEED", 12345ULL)),
      wal_(get_wal_path(index_prefix)) {
  ...
  maybe_set_pq_extra_points(expected_insert);
  h_ = read_header(data_bin_);

  pipeann::Parameters paras;
  paras.Set<unsigned>("L_disk", L_disk);
  ...
  index_ = std::make_unique<pipeann::DynamicSSDIndex<T, TagT>>(
      paras, index_prefix, index_prefix + "_merge", nullptr, metric, 0, false);

  LOG(INFO) << "=== Warmup phase: index loaded ===";
  warmup_search(*index_, data_bin_, h_, rng_, warmup_queries, warmup_k, warmup_l, beamwidth_);

  const uint64_t base = index_->_disk_index->num_points;
  next_tag_.store(base, std::memory_order_relaxed);
  base_tag_.store(base, std::memory_order_relaxed);
  maybe_pre_extend_disk_index(*index_, base + expected_insert);

  replay_wal();
  start_async_worker();
}
```

这段代码对应的启动流程为：

1. 读取环境变量配置
2. 读取数据文件头信息
3. 构造 `DynamicSSDIndex`
4. 执行 warmup search
5. 初始化 `next_tag_`
6. 预扩展磁盘索引文件
7. 回放 WAL
8. 启动异步插入线程

与 phase1 日志对照：

```text
[lazy_wal_hot_service.cpp:270:INFO] PIPEANN_PQ_EXTRA_POINTS already set: 111100
[lazy_wal_hot_service.cpp:299:INFO] [PreExtend] Disk index file size set to 260445 MB (estimate total points=1000111100)
[lazy_wal_hot_service.cpp:843:INFO] [WAL] empty, nothing to replay.
[lazy_wal_hot_service.cpp:1158:INFO] lazy_wal_hot_service listening on 0.0.0.0:18083
```

phase1 日志表明：

* 启动前已经通过环境变量指定了 `PIPEANN_PQ_EXTRA_POINTS=111100`
* 磁盘索引文件被提前扩展到预估容量
* WAL 文件为空，因此无需恢复

与 phase2 日志对照：

```text
[lazy_wal_hot_service.cpp:888:INFO] [WAL] replay done. ... entries=233300 last_merge_idx=-1 replay_inserts=111100 replay_deletes=122200 next_tag=1000111100 pending_inserted=100000
```

phase2 与 phase1 的主要区别在于，phase2 启动时执行了完整的 WAL replay。

---

#### 在线插入路径

本次实验使用的是 `INSERT_ROWS` 路径，其核心实现如下：

```cpp
std::string insert_rows_cmd(const std::vector<uint64_t> &rows, uint32_t use_threads) {
  if (rows.empty()) return "ERR empty rows";
  for (auto r : rows) {
    if (r >= h_.npts) return "ERR row out of range";
  }

  uint64_t start_tag = 0;
  if (!reserve_tags(rows.size(), start_tag)) {
    return "ERR tag overflow";
  }

  std::vector<T> vecs;
  read_rows<T>(data_bin_, h_, rows, vecs);
  std::vector<TagT> tags(rows.size());
  for (size_t i = 0; i < rows.size(); ++i) {
    tags[i] = (TagT) (start_tag + i);
  }

  std::vector<WalEntry> entries;
  entries.reserve(rows.size());
  for (size_t i = 0; i < rows.size(); ++i) {
    entries.push_back(WalEntry{'I', (uint64_t) tags[i], rows[i]});
  }
  const double wal_ms = append_wal_entries(entries, false);

  {
    std::shared_lock<std::shared_mutex> lk(state_mu_);
    parallel_insert(vecs, tags, use_threads);
  }

  {
    std::unique_lock<std::shared_mutex> lk(map_mu_);
    for (size_t i = 0; i < rows.size(); ++i) {
      tag_to_row_[tags[i]] = rows[i];
    }
  }
  ...
}
```

这条路径的执行顺序如下：

* 校验输入 row 合法性
* 通过 `reserve_tags()` 原子分配连续的新 tag
* 从数据文件中读取待插入向量
* 将插入操作转为一批 `WalEntry{'I', tag, row}`
* 调用 `append_wal_entries()` 先写 WAL
* 调用 `parallel_insert()` 真正插入索引
* 更新 `tag_to_row_`

其中，tag 分配函数如下：

```cpp
bool reserve_tags(uint64_t n, uint64_t &start) {
  while (true) {
    uint64_t cur = next_tag_.load(std::memory_order_relaxed);
    if (cur + n > (uint64_t) std::numeric_limits<TagT>::max()) {
      return false;
    }
    if (next_tag_.compare_exchange_weak(cur, cur + n, std::memory_order_relaxed)) {
      start = cur;
      return true;
    }
  }
}
```

这表明新 tag 的分配采用原子递增方式，不会与并发请求冲突。

本次实验中的插入日志：

```text
Insert phase ok: OK insert_rows n=100 start_tag=1000000000 ...
Insert phase ok: OK insert_rows n=1000 start_tag=1000000100 ...
Insert phase ok: OK insert_rows n=10000 start_tag=1000001100 ...
```

正是该机制的直接结果。由于基础索引点数为 `1000000000`，所以新插入数据从该编号开始连续分配。

---

#### 并行插入实现

插入实际执行由 `parallel_insert()` 完成：

```cpp
void parallel_insert(const std::vector<T> &vecs, const std::vector<TagT> &tags, uint32_t use_threads) {
  const size_t n = tags.size();
  if (use_threads <= 1) {
    for (size_t i = 0; i < n; ++i) {
      index_->insert(vecs.data() + i * h_.dim, tags[i]);
    }
  } else {
#pragma omp parallel for schedule(static) num_threads(use_threads)
    for (int64_t i = 0; i < (int64_t) n; ++i) {
      index_->insert(vecs.data() + (size_t) i * h_.dim, tags[(size_t) i]);
    }
  }
}
```

可以看出：

* 多线程插入不是服务端“开很多请求线程”实现的
* 而是单次命令内部通过 OpenMP 并行执行 `index_->insert(...)`

因此，本次实验中的：

```text
threads=28
```

对应 28 线程的并行插入执行。吞吐结果如下：

* `100` 条：`2509.85 tps`
* `1000` 条：`3416.81 tps`
* `10000` 条：`4101.10 tps`
* `100000` 条：`2895.76 tps`（该批次走 `INSERT_RANDOM`）

从源码结构看，这些性能数字主要由以下几部分共同决定：

* `append_wal_entries()` 的刷盘成本
* `read_rows()` 的数据读取成本
* `parallel_insert()` 的多线程执行效率

---

#### 在线删除路径

本次实验删除是“混合删除”：

* 基座删除：`DELETE_BASE_RANDOM n`
* 新插入删除：脚本随机抽样 `ACTIVE_INSERTED_TAGS` 后调用 `DELETE_TAGS csv_tags`

`DELETE_TAGS` 的实现如下：

```cpp
std::string delete_tags_cmd(const std::vector<TagT> &tags) {
  if (tags.empty()) return "ERR empty tags";
  std::vector<WalEntry> entries;
  entries.reserve(tags.size());
  for (auto tg : tags) entries.push_back(WalEntry{'D', (uint64_t) tg, 0});
  const double wal_ms = append_wal_entries(entries, false);

  {
    std::shared_lock<std::shared_mutex> lk(state_mu_);
    for (auto tg : tags) {
      index_->lazy_delete(tg);
    }
  }

  {
    std::unique_lock<std::shared_mutex> lk(map_mu_);
    for (auto tg : tags) tag_to_row_.erase(tg);
  }
  ...
}
```

这条路径同样遵循“先写 WAL，再修改索引”的顺序。

删除真正执行的操作是：

```cpp
index_->lazy_delete(tg);
```

从函数名可以看出，这里采用的是逻辑删除，而不是立即改写主索引。这种方式适合与 WAL 结合：

* 更新成本低
* merge 可延后
* 崩溃后可以通过 replay 重新建立删除状态

本次实验日志中，两条删除路径都被覆盖：

```text
Delete base random phase ok: OK delete_base_random n=100 ...
Delete base random phase ok: OK delete_base_random n=1000 ...
Delete base random phase ok: OK delete_base_random n=10000 ...


Delete inserted phase ok: OK delete_tags n=100 ...
Delete inserted phase ok: OK delete_tags n=1000 ...
Delete inserted phase ok: OK delete_tags n=10000 ...
```

说明“随机基座删除 + 已插入删除”两类操作均已执行，并写入 WAL。

---

#### 插入后立即可搜索的原因

源码中 `SEARCH_TAG` 的实现如下：

```cpp
std::string search_tag(TagT tag, uint64_t k, uint64_t l_search, uint32_t beam) {
  uint64_t row = 0;
  {
    std::shared_lock<std::shared_mutex> lk(map_mu_);
    auto it = tag_to_row_.find(tag);
    if (it == tag_to_row_.end()) {
      return "ERR tag not found in pending inserted set";
    }
    row = it->second;
  }
  return search_row(row, k, l_search, beam);
}
```

而 `search_row()` 的核心是：

```cpp
read_rows<T>(data_bin_, h_, rows, qv);
...
index_->search(qv.data(), k, 0, l_search, beam, tags.data(), dists.data(), nullptr, true);
```

实现逻辑如下：

* `SEARCH_TAG` 并不是直接根据 tag 找到存储位置
* 而是先通过 `tag_to_row_` 找到该 tag 对应的原始 row
* 再从数据文件读取这条向量
* 最后调用 `index_->search(...)` 对该向量做 ANN 搜索

因此，只要插入已经通过 `index_->insert(...)` 生效，那么以该向量作为查询时，搜索结果中应当能够命中刚插入的 tag。

本次实验中三轮校验都成功：

```text
phase1 immediate count=100: searchable verification passed for 100 tags
phase1 immediate count=1000: searchable verification passed for 1000 tags
phase1 immediate count=10000: searchable verification passed for 10000 tags
```

上述结果对应以下状态：

* WAL 已成功写入
* 索引插入已成功执行
* `tag_to_row_` 映射已建立
* 新插入向量已经进入了检索路径

---

#### 异步插入机制

虽然本实验未重点使用异步插入，但源码中提供了完整的异步路径：

* `INSERT_RANDOM_ASYNC`
* `INSERT_ROWS_ASYNC`

其核心逻辑是：

* 主线程负责分配 tag、写 WAL、建立 `tag_to_row_`
* 后台线程负责真正执行 `parallel_insert()`

任务入队：

```cpp
void enqueue_insert_task(PendingInsertTask &&task) {
  const uint64_t n = (uint64_t) task.tags.size();
  {
    std::lock_guard<std::mutex> lk(async_mu_);
    async_queue_.emplace_back(std::move(task));
  }
  async_pending_points_.fetch_add(n, std::memory_order_relaxed);
  async_enqueued_points_.fetch_add(n, std::memory_order_relaxed);
  async_cv_.notify_one();
}
```

后台执行：

```cpp
void async_insert_worker_loop() {
  while (true) {
    ...
    read_rows<T>(data_bin_, h_, task.rows, vecs);
    {
      std::shared_lock<std::shared_mutex> lk(state_mu_);
      parallel_insert(vecs, task.tags, task.use_threads);
    }
    total_inserted_.fetch_add(n, std::memory_order_relaxed);
    async_applied_points_.fetch_add(n, std::memory_order_relaxed);
    ...
  }
}
```

这套机制的特点在于：

* 请求接受与实际插入解耦
* WAL 持久化优先于索引变更
* 即便后台执行较慢，只要 WAL 已落盘，崩溃后仍可通过 replay 恢复

---

#### merge路径与WAL清空

源码中的 merge 逻辑如下：

```cpp
std::string merge_now(uint32_t use_threads) {
  wait_async_drain(0);
  {
    std::unique_lock<std::shared_mutex> lk(state_mu_);
    ...
    index_->final_merge_stream_pq_tags(use_threads);
    const uint64_t now = index_->_disk_index ? index_->_disk_index->num_points : next_tag_.load(...);
    next_tag_.store(now, std::memory_order_relaxed);
    base_tag_.store(now, std::memory_order_relaxed);
  }
  {
    std::unique_lock<std::shared_mutex> lk(map_mu_);
    tag_to_row_.clear();
  }
  ...
  flush_wal_pending_locked();
  wal_.append_merge_marker();
  wal_.clear();
  wal_pending_.clear();
  ...
}
```

merge 完成后程序会：

* 将动态更新合并进主索引
* 清空 `tag_to_row_`
* 在 WAL 中追加一个 `M`
* 随后直接清空整个 WAL 文件

因此，当前设计中的 WAL 并不承担长期保存全部历史的职责，而是记录“自上次 merge 之后尚未固化的增量操作”。

本次实验日志中：

```text
Phase1 WAL counts (I D M): 111100 122200 0
Phase2 WAL counts (I D M): 111100 122200 0
```

这表明整个实验过程中未触发 merge，因此恢复阶段需要完整回放全部更新。

---

#### WAL replay恢复机制

恢复逻辑由 `replay_wal()` 实现：

```cpp
void replay_wal() {
  std::vector<WalEntry> all_entries;
  {
    std::lock_guard<std::mutex> g(wal_mu_);
    all_entries = wal_.load_all();
  }
  if (all_entries.empty()) {
    LOG(INFO) << "[WAL] empty, nothing to replay. path=" << wal_.path();
    return;
  }

  ssize_t last_merge = -1;
  for (size_t i = 0; i < all_entries.size(); ++i) {
    if (all_entries[i].op == 'M') last_merge = (ssize_t) i;
  }

  uint64_t replay_ins = 0, replay_del = 0;
  uint64_t max_seen_tag = next_tag_.load(std::memory_order_relaxed);
  for (size_t i = (size_t) (last_merge + 1); i < all_entries.size(); ++i) {
    const auto &e = all_entries[i];
    if (e.op == 'I') {
      ...
      index_->insert(vecs.data(), tg);
      tag_to_row_[tg] = e.row;
      max_seen_tag = std::max<uint64_t>(max_seen_tag, e.tag + 1);
      ++replay_ins;
    } else if (e.op == 'D') {
      index_->lazy_delete((TagT) e.tag);
      tag_to_row_.erase((TagT) e.tag);
      max_seen_tag = std::max<uint64_t>(max_seen_tag, e.tag + 1);
      ++replay_del;
    }
  }
  wait_bg_tasks();
  next_tag_.store(std::max<uint64_t>(next_tag_.load(...), max_seen_tag), ...);
  replayed_inserts_.store(replay_ins, ...);
  replayed_deletes_.store(replay_del, ...);
  LOG(INFO) << "[WAL] replay done. ..."
}
```

恢复流程如下：

* 读出 WAL 全部记录
* 找到最后一个 merge marker
* 仅回放其后的记录
* 对 `I` 重新执行 `index_->insert(...)`
* 对 `D` 重新执行 `index_->lazy_delete(...)`
* 重建 `tag_to_row_`
* 修正 `next_tag_`

由此可见，replay 恢复的对象不是统计量，而是真实的可搜索状态。

本次 phase2 日志：

```text
[WAL] replay done. path=... entries=233300 last_merge_idx=-1 replay_inserts=111100 replay_deletes=122200 next_tag=1000111100 pending_inserted=100000
```

日志字段可与源码逐项对应：

* `entries=233300`：共读取到 `111100` 条插入记录和 `122200` 条删除记录
* `last_merge_idx=-1`：没有 merge marker
* `replay_inserts=111100`：回放了全部插入
* `replay_deletes=122200`：回放了全部删除
* `next_tag=1000111100`：下一个可分配 tag 被正确推进
* `pending_inserted=100000`：仍有 `100000` 个未 merge 的新插入 tag

---

#### 本次实验结果分析（2026-04-13，28线程）

##### 正确性与恢复结果

本次实验执行流程为：

* phase1 启动服务
* 删除 `100 / 1000 / 10000 / 100000`
* 插入 `100 / 1000 / 10000 / 100000`
* `SIGKILL` 强制杀死服务
* phase2 重启服务并 replay WAL

最终结果为：

```text
PASS: delete/insert immediate-search and restart replay validation succeeded without merge.
Replay counters ok: replayed_inserts=111100, replayed_deletes=122200, pending_inserted=100000
```

WAL 计数一致：

```text
Phase1 WAL counts (I D M): 111100 122200 0
Phase2 WAL counts (I D M): 111100 122200 0
```

说明在未 merge 的条件下，崩溃后仅依赖 WAL replay 即可恢复全部动态增量状态。

##### Lazy delete 速率

本轮删除分两段：

* 基座随机删除（`DELETE_BASE_RANDOM`）：
  * `n=100`: `time_ms=0.03`, `wal_ms=0.569`
  * `n=1000`: `time_ms=0.073`, `wal_ms=2.818`
  * `n=10000`: `time_ms=0.503`, `wal_ms=23.413`
  * `n=100000`: `time_ms=4.126`, `wal_ms=220.397`
* 已插入随机删除（`DELETE_TAGS`）：
  * `n=100`: `time_ms=0.012`, `wal_ms=0.611`
  * `n=1000`: `time_ms=0.085`, `wal_ms=2.713`
  * `n=10000`: `time_ms=0.569`, `wal_ms=22.671`

其中 `lazy_delete` 本体（`time_ms`）开销很小，删除路径的主要成本来自 WAL 持久化与批次 I/O。

##### Lazy insert 速率与 WAL 占比

插入命令（`INSERT_ROWS/INSERT_RANDOM`）输出了 `tps`、`time_ms` 与 `wal_ms`，`wal_ratio_pct` 定义为：

`wal_ratio_pct = wal_ms / (time_ms + wal_ms) * 100%`

分批结果如下：

| 批次 | 命令 | time_ms | wal_ms | tps | wal_ratio_pct |
|---|---|---:|---:|---:|---:|
| 100 | INSERT_ROWS | 65.628 | 0.572 | 1523.74 | 0.864% |
| 1000 | INSERT_ROWS | 395.959 | 2.738 | 2525.51 | 0.687% |
| 10000 | INSERT_ROWS | 2208.64 | 24.086 | 4527.68 | 1.079% |
| 100000 | INSERT_RANDOM | 30135.1 | 229.116 | 3318.39 | 0.755% |

总计结果（脚本汇总）：

* `total_vectors=111100`
* `insert_time_ms_sum=32805.327`
* `wal_time_ms_sum=256.512`
* `e2e_time_ms_sum=33061.839`
* `core_tps=3386.645`
* `e2e_tps_with_wal=3360.370`
* `wal_ratio_pct=0.776`

这组数据表明：在 `GROUP_COMMIT_OPS=1`、`GROUP_COMMIT_MS=0`（同步刷盘）下，WAL 对热插入端到端耗时的占比约为 `0.776%`，对总体吞吐的影响较小。

##### 插入与删除后的可搜索性验证（含混合删除）

围绕 `lazy insert` 和 `lazy delete` 后的检索状态，当前实验的验证链条可分为三层：

1. **代码执行顺序保证（先 WAL，后索引）**

在 `insert_rows_cmd()` 与 `delete_tags_cmd()` 中，都是先构造 `WalEntry`，调用 `append_wal_entries(...)`，再执行 `index_->insert(...)` / `index_->lazy_delete(...)`。  
这保证了崩溃恢复时至少不会出现“索引变了但 WAL 没记住”的顺序反转。

2. **在线可搜索性验证（phase1）**

脚本在每轮插入后都会做 `SEARCH_TAG` 验证：  
小批次（100/1000/10000）做全量验证；大批次（100000）做抽样验证，且使用更强参数 `k=50, L=2000, beam=64` 与命中率阈值 `0.98`。

本次结果：

* phase1 `count=10000` 抽样：`hit=3000/3000, hit_ratio=1.0000`
* phase1 `count=100000` 抽样：`hit=3000/3000, hit_ratio=1.0000`
* phase2 replay 后抽样：`hit=3000/3000, hit_ratio=1.0000`

这表明插入数据在 phase1 可以被检索命中，且重启并 replay 后仍可命中。

3. **崩溃恢复一致性验证（phase2）**

`SIGKILL` 后重启，日志给出：

* `replayed_inserts=111100`
* `replayed_deletes=122200`
* `Phase1/Phase2 WAL counts` 一致

结合 `replay_wal()` 的实现方式，即逐条回放 `I/D` 到 `index_`，可以看出恢复的是索引状态本身，而不仅是计数器。

删除验证分为两类：

1. **删除已插入向量后的可见性验证**

脚本新增了 `INSERTED_DELETE_COUNTS` 阶段，对刚插入的新 tag 再执行删除，并立即验证：

* 期望：`SEARCH_TAG <deleted_tag>` 返回 `ERR tag not found in pending inserted set`
* 含义：删除后这些新插入 tag 已从可搜索集合中移除

对应日志如下：

* `phase1 delete_inserted count=100 ... deleted inserted tags are not searchable`
* `phase1 delete_inserted count=1000 ... deleted inserted tags are not searchable`
* `phase1 delete_inserted count=10000 ... deleted inserted tags are not searchable`

2. **删除基座 tag 后的可见性验证**

基座删除改为 `DELETE_BASE_RANDOM` 后，不再依赖固定 tag 区间。  
在当前服务语义下，基座删除正确性主要由以下几项结果共同支撑：

* phase1 命令返回 `OK delete_base_random n=...`
* phase1 WAL 计数中的 `D` 与配置总删除量一致
* phase2 `replayed_deletes` 与预期一致
* phase2 可搜索状态（active_inserted / deleted_inserted）与 phase1 一致

##### 混合删除补充实验（基座 + 新插入）

本轮删除总量由两部分构成：

* 基座删除：`100 + 1000 + 10000 + 100000 = 111100`
* 新插入删除：`100 + 1000 + 10000 = 11100`
* 总删除：`122200`

phase2 启动后的日志如下：

* `Phase1 WAL counts (I D M): 111100 122200 0`
* `Replay counters ok: replayed_inserts=111100, replayed_deletes=122200, pending_inserted=100000`

该结果与脚本配置一致，表明：

* 混合删除（基座 + 新插入）都被写入 WAL
* 崩溃后 replay 能恢复同样的删除状态
* 已插入向量删除后的不可搜索性已在 phase1 中得到直接验证

##### 关于 client 日志里 `ERR tag not found in pending inserted set`

这条 `ERR` 在 `deleted_inserted` 校验阶段是预期信号，表示该 tag 已从 `pending inserted` 集合中移除。  
它并不表示流程失败。真正的失败条件是：

* 返回了 `OK search_row ...`（说明删后还能搜到）
* 或返回了其他非预期 `ERR`

脚本已可将该条预期 `ERR` 归一化显示为 `OK expected_not_found_deleted_tag`，仅改变日志可读性，不改变判定逻辑。

##### WAL 时间统计口径与“为什么看起来很小”（详细）

本实验中 `wal_ms` 来自服务端每个命令返回值中的字段（例如 `OK insert_rows ... wal_ms=...`），该字段由 `append_wal_entries(...)` 计时得到。  
这段计时包含：

* `wal_pending_` 追加与锁开销
* 触发刷盘时的 `flush_wal_pending_locked()`
* `append_batch()` 内部 `write + fsync` 的耗时

在当前配置 `GROUP_COMMIT_OPS=1, GROUP_COMMIT_MS=0` 下，每条更新命令都会立即刷盘，因此 `wal_ms` 可以近似看作该批命令一次同步 WAL 提交的端到端时间。

脚本里的统计公式是：

* `e2e_time_ms = time_ms + wal_ms`
* `wal_ratio_pct = wal_ms / (time_ms + wal_ms) * 100%`
* 汇总吞吐：`core_tps = total_vectors * 1000 / insert_time_ms_sum`
* 汇总端到端吞吐：`e2e_tps_with_wal = total_vectors * 1000 / e2e_time_ms_sum`

按本轮数据折算：

* 插入 WAL 总时间：`256.512 ms / 111100 vectors = 2.309 us/vector`
* 基座删除 WAL 总时间：`247.197 ms / 111100 deletes = 2.225 us/op`
* 已插入删除 WAL 总时间：`25.995 ms / 11100 deletes = 2.342 us/op`

`wal_ms` 占比较小的主要原因如下：

1. **批量摊薄效应**：每条命令一次 `fsync`，但一次命令包含大量 `I/D` 记录，单条记录平均成本被摊薄。  
2. **更新主成本在索引侧**：插入链路里 `read_rows + parallel_insert` 通常比 WAL 追加更重。  
3. **顺序追加写模式**：WAL 是 append-only 文本写，I/O 模式对磁盘较友好。  
4. **硬件与系统缓存条件较好**：在该环境下 `fsync` 延迟较低，因此整体占比较低。

需要说明的是，`wal_ratio_pct` 是该工作负载在当前机器上的实测值，并非常数。更换磁盘、文件系统、挂载参数或并发配置后，该占比可能变化。

---

#### 结论

基于源码和本次实验结果，可得到以下结论：

* `lazy_insert` 与 `lazy_delete` 均可在在线服务中工作；本次热插入 `core_tps=3386.645`，计入 WAL 后的 `e2e_tps=3360.370`。
* 崩溃恢复链路已完成验证：`SIGKILL` 后 replay 得到 `replayed_inserts=111100`、`replayed_deletes=122200`、`pending_inserted=100000`，与配置及 WAL 计数一致。
* 混合删除（基座随机 + 新插入随机）均可被 WAL 持久化并在重启后恢复；已插入数据删除后的不可搜索性在实验中得到验证。
* 当前配置下插入侧 WAL 占比为 `0.776%`，整体上对吞吐影响较小。
