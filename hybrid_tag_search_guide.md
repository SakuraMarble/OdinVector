# Hybrid Tag-Vector Search Guide

## 功能说明

`hybrid_tag_search` 实现了混合标签向量搜索策略，对每个查询按 hit_rate 自动选择最优路径：

| hit_rate | 策略 | 适用场景 |
|----------|------|----------|
| < threshold | **IVF 路径**：PQ 粗排 + mmap 精排 | 候选集小，精确过滤效果好 |
| ≥ threshold | **图搜索路径**：Beam Search + result_heap 边搜索边过滤 | 候选集大，图搜索效率高 |

## 前置依赖

### CRoaring 库

本机安装路径：`~/.local/lib/croaring`（CMakeLists.txt 已配置，无需额外操作）。

若需在新机器上安装：
```bash
sudo apt-get install libcroaring-dev
```
或从源码编译后将安装路径写入 `tests/CMakeLists.txt` 的 `CROARING_ROOT`。

### 预先构建的文件
- `<prefix>_disk.index`：PipeANN 磁盘索引（由 `build_disk_index` 生成）
- `ivf.index`：倒排索引二进制文件（由 `InvertedIndexBuilder::build` 生成）
- `raw_vector.bin`：原始向量文件，格式 `[num:u32][dim:u32][data...]`
- `query_label.txt`：每行一个布尔表达式（见下方格式说明）

## 构建步骤

```bash
cd /home/mabojing/26FAST-PipeANN/build
cmake ..
make hybrid_tag_search -j4
```

## 使用示例

```bash
./tests/hybrid_tag_search float \
  /path/to/vector_disk_index \
  16 \
  4 \
  /path/to/query_vector.bin \
  /path/to/groundtruth.bin \
  10 \
  l2 \
  0 \
  /path/to/ivf.index \
  /path/to/raw_vector.bin \
  /path/to/query_label.txt \
  0.1 \
  3 \
  50 100 150
```

## 参数说明

| 位置 | 参数 | 说明 |
|------|------|------|
| 1 | `data_type` | 向量类型：`float` / `int8` / `uint8` |
| 2 | `index_prefix_path` | 磁盘索引前缀路径（不含 `_disk.index` 后缀） |
| 3 | `num_threads` | OpenMP 线程数 |
| 4 | `beamwidth` | Beam Search 宽度 |
| 5 | `query_file` | 查询向量二进制文件 |
| 6 | `truthset` | Groundtruth 文件（无则写 `null`） |
| 7 | `K` | 返回 Top-K 结果 |
| 8 | `similarity` | 距离度量：`l2` 或 `cosine` |
| 9 | `mem_L` | 内存索引 L 参数（`0` = 不使用内存索引） |
| 10 | `ivf_index_path` | 倒排索引文件路径（`ivf.index`） |
| 11 | `raw_vector_path` | 原始向量文件路径（用于 IVF 精排） |
| 12 | `query_label_file` | 查询标签表达式文件 |
| 13 | `hit_rate_threshold` | 策略切换阈值（推荐 `0.1`） |
| 14 | `ivf_topL_multiplier` | IVF 粗排扩展倍数（推荐 `3`） |
| 15+ | `L1 [L2] ...` | 搜索列表大小（可多个，需 ≥ K） |

## query_label.txt 格式

每行对应一个查询向量，支持布尔表达式：

```
# 单个 tag（仅匹配该 tag 的向量）
42

# AND：两个 tag 同时满足
1 & 2

# OR：任意 tag 满足
3 | 5

# 差集
10 - 3

# 括号改变优先级
(1 & 2) | 3
```

**注意**：倒排索引内置了 Universal Bitmap（通用集合），满足通用条件的向量会自动并入所有查询结果（在 `InvertedIndexSearcher::query_bitmap` 内部完成，对调用方透明）。

## 输出格式

```
   L   Beamwidth         QPS    AvgLat(us)     P99.9 Lat     IVF/Graph  IVFcand(avg/max) Graphcand(avg/max)    Recall@K
```

- `IVF/Graph`：使用 IVF 路径的查询数 / 使用图搜索路径的查询数
- `IVFcand(avg/max)`：IVF 路径中候选集的平均大小 / 最大大小
- `Graphcand(avg/max)`：图搜索路径中候选集的平均大小 / 最大大小

## 新增/修改的文件

| 文件 | 类型 | 说明 |
|------|------|------|
| `tests/InvertedIndex.hpp` | 新建 | 倒排索引头文件（从 `ref_codes/ivf/` 移植） |
| `tests/InvertedIndex.cpp` | 新建 | 倒排索引实现（Builder + Searcher） |
| `tests/hybrid_tag_search.cpp` | 新建 | 混合搜索主程序 |
| `tests/CMakeLists.txt` | 修改 | 添加 `hybrid_tag_search` 编译目标，自动定位 CRoaring |

## 核心算法说明

### IVF 路径（低 hit_rate）

适用于候选集较小（< threshold × 数据集总量）的情况，三阶段处理：

**阶段一：PQ 批量粗排（`compute_pq_dists_batch`）**

对标签匹配的所有候选向量计算 PQ 近似距离：

1. 调用 `SSDIndex::pq_table.populate_chunk_distances(query, pq_dists)` 生成 `256 × n_chunks` 距离查找表
2. 按 256 个候选一组分块处理：
   - **Gather 阶段**：将 PQ codes 复制到线程本地缓冲（提升 L1 locality），同时用 `_mm_prefetch(..., _MM_HINT_T0)` 预取后续 32 条
   - **Lookup 阶段**：4-chunk 循环展开查表累加距离（利用指令级并行）
3. 候选数 > 10000 时启用 `#pragma omp parallel if(n > 10000) schedule(dynamic, 1)` 块级并行；在查询级并行的外层 OMP 循环中，嵌套并行自动降级为单线程，行为安全

**阶段二：粗排选 top-L**

`std::nth_element`（O(n)）找出 top-L，再对 top-L 排序。`top_L = K × ivf_topL_multiplier`。

**阶段三：mmap 精排**

对 top-L 调用 AVX2 指令集（`L2_AVX2_Float` / `L2_AVX2_Uint8`）计算精确距离，选出 top-K。

---

### 图搜索路径（高 hit_rate）

适用于候选集较大（≥ threshold × 数据集总量）的情况，采用**边搜索边过滤**策略，与 `cached_beam_search_with_multi_entry` 设计一致：

**两个独立数据结构分工：**

| 数据结构 | 类型 | 作用 | tag 过滤 |
|----------|------|------|----------|
| search_ctx（图遍历上下文） | min-heap（内置于 `do_beam_search`） | 引导图遍历，扩展最近邻 | **无**，遍历全部节点以保证搜索质量 |
| result_heap | max-heap，容量 k | 收集结果 | **有**，只收录 tag 匹配的节点 |

**执行流程：**

1. 调用 `SSDIndex::do_beam_search`，以全图节点（不做 tag 过滤）引导遍历，返回 `expanded_nodes_info`（所有已访问节点，按距离**升序**排列）
2. 遍历 `expanded_nodes_info`，对每个节点执行 result_heap 更新：
   - 若该节点的 tag 在 `valid_tag_set` 中：
     - heap 未满（< k）→ 直接入堆
     - heap 已满且当前节点更近 → 弹出最远节点，插入当前节点
3. **早停优化**：`expanded_nodes_info` 按距离升序，当 result_heap 满且当前节点距离 ≥ heap 顶（当前最差结果）时，后续所有节点必然更远，立即 break
4. 将 result_heap 中的结果按距离升序输出

**与后置过滤的区别：**

后置过滤（旧实现）从 `expanded_nodes_info` 顺序取前 k 个 tag 匹配节点，没有 heap 的动态剪枝。result_heap 方案维护精确的 k-best 候选集，当 heap 满后每次插入都保证结果集的最优性，并可提前终止——语义上完全等价于参考实现的 inline 过滤，同时复用了 PipeANN 已有的 `do_beam_search` 接口，无需修改核心文件。

---

### 假设说明

IVF 路径中 `index->data[id × n_chunks]` 直接以候选集 ID 索引 PQ codes，假设 **vector ID == internal node ID**（即 `NO_MAPPING` 或标签文件按行号顺序对应磁盘索引节点）。若存在 ID→位置的重映射，需在调用前将 tag 转换为 internal ID。
