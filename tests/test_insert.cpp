#ifndef PIPEANN_INSERT_VARIANT_NAME
#define PIPEANN_INSERT_VARIANT_NAME "test_insert"
#endif

#include "ssd_index.h"
#include "v2/dynamic_index.h"
#include <index.h>
#include <omp.h>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <atomic>
#include <limits>
#include <algorithm>
#include <random>
#include <sstream>
#include <sys/stat.h>
#include <unistd.h>
#include <stdexcept>
#include <iomanip>
#include <cstdlib>
#include "utils.h"         // Log macros
#include "timer.h"         // pipeann::Timer

// ================= 全局配置 =================
int NUM_INSERT_THREADS = 10;
pipeann::Timer globalTimer;
int begin_time = 0;

// ================= 工具函数 =================

// 显示内存占用
void ShowMemoryStatus(const std::string &index_prefix) {
    int current_time = globalTimer.elapsed() / 1.0e6f - begin_time;
    int tSize = 0, resident = 0, share = 0;
    std::ifstream buffer("/proc/self/statm");
    if (buffer.is_open()) {
        buffer >> tSize >> resident >> share;
        buffer.close();
    }
    long page_size_kb = sysconf(_SC_PAGE_SIZE) / 1024;
    double rss = resident * page_size_kb; 

    struct stat st;
    memset(&st, 0, sizeof(struct stat));
    std::string disk_file = index_prefix + "_disk.index";
    stat(disk_file.c_str(), &st);

    LOG(INFO) << "[MemStatus] Time: " << current_time << "s | RSS: " << rss / 1024.0 << " MB | " 
              << "DiskIndex File: " << (st.st_size / (1 << 20)) << " MB";
}


static std::string format_bytes(double bytes) {
    const char *units[] = {"B", "KB", "MB", "GB", "TB"};
    int u = 0;
    while (bytes >= 1024.0 && u < 4) {
        bytes /= 1024.0;
        ++u;
    }
    std::ostringstream os;
    os << std::fixed << std::setprecision(2) << bytes << " " << units[u];
    return os.str();
}

// ================= 插入内核 =================
/*data_load: 要插入的数据指针
  sync_index: 动态SSD索引实例的引用
  start_tag: 开始标签值
  npts: 要插入的点数量
  dim: 向量维度
  total_done: 原子计数器，记录总完成的插入数量
  total_target: 总目标插入数量*/
template<typename T, typename TagT>
void insertion_kernel(T *data_load, pipeann::DynamicSSDIndex<T, TagT> &sync_index, TagT start_tag, size_t npts, size_t dim, std::atomic_size_t &total_done, size_t total_target) {
    pipeann::Timer timer;
    std::atomic_size_t done(0);// 计数器
    const size_t log_every = std::max<size_t>(1000, npts / 20);
    
    // OMP 并行插入
    #pragma omp parallel for num_threads(NUM_INSERT_THREADS)
    for (_s64 i = 0; i < (_s64) npts; i++) {
        sync_index.insert(data_load + dim * i, static_cast<TagT>(start_tag + (TagT)i));
        size_t chunk_cur = done.fetch_add(1) + 1;
        size_t total_cur = total_done.fetch_add(1) + 1;
        if ((chunk_cur % log_every) == 0 || chunk_cur == npts) {
            double elapsed = timer.elapsed() / 1.0e6f;
            double pct = (double) chunk_cur * 100.0 / (double) npts;
            double total_pct = total_target ? ((double) total_cur * 100.0 / (double) total_target) : 0.0;
            double tps = (elapsed > 0.0) ? (chunk_cur / elapsed) : 0.0;
            LOG(INFO) << "  -> Insert progress: chunk " << chunk_cur << "/" << npts << " (" << pct << "%), "
                      << "total " << total_cur << "/" << total_target << " (" << total_pct << "%), "
                      << "elapsed " << elapsed << "s, avg " << tps << " TPS";
        }
    }

    float time_secs = timer.elapsed() / 1.0e6f;
    LOG(INFO) << "  -> Inserted " << npts << " pts. Time: " << time_secs << "s. Speed: " << (npts/time_secs) << " TPS";
}

// ================= 数据读取 (独立文件读取 + 全局ID映射) =================
template<typename T, typename TagT = uint32_t>
void get_trace(std::string insert_data_bin, uint64_t file_read_offset, uint64_t global_start_tag, uint64_t n, 
               std::vector<TagT> &insert_tags, std::vector<T> &data_load) {
    
    // 1. 生成全局 Tags (Tag Remapping)
    insert_tags.reserve(n);
    for (uint64_t i = 0; i < n; ++i) {
        insert_tags.push_back((TagT)(global_start_tag + i));
    }

    // 2. 读取数据
    std::ifstream reader(insert_data_bin, std::ios::binary | std::ios::ate);
    if (!reader.is_open()) {
        throw std::runtime_error("Error: Cannot open insert data file: " + insert_data_bin);
    }
    
    // 读取头部信息
    reader.seekg(0, reader.beg);
    int npts_i32, dim_i32;// 文件头部的点数和维度
    reader.read((char *) &npts_i32, sizeof(int));
    reader.read((char *) &dim_i32, sizeof(int));

    size_t data_dim = dim_i32;
    size_t header_size = 2 * sizeof(int);
    
    // 计算读取位置
    size_t required_offset_bytes = header_size + file_read_offset * data_dim * sizeof(T);
    size_t required_bytes = n * data_dim * sizeof(T);// 计算需要读取的总字节数（n个向量×每个向量的维度×每个维度的数据类型大小）

    data_load.resize(n * data_dim);
    reader.seekg(required_offset_bytes, reader.beg);
    if (!reader.read((char *) data_load.data(), required_bytes)) {
        throw std::runtime_error("Error: Failed to read vector data. File might be too short.");
    }
}

// ================= 主流程 =================
template<typename T>
void read_vectors_chunk(std::ifstream &reader, uint64_t file_read_offset, uint64_t n, size_t data_dim,
                        size_t header_size, std::vector<T> &data_load) {
    const size_t vec_bytes = data_dim * sizeof(T);// 单个向量占用的字节数（维度 × 每个元素的大小）
    const size_t required_offset_bytes = header_size + file_read_offset * vec_bytes;// 实际读取位置的字节偏移量（文件头大小 + 偏移向量数 × 单个向量大小）
    const size_t required_bytes = n * vec_bytes;// 需要读取的总字节数（向量数量 × 单个向量大小）

    data_load.resize(n * data_dim);
    reader.clear();
    reader.seekg(required_offset_bytes, reader.beg);
    if (!reader.read((char *) data_load.data(), required_bytes)) {
        throw std::runtime_error("Error: Failed to read vector data. File might be too short.");
    }
}

/* insert_data_bin: 二进制向量数据文件路径        L_disk: 磁盘索引参数，控制搜索范围
vecs_per_step: 每步插入的向量数量                 num_steps: 总步骤数
beam_width: 搜索宽度参数                         chunk_size: 内部处理块大小
nodes_to_cache: 缓存节点数量                     index_prefix: 索引文件前缀
dist_cmp: 距离计算对象指针 */

template<typename T, typename TagT>
void run_insert_merge(const std::string &insert_data_bin, const unsigned L_disk, int vecs_per_step, int num_steps,
                      unsigned beam_width, size_t chunk_size, unsigned nodes_to_cache,
                      const std::string &index_prefix, pipeann::Distance<T> *dist_cmp) {
    // [Check 1] validate file header
    std::ifstream reader(insert_data_bin, std::ios::binary);
    if (!reader.is_open()) {
        throw std::runtime_error("Failed to open insert data file: " + insert_data_bin);
    }
    int new_data_total_vecs = 0;
    int new_data_dim = 0;
    reader.read((char *)&new_data_total_vecs, sizeof(int));
    reader.read((char *)&new_data_dim, sizeof(int));
    if (!reader) {
        throw std::runtime_error("Failed to read header from insert data file.");
    }

    const size_t data_dim = static_cast<size_t>(new_data_dim);
    const size_t header_size = 2 * sizeof(int);

    uint64_t total_plan_insert = (uint64_t)vecs_per_step * num_steps;// 计划插入的总向量数
    if (total_plan_insert > (uint64_t)new_data_total_vecs) {
        throw std::runtime_error(
            "Check Failed: Plan to insert " + std::to_string(total_plan_insert) +
            " vecs, but file only has " + std::to_string(new_data_total_vecs));
    }
    if (chunk_size == 0) {
        chunk_size = (size_t)vecs_per_step;
    }
    chunk_size = std::min(chunk_size, (size_t)vecs_per_step);
    LOG(INFO) << "[Check Passed] Data File has " << new_data_total_vecs << " vecs. Plan to insert " << total_plan_insert;

    // Pre-reserve PQ capacity to avoid reallocation spikes during insert.
    {
        const char *extra_env = std::getenv("PIPEANN_PQ_EXTRA_POINTS");
        if (extra_env == nullptr || extra_env[0] == '\0') {
            std::string extra = std::to_string(total_plan_insert);
#ifdef _WIN32
            _putenv_s("PIPEANN_PQ_EXTRA_POINTS", extra.c_str());
#else
            setenv("PIPEANN_PQ_EXTRA_POINTS", extra.c_str(), 1);
#endif
            LOG(INFO) << "PIPEANN_PQ_EXTRA_POINTS=" << extra;
        }
    }

    // 1. init SSD index
    pipeann::Parameters paras;
    paras.Set<unsigned>("L_disk", L_disk);
    paras.Set<unsigned>("R_disk", 0);
    paras.Set<float>("alpha_disk", 1.2);
    paras.Set<unsigned>("C", 384);
    paras.Set<unsigned>("beamwidth", beam_width);
    paras.Set<unsigned>("nodes_to_cache", nodes_to_cache);
    paras.Set<unsigned>("num_threads", NUM_INSERT_THREADS);

    pipeann::Metric metric = pipeann::Metric::L2;
    pipeann::DynamicSSDIndex<T, TagT> sync_index(paras, index_prefix, index_prefix + "_merge", dist_cmp, metric, 0, false);

    begin_time = globalTimer.elapsed() / 1.0e6f;

    uint64_t base_count = sync_index._disk_index->num_points;// 当前索引中的向量数量
    LOG(INFO) << "=== Index Loaded ===";
    if (base_count == 0) {
        LOG(INFO) << "[Warning] Base index is EMPTY! New vectors will start from Tag 0.";
    } else {
        LOG(INFO) << "Current Index Size: " << base_count << " vectors.";
        LOG(INFO) << "New vectors will be appended starting from Tag: " << base_count;
    }

    const uint64_t final_tag = base_count + (uint64_t)vecs_per_step * num_steps - 1;// 计算最终标签值
    if (final_tag > std::numeric_limits<TagT>::max()) {
        throw std::runtime_error("Tag range exceeds TagT max. Reduce total inserts or use larger TagT.");
    }

    const uint64_t total_after = base_count + (uint64_t) vecs_per_step * num_steps;
    const uint64_t n_chunks = sync_index._disk_index->n_chunks;// 获取 PQ 块数量
    const uint64_t pq_bytes = total_after * n_chunks;// 计算 PQ 压缩后占用的字节数
    const uint64_t tag_bytes = total_after * sizeof(TagT);// 计算标签数组占用的字节数
    const uint64_t chunk_bytes = (uint64_t) chunk_size * data_dim * sizeof(T);// 计算块缓冲区占用的字节数
    LOG(INFO) << "[MemEstimate] Total vectors after insert: " << total_after;
    LOG(INFO) << "[MemEstimate] PQ compressed bytes (approx): " << format_bytes((double) pq_bytes)
              << " (n_chunks=" << n_chunks << ")";
    LOG(INFO) << "[MemEstimate] Tags array bytes (min): " << format_bytes((double) tag_bytes);
    LOG(INFO) << "[MemEstimate] Chunk buffer bytes: " << format_bytes((double) chunk_bytes);
    LOG(INFO) << "[MemEstimate] Actual RSS will be higher due to hash maps, page layout, and thread scratch buffers.";

    // Pre-extend disk index file to final size to avoid sparse writes / missing edges during insert.
    {
        const uint64_t nnodes_per_sector = sync_index._disk_index->nnodes_per_sector;
        const std::string persist_prefix = sync_index._disk_index_prefix_in;
        const std::string disk_index_path = persist_prefix + "_disk.index";
        if (nnodes_per_sector == 0) {
            throw std::runtime_error("nnodes_per_sector is 0; cannot pre-extend disk index file.");
        }
        const uint64_t file_size =
            SECTOR_LEN + ROUND_UP(total_after, nnodes_per_sector) / nnodes_per_sector * SECTOR_LEN;
        (void)truncate(disk_index_path.c_str(), file_size);
        LOG(INFO) << "[PreExtend] Disk index file size set to " << (file_size / (1 << 20)) << " MB";
    }

    // 2. insertion loop (chunked)
    LOG(INFO) << "=== Start Insertion Loop (" << num_steps << " steps) ===";

    std::atomic_size_t total_done(0);
    const size_t total_target = (size_t) total_plan_insert;// 总目标插入数量


    const size_t dim_for_log = data_dim;
    for (int i = 0; i < num_steps; i++) {
        std::vector<T> data_load;
        data_load.reserve((size_t)chunk_size * data_dim);// 为当前批次创建数据缓冲区并预留空间

        const uint64_t step_offset = (uint64_t)i * vecs_per_step;
        uint64_t step_remaining = (uint64_t)vecs_per_step;

        LOG(INFO) << "Batch [" << i + 1 << "/" << num_steps << "]: Read Offset " << step_offset
                  << " -> Map to Tag " << (base_count + step_offset);

        while (step_remaining > 0) {
            uint64_t chunk_n = std::min<uint64_t>(step_remaining, chunk_size);
            uint64_t file_read_offset = step_offset + ((uint64_t)vecs_per_step - step_remaining);
            uint64_t global_start_tag = base_count + file_read_offset;

            read_vectors_chunk<T>(reader, file_read_offset, chunk_n, data_dim, header_size, data_load);
            insertion_kernel<T, TagT>(data_load.data(), sync_index, (TagT)global_start_tag, chunk_n, dim_for_log, total_done, total_target);
            step_remaining -= chunk_n;
        }

        ShowMemoryStatus(index_prefix);
    }

    uint64_t total_vectors_now = base_count + (uint64_t)vecs_per_step * num_steps;

    LOG(INFO) << "=== Insertion Finished ===";
    LOG(INFO) << "[Tag Range] Base Index:  0 ~ " << (base_count > 0 ? base_count - 1 : 0);
    LOG(INFO) << "[Tag Range] New Added:   " << base_count << " ~ " << total_vectors_now - 1;
    LOG(INFO) << "[Tag Range] Final Total: 0 ~ " << total_vectors_now - 1;

    // Optional: lazy delete before final merge.
    // Control via env:
    //   PIPEANN_DELETE_TAGS_FILE=/path/to/tags.txt   (one tag per line)
    //   PIPEANN_DELETE_RATIO=0.01   (delete 1%)
    //   PIPEANN_DELETE_SCOPE=all|inserted|base
    //   PIPEANN_DELETE_SEED=12345
    {
        bool did_delete = false;

        // 1) Delete by explicit tags file (one tag per line).
        const char *tags_file_env = std::getenv("PIPEANN_DELETE_TAGS_FILE");
        if (tags_file_env && tags_file_env[0] != '\0') {
            std::ifstream tag_in(tags_file_env);
            if (!tag_in.is_open()) {
                LOG(ERROR) << "[Delete] Failed to open PIPEANN_DELETE_TAGS_FILE: " << tags_file_env;
            } else {
                uint64_t tag = 0;
                uint64_t count = 0;
                while (tag_in >> tag) {
                    sync_index.lazy_delete((TagT) tag);
                    ++count;
                }
                LOG(INFO) << "[Delete] Lazy delete from file: " << tags_file_env
                          << ", count=" << count;
                did_delete = true;
            }
        }

        // 2) If no file provided, fallback to random ratio delete.
        if (!did_delete) {
            const char *ratio_env = std::getenv("PIPEANN_DELETE_RATIO");
            if (ratio_env && ratio_env[0] != '\0') {
                double ratio = std::atof(ratio_env);
                if (ratio > 0.0) {
                    const char *scope_env = std::getenv("PIPEANN_DELETE_SCOPE");
                    std::string scope = scope_env ? std::string(scope_env) : "all";

                    uint64_t del_min = 0;
                    uint64_t del_max = total_vectors_now; // exclusive
                    if (scope == "inserted") {
                        del_min = base_count;
                    } else if (scope == "base") {
                        del_max = base_count;
                    }
                    if (del_min >= del_max) {
                        LOG(WARNING) << "[Delete] Skip: empty delete range for scope=" << scope;
                    } else {
                        uint64_t range = del_max - del_min;
                        uint64_t del_cnt = (uint64_t) (range * ratio);
                        const char *seed_env = std::getenv("PIPEANN_DELETE_SEED");
                        uint64_t seed = seed_env ? std::strtoull(seed_env, nullptr, 10) : 12345ULL;
                        std::mt19937_64 rng(seed);
                        LOG(INFO) << "[Delete] Lazy delete ratio=" << ratio
                                  << ", scope=" << scope
                                  << ", count=" << del_cnt;
                        for (uint64_t i = 0; i < del_cnt; ++i) {
                            uint64_t tag = del_min + (rng() % range);
                            sync_index.lazy_delete((TagT) tag);
                        }
                    }
                }
            }
        }
    }

    // 3. merge to disk (stream PQ/tags while keeping delete logic)
    LOG(INFO) << "=== Merging to Disk (Stream PQ/Tags) ===";
    pipeann::Timer merge_timer;

    // Ensure all background insert IO has landed on disk before merge reads it.
    {
        auto *disk = sync_index._disk_index;
        if (disk) {
            while (!disk->bg_tasks.empty()) {
                sleep(1);
            }
            while (disk->bg_tasks_inflight.load(std::memory_order_relaxed) != 0) {
                sleep(1);
            }
            disk->flush_page_cache();
        }
    }

    sync_index.final_merge_stream_pq_tags(NUM_INSERT_THREADS);

    float merge_time = merge_timer.elapsed() / 1.0e6f;
    LOG(INFO) << "Merge completed in " << merge_time << "s!";
    LOG(INFO) << ">>> All Done. New index saved to: " << index_prefix << "_merge";
}

int main(int argc, char **argv) {
    
    // 异常捕获封装
    try {
        if (argc < 7) {
            LOG(INFO) << "Usage: ./test_pure_insert_merge <DataType> <NewDataBin> <L_disk> <VecsPerStep> <Steps> <Threads> <IndexPrefix> [Beamwidth] [ChunkSize] [NodesToCache]";
            exit(-1);
        }

        int arg_no = 1;
        std::string type_str(argv[arg_no++]);
        std::string insert_data_bin(argv[arg_no++]);
        unsigned L_disk = (unsigned) atoi(argv[arg_no++]);
        int vecs_per_step = (int) std::atoi(argv[arg_no++]);
        int num_steps = (int) std::atoi(argv[arg_no++]);
        NUM_INSERT_THREADS = (int) std::atoi(argv[arg_no++]);
        std::string index_prefix(argv[arg_no++]);
        unsigned beam_width = 4;
        size_t chunk_size = (size_t)vecs_per_step;
        unsigned nodes_to_cache = 0;
        // Backward-compatible parsing:
        //  - 1 trailing arg: treat as ChunkSize
        //  - 2 trailing args: Beamwidth, ChunkSize
        //  - 3 trailing args: Beamwidth, ChunkSize, NodesToCache
        if (argc > arg_no) {
            int remaining = argc - arg_no;
            if (remaining >= 2) {
                beam_width = (unsigned) std::atoi(argv[arg_no++]);
                chunk_size = (size_t) std::atoll(argv[arg_no++]);
                remaining -= 2;
            } else {
                chunk_size = (size_t) std::atoll(argv[arg_no++]);
                remaining -= 1;
            }
            if (remaining >= 1) {
                nodes_to_cache = (unsigned) std::atoi(argv[arg_no++]);
            }
        }

        // [Validation] 参数合法性校验
        if (vecs_per_step <= 0 || num_steps <= 0 || NUM_INSERT_THREADS <= 0) {
            LOG(INFO) << "Error: vecs_per_step/steps/threads must be positive integers!";
            exit(-1);
        }
        if (L_disk <= 0) {
            LOG(INFO) << "Error: L_disk must be positive!";
            exit(-1);
        }
        if (beam_width == 0) {
            LOG(INFO) << "Warning: beamwidth=0 may hurt insert connectivity/recall.";
        }
        if (chunk_size == 0) {
            LOG(INFO) << "Error: chunk_size must be positive!";
            exit(-1);
        }

        LOG(INFO) << "--- Configuration ---";
        LOG(INFO) << "Type: " << type_str;
        LOG(INFO) << "New Data: " << insert_data_bin;
        LOG(INFO) << "Index Prefix: " << index_prefix;
        LOG(INFO) << "Threads: " << NUM_INSERT_THREADS;
        LOG(INFO) << "Beamwidth: " << beam_width;
        LOG(INFO) << "NodesToCache: " << nodes_to_cache;
        LOG(INFO) << "Total Insert: " << (long)vecs_per_step * num_steps;
        LOG(INFO) << "Chunk Size: " << chunk_size;

        if (type_str == "int8") {
            pipeann::DistanceL2Int8 dist_cmp;
            run_insert_merge<int8_t, unsigned>(insert_data_bin, L_disk, vecs_per_step, num_steps, beam_width, chunk_size,
                                               nodes_to_cache, index_prefix, &dist_cmp);
        } else if (type_str == "uint8") {
            pipeann::DistanceL2UInt8 dist_cmp;
            run_insert_merge<uint8_t, unsigned>(insert_data_bin, L_disk, vecs_per_step, num_steps, beam_width, chunk_size,
                                                nodes_to_cache, index_prefix, &dist_cmp);
        } else {
            pipeann::DistanceL2 dist_cmp;
            run_insert_merge<float, unsigned>(insert_data_bin, L_disk, vecs_per_step, num_steps, beam_width, chunk_size,
                                              nodes_to_cache, index_prefix, &dist_cmp);
        }
    } catch (const std::runtime_error &e) {
        LOG(INFO) << "Runtime Error: " << e.what();
        exit(-1);
    } catch (const std::exception &e) {
        LOG(INFO) << "Exception: " << e.what();
        exit(-1);
    } catch (...) {
        LOG(INFO) << "Unknown Error!";
        exit(-1);
    }

    return 0;
}
