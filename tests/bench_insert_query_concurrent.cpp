#include "v2/dynamic_index.h"

#include <index.h>
#include <timer.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include "utils.h"

namespace {

struct DataHeader {
  uint64_t npts = 0;
  uint32_t dim = 0;
};

uint64_t get_env_u64(const char *name, uint64_t fallback) {
  const char *env = std::getenv(name);
  if (env == nullptr || env[0] == '\0') {
    return fallback;
  }
  return std::strtoull(env, nullptr, 10);
}

DataHeader read_header(const std::string &bin) {
  std::ifstream reader(bin, std::ios::binary);
  if (!reader.is_open()) {
    throw std::runtime_error("Failed to open data bin: " + bin);
  }
  int npts_i32 = 0, dim_i32 = 0;
  reader.read((char *) &npts_i32, sizeof(int));
  reader.read((char *) &dim_i32, sizeof(int));
  if (!reader.good() || npts_i32 <= 0 || dim_i32 <= 0) {
    throw std::runtime_error("Invalid header in data bin: " + bin);
  }
  DataHeader h;
  h.npts = (uint64_t) npts_i32;
  h.dim = (uint32_t) dim_i32;
  return h;
}

template<typename IntT>
std::vector<IntT> sample_unique_in_range(uint64_t min_id, uint64_t max_id, size_t n, std::mt19937_64 &rng) {
  if (max_id <= min_id || (max_id - min_id) < n) {
    throw std::runtime_error("Not enough ids to sample");
  }
  std::vector<IntT> ids;
  ids.reserve(n);
  std::unordered_set<uint64_t> used;
  used.reserve(n * 2 + 1);
  while (ids.size() < n) {
    uint64_t t = min_id + (rng() % (max_id - min_id));
    if (used.insert(t).second) {
      ids.push_back((IntT) t);
    }
  }
  return ids;
}

std::vector<uint64_t> sample_rows_for_insert(uint64_t npts, size_t n, std::mt19937_64 &rng) {
  if (npts == 0) {
    throw std::runtime_error("Empty data bin");
  }
  if (npts >= n) {
    return sample_unique_in_range<uint64_t>(0, npts, n, rng);
  }
  std::vector<uint64_t> rows;
  rows.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    rows.push_back(rng() % npts);
  }
  return rows;
}

template<typename T>
void read_rows(const std::string &bin, const DataHeader &h, const std::vector<uint64_t> &rows, std::vector<T> &out) {
  std::ifstream reader(bin, std::ios::binary);
  if (!reader.is_open()) {
    throw std::runtime_error("Failed to open data bin: " + bin);
  }
  const uint64_t header_bytes = 2 * sizeof(int);
  const uint64_t vec_bytes = (uint64_t) h.dim * sizeof(T);
  out.resize(rows.size() * h.dim);
  for (size_t i = 0; i < rows.size(); ++i) {
    if (rows[i] >= h.npts) {
      throw std::runtime_error("row id out of range");
    }
    reader.seekg((std::streamoff) header_bytes + (std::streamoff) rows[i] * (std::streamoff) vec_bytes, std::ios::beg);
    reader.read((char *) (out.data() + i * h.dim), (std::streamsize) vec_bytes);
    if (!reader.good()) {
      throw std::runtime_error("Failed to read row from data bin");
    }
  }
}

template<typename T>
void read_block(const std::string &bin, const DataHeader &h, uint64_t start, uint64_t n, std::vector<T> &out) {
  if (start + n > h.npts) {
    throw std::runtime_error("read_block out of range");
  }
  std::ifstream reader(bin, std::ios::binary);
  if (!reader.is_open()) {
    throw std::runtime_error("Failed to open data bin: " + bin);
  }
  const uint64_t header_bytes = 2 * sizeof(int);
  const uint64_t vec_bytes = (uint64_t) h.dim * sizeof(T);
  reader.seekg((std::streamoff) header_bytes + (std::streamoff) start * (std::streamoff) vec_bytes, std::ios::beg);
  out.resize((size_t) n * h.dim);
  reader.read((char *) out.data(), (std::streamsize) (n * vec_bytes));
  if (!reader.good()) {
    throw std::runtime_error("Failed to read data block");
  }
}

template<typename T, typename TagT>
void warmup_search(pipeann::DynamicSSDIndex<T, TagT> &index, const std::string &data_bin, const DataHeader &h,
                   std::mt19937_64 &rng, uint64_t warmup_queries, uint64_t warmup_k, uint64_t warmup_l,
                   uint32_t beamwidth) {
  if (warmup_queries == 0) {
    LOG(INFO) << "Warmup search disabled.";
    return;
  }
  uint64_t q_in_mem = std::min<uint64_t>(warmup_queries, h.npts);
  uint64_t start = (h.npts > q_in_mem) ? (rng() % (h.npts - q_in_mem + 1)) : 0;
  std::vector<T> queries;
  read_block<T>(data_bin, h, start, q_in_mem, queries);

  std::vector<TagT> tags((size_t) warmup_k);
  std::vector<float> dists((size_t) warmup_k);
  pipeann::Timer t;
  for (uint64_t i = 0; i < warmup_queries; ++i) {
    const T *q = queries.data() + (i % q_in_mem) * h.dim;
    index.search(q, warmup_k, 0, warmup_l, beamwidth, tags.data(), dists.data(), nullptr, true);
  }
  double sec = t.elapsed() / 1.0e6f;
  LOG(INFO) << "Warmup done: queries=" << warmup_queries << ", elapsed=" << (sec * 1000.0)
            << " ms, qps=" << ((sec > 0.0) ? (warmup_queries / sec) : 0.0);
}

template<typename T, typename TagT>
void run_bench(const std::string &index_prefix, const std::string &data_bin, unsigned L_disk, unsigned insert_threads,
               unsigned query_threads, unsigned beamwidth, unsigned nodes_to_cache) {
  const uint64_t insert_n = get_env_u64("PIPEANN_CONCUR_INSERT_N", 1000);
  const uint64_t query_k = get_env_u64("PIPEANN_CONCUR_QUERY_K", 10);
  const uint64_t query_l = get_env_u64("PIPEANN_CONCUR_QUERY_L", L_disk);
  const uint64_t warmup_queries = get_env_u64("PIPEANN_CONCUR_WARMUP_QUERIES", 2000);
  const uint64_t warmup_k = get_env_u64("PIPEANN_CONCUR_WARMUP_K", 10);
  const uint64_t warmup_l = get_env_u64("PIPEANN_CONCUR_WARMUP_L", L_disk);
  const uint64_t post_insert_ms = get_env_u64("PIPEANN_CONCUR_POST_INSERT_MS", 2000);
  const uint64_t report_ms = get_env_u64("PIPEANN_CONCUR_REPORT_MS", 5000);

  const char *seed_env = std::getenv("PIPEANN_CONCUR_SEED");
  uint64_t seed = seed_env ? std::strtoull(seed_env, nullptr, 10) : 12345ULL;
  std::mt19937_64 rng(seed);

  DataHeader h = read_header(data_bin);
  LOG(INFO) << "Data header: npts=" << h.npts << ", dim=" << h.dim;

  std::vector<uint64_t> rows = sample_rows_for_insert(h.npts, (size_t) insert_n, rng);
  std::vector<T> insert_vecs;
  read_rows<T>(data_bin, h, rows, insert_vecs);

  pipeann::Parameters paras;
  paras.Set<unsigned>("L_disk", L_disk);
  paras.Set<unsigned>("R_disk", 0);
  paras.Set<float>("alpha_disk", 1.2);
  paras.Set<unsigned>("C", 384);
  paras.Set<unsigned>("beamwidth", beamwidth);
  paras.Set<unsigned>("nodes_to_cache", nodes_to_cache);
  paras.Set<unsigned>("num_threads", std::max<unsigned>(8, insert_threads + query_threads + 2));

  pipeann::Metric metric = pipeann::Metric::L2;
  pipeann::DynamicSSDIndex<T, TagT> index(paras, index_prefix, index_prefix + "_merge", nullptr, metric, 0, false);

  LOG(INFO) << "=== Warmup phase: index loaded ===";
  warmup_search(index, data_bin, h, rng, warmup_queries, warmup_k, warmup_l, beamwidth);

  uint64_t base_tag = index._disk_index->num_points;
  if (base_tag + insert_n > (uint64_t) std::numeric_limits<TagT>::max()) {
    throw std::runtime_error("Tag range overflow for current TagT");
  }
  LOG(INFO) << "Base points: " << base_tag;
  LOG(INFO) << "Concurrent config: insert_n=" << insert_n << ", insert_threads=" << insert_threads
            << ", query_threads=" << query_threads << ", query_k=" << query_k << ", query_l=" << query_l;

  std::vector<std::atomic<uint8_t>> inserted_flags((size_t) insert_n);
  for (auto &x : inserted_flags) {
    x.store(0, std::memory_order_relaxed);
  }

  std::atomic<uint64_t> next_insert{0};
  std::atomic<uint64_t> inserted_cnt{0};
  std::atomic<uint64_t> query_cnt{0};
  std::atomic<uint64_t> query_lat_us_sum{0};
  std::atomic<uint64_t> query_hit_new_any{0};
  std::atomic<uint64_t> query_probe_eligible{0};
  std::atomic<uint64_t> query_probe_hit{0};
  std::atomic<bool> stop_queries{false};
  std::atomic<bool> stop_report{false};

  std::vector<std::thread> query_workers;
  query_workers.reserve(query_threads);
  for (unsigned t = 0; t < query_threads; ++t) {
    query_workers.emplace_back([&, t]() {
      std::mt19937_64 local_rng(seed + 1000003ULL * (t + 1));
      std::vector<TagT> tags((size_t) query_k);
      std::vector<float> dists((size_t) query_k);
      while (!stop_queries.load(std::memory_order_relaxed)) {
        uint64_t idx = local_rng() % insert_n;
        const T *q = insert_vecs.data() + idx * h.dim;
        auto st = std::chrono::steady_clock::now();
        index.search(q, query_k, 0, query_l, beamwidth, tags.data(), dists.data(), nullptr, true);
        auto ed = std::chrono::steady_clock::now();
        uint64_t us = (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(ed - st).count();
        query_cnt.fetch_add(1, std::memory_order_relaxed);
        query_lat_us_sum.fetch_add(us, std::memory_order_relaxed);

        bool any_new = false;
        for (size_t i = 0; i < tags.size(); ++i) {
          TagT tg = tags[i];
          if (tg >= (TagT) base_tag && tg < (TagT) (base_tag + insert_n)) {
            any_new = true;
            break;
          }
        }
        if (any_new) {
          query_hit_new_any.fetch_add(1, std::memory_order_relaxed);
        }
        if (inserted_flags[(size_t) idx].load(std::memory_order_relaxed) != 0) {
          query_probe_eligible.fetch_add(1, std::memory_order_relaxed);
          TagT expect_tag = (TagT) (base_tag + idx);
          bool hit = false;
          for (size_t i = 0; i < tags.size(); ++i) {
            if (tags[i] == expect_tag) {
              hit = true;
              break;
            }
          }
          if (hit) {
            query_probe_hit.fetch_add(1, std::memory_order_relaxed);
          }
        }
      }
    });
  }

  std::thread reporter([&]() {
    uint64_t prev_ins = 0;
    uint64_t prev_q = 0;
    while (!stop_report.load(std::memory_order_relaxed)) {
      std::this_thread::sleep_for(std::chrono::milliseconds(report_ms));
      uint64_t ins = inserted_cnt.load(std::memory_order_relaxed);
      uint64_t q = query_cnt.load(std::memory_order_relaxed);
      LOG(INFO) << "[Live] inserted=" << ins << "/" << insert_n << " (" << (ins - prev_ins) << " per interval), "
                << "queries=" << q << " (" << (q - prev_q) << " per interval)";
      prev_ins = ins;
      prev_q = q;
    }
  });

  auto insert_st = std::chrono::steady_clock::now();
  std::vector<std::thread> insert_workers;
  insert_workers.reserve(insert_threads);
  for (unsigned t = 0; t < insert_threads; ++t) {
    insert_workers.emplace_back([&, t]() {
      (void) t;
      while (true) {
        uint64_t i = next_insert.fetch_add(1, std::memory_order_relaxed);
        if (i >= insert_n) {
          break;
        }
        index.insert(insert_vecs.data() + i * h.dim, (TagT) (base_tag + i));
        inserted_flags[(size_t) i].store(1, std::memory_order_relaxed);
        inserted_cnt.fetch_add(1, std::memory_order_relaxed);
      }
    });
  }

  for (auto &th : insert_workers) {
    th.join();
  }
  auto insert_ed = std::chrono::steady_clock::now();
  double insert_sec = (double) std::chrono::duration_cast<std::chrono::microseconds>(insert_ed - insert_st).count() /
                      1.0e6;

  LOG(INFO) << "Insert workers finished. Sleeping " << post_insert_ms << " ms for concurrent query observation.";
  std::this_thread::sleep_for(std::chrono::milliseconds(post_insert_ms));

  stop_queries.store(true, std::memory_order_relaxed);
  for (auto &th : query_workers) {
    th.join();
  }
  stop_report.store(true, std::memory_order_relaxed);
  reporter.join();

  uint64_t final_inserted = inserted_cnt.load(std::memory_order_relaxed);
  uint64_t final_queries = query_cnt.load(std::memory_order_relaxed);
  uint64_t final_q_lat = query_lat_us_sum.load(std::memory_order_relaxed);
  uint64_t hit_any = query_hit_new_any.load(std::memory_order_relaxed);
  uint64_t probe_eligible = query_probe_eligible.load(std::memory_order_relaxed);
  uint64_t probe_hit = query_probe_hit.load(std::memory_order_relaxed);

  LOG(INFO) << "=== Concurrent Insert+Query Result ===";
  LOG(INFO) << "[InsertOnly] n=" << final_inserted << " time=" << (insert_sec * 1000.0) << " ms, avg="
            << ((final_inserted > 0) ? (insert_sec * 1.0e6 / (double) final_inserted) : 0.0) << " us/op, tps="
            << ((insert_sec > 0.0) ? ((double) final_inserted / insert_sec) : 0.0);
  LOG(INFO) << "[QueryConcurrent] total=" << final_queries << ", avg_lat="
            << ((final_queries > 0) ? ((double) final_q_lat / (double) final_queries) : 0.0)
            << " us, hit_any_inserted_ratio="
            << ((final_queries > 0) ? ((double) hit_any / (double) final_queries) : 0.0);
  LOG(INFO) << "[ProbeInserted] eligible=" << probe_eligible << ", hit=" << probe_hit << ", ratio="
            << ((probe_eligible > 0) ? ((double) probe_hit / (double) probe_eligible) : 0.0);
}

}  // namespace

int main(int argc, char **argv) {
  if (argc < 7) {
    std::cerr << "Usage: " << argv[0]
              << " <type(float/int8/uint8)> <index_prefix> <data_bin> <L_disk> <insert_threads> <query_threads> "
                 "[beamwidth] [nodes_to_cache]\n";
    return -1;
  }

  int arg_no = 1;
  std::string type_str(argv[arg_no++]);
  std::string index_prefix(argv[arg_no++]);
  std::string data_bin(argv[arg_no++]);
  unsigned L_disk = (unsigned) std::atoi(argv[arg_no++]);
  unsigned insert_threads = (unsigned) std::atoi(argv[arg_no++]);
  unsigned query_threads = (unsigned) std::atoi(argv[arg_no++]);
  unsigned beamwidth = 4;
  unsigned nodes_to_cache = 0;
  if (argc > arg_no) {
    beamwidth = (unsigned) std::atoi(argv[arg_no++]);
  }
  if (argc > arg_no) {
    nodes_to_cache = (unsigned) std::atoi(argv[arg_no++]);
  }

  if (type_str == "int8") {
    run_bench<int8_t, unsigned>(index_prefix, data_bin, L_disk, insert_threads, query_threads, beamwidth,
                                nodes_to_cache);
  } else if (type_str == "uint8") {
    run_bench<uint8_t, unsigned>(index_prefix, data_bin, L_disk, insert_threads, query_threads, beamwidth,
                                 nodes_to_cache);
  } else {
    run_bench<float, unsigned>(index_prefix, data_bin, L_disk, insert_threads, query_threads, beamwidth,
                               nodes_to_cache);
  }
  return 0;
}
