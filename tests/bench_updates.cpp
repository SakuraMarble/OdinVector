#include "v2/dynamic_index.h"

#include <index.h>
#include <omp.h>
#include <timer.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>
#include <unistd.h>

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

template<typename IntT>
std::vector<IntT> sample_unique_in_range(uint64_t min_id, uint64_t max_id, size_t n, std::mt19937_64 &rng) {
  if (max_id <= min_id || (max_id - min_id) < n) {
    throw std::runtime_error("Not enough tags to sample");
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

template<typename T, typename TagT>
double time_lazy_delete(pipeann::DynamicSSDIndex<T, TagT> &index, const std::vector<TagT> &tags) {
  pipeann::Timer t;
  for (auto tag : tags) {
    index.lazy_delete(tag);
  }
  return t.elapsed() / 1.0e6f;
}

template<typename T, typename TagT>
double time_lazy_insert(pipeann::DynamicSSDIndex<T, TagT> &index, const std::vector<T> &vecs, uint32_t dim,
                        const std::vector<TagT> &tags, uint32_t nthreads) {
  pipeann::Timer t;
  const size_t n = vecs.size() / dim;
  if (n != tags.size()) {
    throw std::runtime_error("time_lazy_insert: size mismatch");
  }
  if (nthreads <= 1) {
    for (size_t i = 0; i < n; ++i) {
      index.insert(vecs.data() + i * dim, tags[i]);
    }
  } else {
#pragma omp parallel for schedule(static) num_threads(nthreads)
    for (int64_t i = 0; i < (int64_t) n; ++i) {
      index.insert(vecs.data() + (size_t) i * dim, tags[(size_t) i]);
    }
  }
  return t.elapsed() / 1.0e6f;
}

template<typename T, typename TagT>
double time_insert_merge(pipeann::DynamicSSDIndex<T, TagT> &index, const std::vector<T> &vecs, uint32_t dim,
                         const std::vector<TagT> &tags, uint32_t nthreads) {
  pipeann::Timer t;
  const size_t n = vecs.size() / dim;
  if (n != tags.size()) {
    throw std::runtime_error("time_insert_merge: size mismatch");
  }
  if (nthreads <= 1) {
    for (size_t i = 0; i < n; ++i) {
      index.insert(vecs.data() + i * dim, tags[i]);
    }
  } else {
#pragma omp parallel for schedule(static) num_threads(nthreads)
    for (int64_t i = 0; i < (int64_t) n; ++i) {
      index.insert(vecs.data() + (size_t) i * dim, tags[(size_t) i]);
    }
  }

  // Ensure background IO flushed before merge.
  auto *disk = index._disk_index;
  if (disk) {
    while (!disk->bg_tasks.empty()) {
      sleep(1);
    }
    while (disk->bg_tasks_inflight.load(std::memory_order_relaxed) != 0) {
      sleep(1);
    }
    disk->flush_page_cache();
  }

  index.final_merge_stream_pq_tags(nthreads);
  return t.elapsed() / 1.0e6f;
}

void maybe_set_pq_extra_points(uint64_t extra_points) {
  if (extra_points == 0) {
    return;
  }
  const char *extra_env = std::getenv("PIPEANN_PQ_EXTRA_POINTS");
  if (extra_env != nullptr && extra_env[0] != '\0') {
    LOG(INFO) << "PIPEANN_PQ_EXTRA_POINTS already set: " << extra_env;
    return;
  }
  const std::string extra = std::to_string(extra_points);
#ifdef _WIN32
  _putenv_s("PIPEANN_PQ_EXTRA_POINTS", extra.c_str());
#else
  setenv("PIPEANN_PQ_EXTRA_POINTS", extra.c_str(), 1);
#endif
  LOG(INFO) << "PIPEANN_PQ_EXTRA_POINTS=" << extra;
}

template<typename T, typename TagT>
void maybe_pre_extend_disk_index(pipeann::DynamicSSDIndex<T, TagT> &index, uint64_t total_after_estimate) {
  const uint64_t pre_extend = get_env_u64("PIPEANN_BENCH_PRE_EXTEND", 1);
  if (pre_extend == 0) {
    LOG(INFO) << "Pre-extend disabled by PIPEANN_BENCH_PRE_EXTEND=0";
    return;
  }
  auto *disk = index._disk_index;
  if (disk == nullptr) {
    LOG(WARNING) << "Skip pre-extend: disk index is null.";
    return;
  }
  if (disk->nnodes_per_sector == 0) {
    LOG(WARNING) << "Skip pre-extend: nnodes_per_sector is 0.";
    return;
  }
  const uint64_t nsectors = (total_after_estimate + disk->nnodes_per_sector - 1) / disk->nnodes_per_sector;
  const uint64_t file_size = SECTOR_LEN + nsectors * SECTOR_LEN;
  const std::string disk_index_path = index._disk_index_prefix_in + "_disk.index";
  if (truncate(disk_index_path.c_str(), (off_t) file_size) != 0) {
    LOG(WARNING) << "[PreExtend] Failed for " << disk_index_path << ": " << std::strerror(errno);
    return;
  }
  LOG(INFO) << "[PreExtend] Disk index file size set to " << (file_size / (1ULL << 20))
            << " MB (estimate total points=" << total_after_estimate << ")";
}

template<typename T, typename TagT>
void warmup_search(pipeann::DynamicSSDIndex<T, TagT> &index, const std::string &data_bin, const DataHeader &h,
                   std::mt19937_64 &rng, unsigned L_disk, unsigned beamwidth) {
  uint64_t nq = get_env_u64("PIPEANN_BENCH_WARMUP_QUERIES", 0);
  if (nq == 0) {
    LOG(INFO) << "Warmup search disabled (PIPEANN_BENCH_WARMUP_QUERIES=0).";
    return;
  }
  if (h.npts == 0 || h.dim == 0) {
    throw std::runtime_error("Invalid data header for warmup");
  }

  const uint64_t warmup_k_u64 = get_env_u64("PIPEANN_BENCH_WARMUP_K", 10);
  const uint64_t warmup_l_u64 = get_env_u64("PIPEANN_BENCH_WARMUP_SEARCH_L", L_disk);
  // bench_updates builds DynamicSSDIndex with use_mem_index=false, so mem_L must remain 0.
  const uint64_t warmup_mem_l_u64 = get_env_u64("PIPEANN_BENCH_WARMUP_MEM_L", 0);
  if (warmup_mem_l_u64 != 0) {
    LOG(INFO) << "PIPEANN_BENCH_WARMUP_MEM_L=" << warmup_mem_l_u64
              << " requested, but use_mem_index=false in this benchmark. Forcing mem_L=0.";
  }
  const uint64_t q_in_mem = std::min<uint64_t>(nq, h.npts);

  uint64_t start = (h.npts > q_in_mem) ? (rng() % (h.npts - q_in_mem + 1)) : 0;
  std::vector<T> queries;
  read_block<T>(data_bin, h, start, q_in_mem, queries);

  std::vector<TagT> tags((size_t) warmup_k_u64);
  std::vector<float> dists((size_t) warmup_k_u64);
  pipeann::Timer t;
  for (uint64_t i = 0; i < nq; ++i) {
    const T *q = queries.data() + (i % q_in_mem) * h.dim;
    index.search(q, warmup_k_u64, 0, warmup_l_u64, beamwidth, tags.data(), dists.data(), nullptr,
                 true);
  }
  double sec = t.elapsed() / 1.0e6f;
  LOG(INFO) << "Warmup search done: queries=" << nq << ", elapsed=" << (sec * 1000.0) << " ms, qps="
            << ((sec > 0.0) ? ((double) nq / sec) : 0.0);
}

template<typename T, typename TagT>
void run_bench(const std::string &index_prefix, const std::string &data_bin, unsigned L_disk, unsigned nthreads,
               unsigned beamwidth, unsigned nodes_to_cache) {
  std::vector<size_t> counts{10, 100, 1000};
  if (const char *env = std::getenv("PIPEANN_BENCH_COUNTS")) {
    counts.clear();
    std::stringstream ss(env);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
      if (!tok.empty()) {
        counts.push_back((size_t) std::stoull(tok));
      }
    }
    if (counts.empty()) {
      counts = {10, 100, 1000};
    }
  }

  const char *seed_env = std::getenv("PIPEANN_BENCH_SEED");
  uint64_t seed = seed_env ? std::strtoull(seed_env, nullptr, 10) : 12345ULL;
  std::mt19937_64 rng(seed);
  LOG(INFO) << "Bench counts: " << counts.size() << " groups, seed: " << seed;
  LOG(INFO) << "Insert threads: " << nthreads;

  DataHeader h = read_header(data_bin);
  LOG(INFO) << "Data header: npts=" << h.npts << ", dim=" << h.dim;

  uint64_t total_insert_ops = 0;
  for (size_t n : counts) {
    total_insert_ops += 2ULL * (uint64_t) n;  // LazyInsert + Insert+Merge
  }
  maybe_set_pq_extra_points(total_insert_ops);

  pipeann::Parameters paras;
  paras.Set<unsigned>("L_disk", L_disk);
  paras.Set<unsigned>("R_disk", 0);
  paras.Set<float>("alpha_disk", 1.2);
  paras.Set<unsigned>("C", 384);
  paras.Set<unsigned>("beamwidth", beamwidth);
  paras.Set<unsigned>("nodes_to_cache", nodes_to_cache);
  paras.Set<unsigned>("num_threads", nthreads);

  pipeann::Metric metric = pipeann::Metric::L2;
  pipeann::DynamicSSDIndex<T, TagT> index(paras, index_prefix, index_prefix + "_merge", nullptr, metric, 0, false);

  // Warmup: index load + optional warmup search for page cache warming.
  LOG(INFO) << "=== Warmup phase: index loaded ===";
  warmup_search(index, data_bin, h, rng, L_disk, beamwidth);

  uint64_t next_tag = index._disk_index->num_points;
  LOG(INFO) << "Base points: " << next_tag;
  maybe_pre_extend_disk_index(index, next_tag + total_insert_ops);

  for (size_t n : counts) {
    if (n == 0) continue;
    LOG(INFO) << "=== N = " << n << " ===";

    // Lazy delete
    {
      auto tags = sample_unique_in_range<TagT>(0, index._disk_index->num_points, n, rng);
      double tsec = time_lazy_delete(index, tags);
      LOG(INFO) << "[LazyDelete] n=" << n << " time=" << (tsec * 1000.0) << " ms, avg="
                << (tsec * 1.0e6 / (double) n) << " us/op";
    }

    // Lazy insert (no merge)
    {
      auto rows = sample_rows_for_insert(h.npts, n, rng);
      std::vector<T> vecs;
      read_rows<T>(data_bin, h, rows, vecs);
      std::vector<TagT> tags(n);
      for (size_t i = 0; i < n; ++i) {
        tags[i] = (TagT) (next_tag + i);
      }
      double tsec = time_lazy_insert(index, vecs, h.dim, tags, nthreads);
      next_tag += n;
      LOG(INFO) << "[LazyInsert] n=" << n << " time=" << (tsec * 1000.0) << " ms, avg="
                << (tsec * 1.0e6 / (double) n) << " us/op";
    }

    // Insert + merge
    {
      auto rows = sample_rows_for_insert(h.npts, n, rng);
      std::vector<T> vecs;
      read_rows<T>(data_bin, h, rows, vecs);
      std::vector<TagT> tags(n);
      for (size_t i = 0; i < n; ++i) {
        tags[i] = (TagT) (next_tag + i);
      }
      double tsec = time_insert_merge(index, vecs, h.dim, tags, nthreads);
      // After merge, index reloads and cur_id equals new num_points.
      next_tag = index._disk_index->num_points;
      LOG(INFO) << "[Insert+Merge] n=" << n << " total=" << (tsec * 1000.0) << " ms, avg="
                << (tsec * 1.0e6 / (double) n) << " us/op";
    }
  }
}

}  // namespace

int main(int argc, char **argv) {
  if (argc < 6) {
    std::cerr << "Usage: " << argv[0]
              << " <type(float/int8/uint8)> <index_prefix> <data_bin> <L_disk> <threads> [beamwidth] [nodes_to_cache]\n";
    return -1;
  }

  int arg_no = 1;
  std::string type_str(argv[arg_no++]);
  std::string index_prefix(argv[arg_no++]);
  std::string data_bin(argv[arg_no++]);
  unsigned L_disk = (unsigned) std::atoi(argv[arg_no++]);
  unsigned nthreads = (unsigned) std::atoi(argv[arg_no++]);
  unsigned beamwidth = 4;
  unsigned nodes_to_cache = 0;
  if (argc > arg_no) {
    beamwidth = (unsigned) std::atoi(argv[arg_no++]);
  }
  if (argc > arg_no) {
    nodes_to_cache = (unsigned) std::atoi(argv[arg_no++]);
  }

  if (type_str == "int8") {
    run_bench<int8_t, unsigned>(index_prefix, data_bin, L_disk, nthreads, beamwidth, nodes_to_cache);
  } else if (type_str == "uint8") {
    run_bench<uint8_t, unsigned>(index_prefix, data_bin, L_disk, nthreads, beamwidth, nodes_to_cache);
  } else {
    run_bench<float, unsigned>(index_prefix, data_bin, L_disk, nthreads, beamwidth, nodes_to_cache);
  }
  return 0;
}
