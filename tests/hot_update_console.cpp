#include "v2/dynamic_index.h"

#include <index.h>
#include <omp.h>
#include <timer.h>

#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
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

bool parse_u64(const std::string &s, uint64_t &v) {
  if (s.empty()) {
    return false;
  }
  char *end = nullptr;
  errno = 0;
  const unsigned long long x = std::strtoull(s.c_str(), &end, 10);
  if (errno != 0 || end == nullptr || *end != '\0') {
    return false;
  }
  v = (uint64_t) x;
  return true;
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
  const uint64_t pre_extend = get_env_u64("PIPEANN_HOT_PRE_EXTEND", 1);
  if (pre_extend == 0 || total_after_estimate == 0) {
    return;
  }
  auto *disk = index._disk_index;
  if (disk == nullptr || disk->nnodes_per_sector == 0) {
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
double do_parallel_insert(pipeann::DynamicSSDIndex<T, TagT> &index, const std::vector<T> &vecs, uint32_t dim,
                          const std::vector<TagT> &tags, uint32_t nthreads) {
  const size_t n = vecs.size() / dim;
  if (n != tags.size()) {
    throw std::runtime_error("do_parallel_insert: size mismatch");
  }
  pipeann::Timer t;
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
double do_lazy_delete(pipeann::DynamicSSDIndex<T, TagT> &index, const std::vector<TagT> &tags) {
  pipeann::Timer t;
  for (auto tag : tags) {
    index.lazy_delete(tag);
  }
  return t.elapsed() / 1.0e6f;
}

template<typename T, typename TagT>
double do_merge(pipeann::DynamicSSDIndex<T, TagT> &index, uint32_t nthreads) {
  pipeann::Timer t;
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

void print_help() {
  std::cout << "Commands:\n"
            << "  help\n"
            << "  status\n"
            << "  insert_random <n> [threads]\n"
            << "  delete_inserted <n>\n"
            << "  delete_base_random <n>\n"
            << "  merge [threads]\n"
            << "  warmup [queries]\n"
            << "  exit\n";
}

template<typename T, typename TagT>
void run_console(const std::string &index_prefix, const std::string &data_bin, unsigned L_disk, unsigned nthreads,
                 unsigned beamwidth, unsigned nodes_to_cache) {
  const uint64_t warmup_queries = get_env_u64("PIPEANN_CONSOLE_WARMUP_QUERIES", 2000);
  const uint64_t warmup_k = get_env_u64("PIPEANN_CONSOLE_WARMUP_K", 10);
  const uint64_t warmup_l = get_env_u64("PIPEANN_CONSOLE_WARMUP_L", L_disk);
  const uint64_t expected_insert = get_env_u64("PIPEANN_CONSOLE_EXPECTED_INSERT", 0);

  const char *seed_env = std::getenv("PIPEANN_CONSOLE_SEED");
  const uint64_t seed = seed_env ? std::strtoull(seed_env, nullptr, 10) : 12345ULL;
  std::mt19937_64 rng(seed);

  maybe_set_pq_extra_points(expected_insert);

  DataHeader h = read_header(data_bin);
  LOG(INFO) << "Data header: npts=" << h.npts << ", dim=" << h.dim;

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

  LOG(INFO) << "=== Warmup phase: index loaded ===";
  warmup_search(index, data_bin, h, rng, warmup_queries, warmup_k, warmup_l, beamwidth);

  uint64_t next_tag = index._disk_index->num_points;
  LOG(INFO) << "Base points: " << next_tag;
  maybe_pre_extend_disk_index(index, next_tag + expected_insert);

  std::vector<TagT> inserted_live;
  inserted_live.reserve(1024);
  std::unordered_map<TagT, size_t> inserted_pos;
  inserted_pos.reserve(1024);

  std::cout << "Hot-update console is ready. Type 'help' for commands.\n";
  std::string line;
  while (std::cout << "hot-update> ", std::getline(std::cin, line)) {
    std::stringstream ss(line);
    std::string cmd;
    ss >> cmd;
    if (cmd.empty()) {
      continue;
    }
    if (cmd == "help") {
      print_help();
      continue;
    }
    if (cmd == "exit" || cmd == "quit") {
      break;
    }
    if (cmd == "status") {
      const uint64_t disk_pts = index._disk_index ? index._disk_index->num_points : 0;
      std::cout << "status: disk_num_points=" << disk_pts << ", next_tag=" << next_tag
                << ", pending_inserted_tags=" << inserted_live.size() << ", default_threads=" << nthreads << "\n";
      continue;
    }
    if (cmd == "insert_random") {
      std::string n_s, th_s;
      ss >> n_s >> th_s;
      uint64_t n = 0;
      if (!parse_u64(n_s, n) || n == 0) {
        std::cout << "usage: insert_random <n> [threads]\n";
        continue;
      }
      uint32_t use_threads = nthreads;
      if (!th_s.empty()) {
        uint64_t t64 = 0;
        if (!parse_u64(th_s, t64) || t64 == 0 || t64 > std::numeric_limits<uint32_t>::max()) {
          std::cout << "invalid threads\n";
          continue;
        }
        use_threads = (uint32_t) t64;
      }
      if (next_tag + n > (uint64_t) std::numeric_limits<TagT>::max()) {
        std::cout << "tag overflow, cannot insert.\n";
        continue;
      }
      auto rows = sample_rows_for_insert(h.npts, (size_t) n, rng);
      std::vector<T> vecs;
      read_rows<T>(data_bin, h, rows, vecs);
      std::vector<TagT> tags((size_t) n);
      for (size_t i = 0; i < (size_t) n; ++i) {
        tags[i] = (TagT) (next_tag + i);
      }
      double sec = do_parallel_insert(index, vecs, h.dim, tags, use_threads);
      for (auto tg : tags) {
        inserted_pos[tg] = inserted_live.size();
        inserted_live.push_back(tg);
      }
      next_tag += n;
      LOG(INFO) << "[HotInsert] n=" << n << " time=" << (sec * 1000.0) << " ms, avg="
                << (sec * 1.0e6 / (double) n) << " us/op, tps=" << ((sec > 0.0) ? (n / sec) : 0.0)
                << ", threads=" << use_threads;
      continue;
    }
    if (cmd == "delete_inserted") {
      std::string n_s;
      ss >> n_s;
      uint64_t n = 0;
      if (!parse_u64(n_s, n) || n == 0) {
        std::cout << "usage: delete_inserted <n>\n";
        continue;
      }
      if (n > inserted_live.size()) {
        std::cout << "requested n exceeds pending_inserted_tags (" << inserted_live.size() << ")\n";
        continue;
      }
      std::vector<TagT> tags;
      tags.reserve((size_t) n);
      for (size_t i = 0; i < (size_t) n; ++i) {
        size_t pick = (size_t) (rng() % inserted_live.size());
        TagT tg = inserted_live[pick];
        tags.push_back(tg);
        TagT last = inserted_live.back();
        inserted_live[pick] = last;
        inserted_live.pop_back();
        inserted_pos[last] = pick;
        inserted_pos.erase(tg);
      }
      double sec = do_lazy_delete(index, tags);
      LOG(INFO) << "[HotDeleteInserted] n=" << n << " time=" << (sec * 1000.0) << " ms, avg="
                << (sec * 1.0e6 / (double) n) << " us/op";
      continue;
    }
    if (cmd == "delete_base_random") {
      std::string n_s;
      ss >> n_s;
      uint64_t n = 0;
      if (!parse_u64(n_s, n) || n == 0) {
        std::cout << "usage: delete_base_random <n>\n";
        continue;
      }
      const uint64_t base_max = index._disk_index ? index._disk_index->num_points : 0;
      if (base_max < n) {
        std::cout << "not enough base points for unique delete sample\n";
        continue;
      }
      auto tags = sample_unique_in_range<TagT>(0, base_max, (size_t) n, rng);
      double sec = do_lazy_delete(index, tags);
      LOG(INFO) << "[HotDeleteBase] n=" << n << " time=" << (sec * 1000.0) << " ms, avg="
                << (sec * 1.0e6 / (double) n) << " us/op";
      continue;
    }
    if (cmd == "merge") {
      std::string th_s;
      ss >> th_s;
      uint32_t use_threads = nthreads;
      if (!th_s.empty()) {
        uint64_t t64 = 0;
        if (!parse_u64(th_s, t64) || t64 == 0 || t64 > std::numeric_limits<uint32_t>::max()) {
          std::cout << "invalid threads\n";
          continue;
        }
        use_threads = (uint32_t) t64;
      }
      double sec = do_merge(index, use_threads);
      next_tag = index._disk_index->num_points;
      inserted_live.clear();
      inserted_pos.clear();
      LOG(INFO) << "[HotMerge] time=" << (sec * 1000.0) << " ms, threads=" << use_threads
                << ", reloaded_num_points=" << next_tag;
      continue;
    }
    if (cmd == "warmup") {
      std::string nq_s;
      ss >> nq_s;
      uint64_t nq = warmup_queries;
      if (!nq_s.empty()) {
        if (!parse_u64(nq_s, nq)) {
          std::cout << "invalid queries\n";
          continue;
        }
      }
      warmup_search(index, data_bin, h, rng, nq, warmup_k, warmup_l, beamwidth);
      continue;
    }
    std::cout << "unknown command: " << cmd << "\n";
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

  try {
    if (type_str == "int8") {
      run_console<int8_t, unsigned>(index_prefix, data_bin, L_disk, nthreads, beamwidth, nodes_to_cache);
    } else if (type_str == "uint8") {
      run_console<uint8_t, unsigned>(index_prefix, data_bin, L_disk, nthreads, beamwidth, nodes_to_cache);
    } else {
      run_console<float, unsigned>(index_prefix, data_bin, L_disk, nthreads, beamwidth, nodes_to_cache);
    }
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << "\n";
    return -1;
  }
  return 0;
}
