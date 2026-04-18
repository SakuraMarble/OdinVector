#include "v2/dynamic_index.h"

#include <index.h>
#include <omp.h>
#include <timer.h>

#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <mutex>
#include <random>
#include <shared_mutex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "v2/lazy_wal.h"
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

std::vector<std::string> split_csv(const std::string &s) {
  std::vector<std::string> out;
  std::stringstream ss(s);
  std::string tok;
  while (std::getline(ss, tok, ',')) {
    if (!tok.empty()) {
      out.push_back(tok);
    }
  }
  return out;
}

using WalEntry = v2::LazyWalEntry;
using LazyWal = v2::LazyWal;

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

std::string get_wal_path(const std::string &index_prefix) {
  const char *env = std::getenv("PIPEANN_LAZY_WAL_PATH");
  if (env != nullptr && env[0] != '\0') return std::string(env);
  return index_prefix + ".lazywal";
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
  const uint64_t pre_extend = get_env_u64("PIPEANN_SERVICE_PRE_EXTEND", 1);
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

std::string recv_line(int fd) {
  std::string out;
  char c = '\0';
  while (true) {
    ssize_t n = ::recv(fd, &c, 1, 0);
    if (n <= 0) {
      return "";
    }
    if (c == '\n') {
      break;
    }
    if (c != '\r') {
      out.push_back(c);
    }
  }
  return out;
}

bool send_all(int fd, const std::string &s) {
  const char *p = s.data();
  size_t left = s.size();
  while (left > 0) {
    ssize_t n = ::send(fd, p, left, 0);
    if (n <= 0) {
      return false;
    }
    p += n;
    left -= (size_t) n;
  }
  return true;
}

template<typename TagT>
std::string format_topk(const std::vector<TagT> &tags, const std::vector<float> &dists) {
  std::ostringstream oss;
  oss << "topk=";
  for (size_t i = 0; i < tags.size(); ++i) {
    if (i) oss << ",";
    oss << tags[i] << ":" << dists[i];
  }
  return oss.str();
}

template<typename T, typename TagT>
class LazyWalHotService {
 public:
  LazyWalHotService(const std::string &index_prefix, const std::string &data_bin, unsigned L_disk, unsigned nthreads,
                    unsigned beamwidth, unsigned nodes_to_cache)
      : data_bin_(data_bin),
        nthreads_(nthreads),
        beamwidth_(beamwidth),
        L_disk_(L_disk),
        rng_(get_env_u64("PIPEANN_SERVICE_SEED", 12345ULL)),
        wal_(get_wal_path(index_prefix)) {
    const uint64_t expected_insert = get_env_u64("PIPEANN_SERVICE_EXPECTED_INSERT", 0);
    const uint64_t warmup_queries = get_env_u64("PIPEANN_SERVICE_WARMUP_QUERIES", 2000);
    const uint64_t warmup_k = get_env_u64("PIPEANN_SERVICE_WARMUP_K", 10);
    const uint64_t warmup_l = get_env_u64("PIPEANN_SERVICE_WARMUP_L", L_disk);

    maybe_set_pq_extra_points(expected_insert);
    h_ = read_header(data_bin_);

    pipeann::Parameters paras;
    paras.Set<unsigned>("L_disk", L_disk);
    paras.Set<unsigned>("R_disk", 0);
    paras.Set<float>("alpha_disk", 1.2);
    paras.Set<unsigned>("C", 384);
    paras.Set<unsigned>("beamwidth", beamwidth);
    paras.Set<unsigned>("nodes_to_cache", nodes_to_cache);
    paras.Set<unsigned>("num_threads", nthreads);

    pipeann::Metric metric = pipeann::Metric::L2;
    index_ = std::make_unique<pipeann::DynamicSSDIndex<T, TagT>>(
        paras, index_prefix, index_prefix + "_merge", nullptr, metric, 0, false);

    LOG(INFO) << "=== Warmup phase: index loaded ===";
    warmup_search(*index_, data_bin_, h_, rng_, warmup_queries, warmup_k, warmup_l, beamwidth_);

    const uint64_t base = index_-> _disk_index->num_points;
    next_tag_.store(base, std::memory_order_relaxed);
    base_tag_.store(base, std::memory_order_relaxed);
    maybe_pre_extend_disk_index(*index_, base + expected_insert);

    replay_wal();
  }

  std::string status() {
    std::shared_lock<std::shared_mutex> lk(state_mu_);
    const uint64_t disk_pts = index_->_disk_index ? index_->_disk_index->num_points : 0;
    std::ostringstream oss;
    oss << "OK status disk_num_points=" << disk_pts << " next_tag=" << next_tag_.load(std::memory_order_relaxed)
        << " pending_inserted=" << tag_to_row_.size() << " threads=" << nthreads_
        << " wal_path=" << wal_.path();
    return oss.str();
  }

  std::string insert_random(uint64_t n, uint32_t use_threads) {
    if (n == 0) return "ERR n must be > 0";
    uint64_t start_tag = 0;
    if (!reserve_tags(n, start_tag)) {
      return "ERR tag overflow";
    }

    std::vector<uint64_t> rows;
    {
      std::lock_guard<std::mutex> g(rng_mu_);
      rows = sample_rows_for_insert(h_.npts, (size_t) n, rng_);
    }

    std::vector<T> vecs;
    read_rows<T>(data_bin_, h_, rows, vecs);
    std::vector<TagT> tags((size_t) n);
    for (size_t i = 0; i < (size_t) n; ++i) {
      tags[i] = (TagT) (start_tag + i);
    }

    const auto wal_st = std::chrono::steady_clock::now();
    {
      std::lock_guard<std::mutex> g(wal_mu_);
      std::vector<WalEntry> entries;
      entries.reserve((size_t) n);
      for (size_t i = 0; i < (size_t) n; ++i) {
        entries.push_back(WalEntry{'I', (uint64_t) tags[i], rows[i]});
      }
      wal_.append_batch(entries);
    }
    const auto wal_ed = std::chrono::steady_clock::now();
    const double wal_ms =
        (double) std::chrono::duration_cast<std::chrono::microseconds>(wal_ed - wal_st).count() / 1.0e3;

    const auto st = std::chrono::steady_clock::now();
    {
      std::shared_lock<std::shared_mutex> lk(state_mu_);
      parallel_insert(vecs, tags, use_threads);
    }
    const auto ed = std::chrono::steady_clock::now();
    const double sec =
        (double) std::chrono::duration_cast<std::chrono::microseconds>(ed - st).count() / 1.0e6;

    {
      std::unique_lock<std::shared_mutex> lk(map_mu_);
      for (size_t i = 0; i < (size_t) n; ++i) {
        tag_to_row_[tags[i]] = rows[i];
      }
    }
    total_inserted_.fetch_add(n, std::memory_order_relaxed);

    std::ostringstream oss;
    oss << "OK insert_random n=" << n << " start_tag=" << start_tag << " time_ms=" << (sec * 1000.0)
        << " avg_us=" << (sec * 1.0e6 / (double) n)
        << " tps=" << ((sec > 0.0) ? ((double) n / sec) : 0.0)
        << " threads=" << use_threads
        << " wal_ms=" << wal_ms;
    return oss.str();
  }

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

    const auto wal_st = std::chrono::steady_clock::now();
    {
      std::lock_guard<std::mutex> g(wal_mu_);
      std::vector<WalEntry> entries;
      entries.reserve(rows.size());
      for (size_t i = 0; i < rows.size(); ++i) {
        entries.push_back(WalEntry{'I', (uint64_t) tags[i], rows[i]});
      }
      wal_.append_batch(entries);
    }
    const auto wal_ed = std::chrono::steady_clock::now();
    const double wal_ms =
        (double) std::chrono::duration_cast<std::chrono::microseconds>(wal_ed - wal_st).count() / 1.0e3;

    const auto st = std::chrono::steady_clock::now();
    {
      std::shared_lock<std::shared_mutex> lk(state_mu_);
      parallel_insert(vecs, tags, use_threads);
    }
    const auto ed = std::chrono::steady_clock::now();
    const double sec =
        (double) std::chrono::duration_cast<std::chrono::microseconds>(ed - st).count() / 1.0e6;

    {
      std::unique_lock<std::shared_mutex> lk(map_mu_);
      for (size_t i = 0; i < rows.size(); ++i) {
        tag_to_row_[tags[i]] = rows[i];
      }
    }
    total_inserted_.fetch_add(rows.size(), std::memory_order_relaxed);

    std::ostringstream oss;
    oss << "OK insert_rows n=" << rows.size() << " start_tag=" << start_tag << " time_ms=" << (sec * 1000.0)
        << " avg_us=" << (sec * 1.0e6 / (double) rows.size())
        << " tps=" << ((sec > 0.0) ? ((double) rows.size() / sec) : 0.0)
        << " threads=" << use_threads
        << " wal_ms=" << wal_ms;
    return oss.str();
  }

  std::string delete_base_random(uint64_t n) {
    if (n == 0) return "ERR n must be > 0";
    std::vector<TagT> tags;
    {
      std::lock_guard<std::mutex> g(rng_mu_);
      tags = sample_unique_in_range<TagT>(0, index_->_disk_index->num_points, (size_t) n, rng_);
    }

    const auto wal_st = std::chrono::steady_clock::now();
    {
      std::lock_guard<std::mutex> g(wal_mu_);
      std::vector<WalEntry> entries;
      entries.reserve((size_t) n);
      for (auto tg : tags) entries.push_back(WalEntry{'D', (uint64_t) tg, 0});
      wal_.append_batch(entries);
    }
    const auto wal_ed = std::chrono::steady_clock::now();
    const double wal_ms =
        (double) std::chrono::duration_cast<std::chrono::microseconds>(wal_ed - wal_st).count() / 1.0e3;

    const auto st = std::chrono::steady_clock::now();
    {
      std::shared_lock<std::shared_mutex> lk(state_mu_);
      for (auto tag : tags) {
        index_->lazy_delete(tag);
      }
    }
    const auto ed = std::chrono::steady_clock::now();
    const double sec =
        (double) std::chrono::duration_cast<std::chrono::microseconds>(ed - st).count() / 1.0e6;
    total_deleted_.fetch_add(n, std::memory_order_relaxed);

    std::ostringstream oss;
    oss << "OK delete_base_random n=" << n << " time_ms=" << (sec * 1000.0)
        << " avg_us=" << (sec * 1.0e6 / (double) n)
        << " wal_ms=" << wal_ms;
    return oss.str();
  }

  std::string delete_inserted(uint64_t n) {
    if (n == 0) return "ERR n must be > 0";
    std::vector<TagT> pick;
    {
      std::shared_lock<std::shared_mutex> lk(map_mu_);
      if (tag_to_row_.size() < n) {
        std::ostringstream oss;
        oss << "ERR requested n exceeds pending_inserted (" << tag_to_row_.size() << ")";
        return oss.str();
      }
      std::vector<TagT> keys;
      keys.reserve(tag_to_row_.size());
      for (const auto &kv : tag_to_row_) keys.push_back(kv.first);
      {
        std::lock_guard<std::mutex> g(rng_mu_);
        for (size_t i = 0; i < (size_t) n; ++i) {
          size_t idx = (size_t) (rng_() % keys.size());
          pick.push_back(keys[idx]);
          keys[idx] = keys.back();
          keys.pop_back();
        }
      }
    }

    const auto wal_st = std::chrono::steady_clock::now();
    {
      std::lock_guard<std::mutex> g(wal_mu_);
      std::vector<WalEntry> entries;
      entries.reserve((size_t) n);
      for (auto tg : pick) entries.push_back(WalEntry{'D', (uint64_t) tg, 0});
      wal_.append_batch(entries);
    }
    const auto wal_ed = std::chrono::steady_clock::now();
    const double wal_ms =
        (double) std::chrono::duration_cast<std::chrono::microseconds>(wal_ed - wal_st).count() / 1.0e3;

    const auto st = std::chrono::steady_clock::now();
    {
      std::shared_lock<std::shared_mutex> lk(state_mu_);
      for (auto tg : pick) {
        index_->lazy_delete(tg);
      }
    }
    const auto ed = std::chrono::steady_clock::now();
    const double sec =
        (double) std::chrono::duration_cast<std::chrono::microseconds>(ed - st).count() / 1.0e6;
    total_deleted_.fetch_add(n, std::memory_order_relaxed);
    {
      std::unique_lock<std::shared_mutex> lk(map_mu_);
      for (auto tg : pick) tag_to_row_.erase(tg);
    }

    std::ostringstream oss;
    oss << "OK delete_inserted n=" << n << " time_ms=" << (sec * 1000.0)
        << " avg_us=" << (sec * 1.0e6 / (double) n)
        << " wal_ms=" << wal_ms;
    return oss.str();
  }

  std::string delete_tags_cmd(const std::vector<TagT> &tags) {
    if (tags.empty()) return "ERR empty tags";
    const auto wal_st = std::chrono::steady_clock::now();
    {
      std::lock_guard<std::mutex> g(wal_mu_);
      std::vector<WalEntry> entries;
      entries.reserve(tags.size());
      for (auto tg : tags) entries.push_back(WalEntry{'D', (uint64_t) tg, 0});
      wal_.append_batch(entries);
    }
    const auto wal_ed = std::chrono::steady_clock::now();
    const double wal_ms =
        (double) std::chrono::duration_cast<std::chrono::microseconds>(wal_ed - wal_st).count() / 1.0e3;

    const auto st = std::chrono::steady_clock::now();
    {
      std::shared_lock<std::shared_mutex> lk(state_mu_);
      for (auto tg : tags) {
        index_->lazy_delete(tg);
      }
    }
    const auto ed = std::chrono::steady_clock::now();
    const double sec =
        (double) std::chrono::duration_cast<std::chrono::microseconds>(ed - st).count() / 1.0e6;

    {
      std::unique_lock<std::shared_mutex> lk(map_mu_);
      for (auto tg : tags) tag_to_row_.erase(tg);
    }
    total_deleted_.fetch_add(tags.size(), std::memory_order_relaxed);

    std::ostringstream oss;
    oss << "OK delete_tags n=" << tags.size()
        << " time_ms=" << (sec * 1000.0)
        << " avg_us=" << (sec * 1.0e6 / (double) tags.size())
        << " wal_ms=" << wal_ms;
    return oss.str();
  }

  std::string search_row(uint64_t row, uint64_t k, uint64_t l_search, uint32_t beam) {
    if (row >= h_.npts) return "ERR row out of range";
    if (k == 0) return "ERR k must be > 0";
    std::vector<uint64_t> rows{row};
    std::vector<T> qv;
    read_rows<T>(data_bin_, h_, rows, qv);

    std::vector<TagT> tags((size_t) k);
    std::vector<float> dists((size_t) k);
    const auto st = std::chrono::steady_clock::now();
    {
      std::shared_lock<std::shared_mutex> lk(state_mu_);
      index_->search(qv.data(), k, 0, l_search, beam, tags.data(), dists.data(), nullptr, true);
    }
    const auto ed = std::chrono::steady_clock::now();
    const uint64_t us =
        (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(ed - st).count();
    total_queries_.fetch_add(1, std::memory_order_relaxed);
    total_query_us_.fetch_add(us, std::memory_order_relaxed);

    std::ostringstream oss;
    oss << "OK search_row row=" << row << " latency_us=" << us << " " << format_topk(tags, dists);
    return oss.str();
  }

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

  std::string search_random(uint64_t nq, uint64_t k, uint64_t l_search, uint32_t beam) {
    if (nq == 0 || k == 0) return "ERR nq and k must be > 0";
    std::vector<uint64_t> rows;
    rows.reserve((size_t) nq);
    {
      std::lock_guard<std::mutex> g(rng_mu_);
      for (size_t i = 0; i < (size_t) nq; ++i) {
        rows.push_back(rng_() % h_.npts);
      }
    }
    std::vector<T> qv;
    read_rows<T>(data_bin_, h_, rows, qv);
    std::vector<TagT> tags((size_t) k);
    std::vector<float> dists((size_t) k);

    uint64_t lat_sum = 0;
    for (size_t i = 0; i < (size_t) nq; ++i) {
      const auto st = std::chrono::steady_clock::now();
      {
        std::shared_lock<std::shared_mutex> lk(state_mu_);
        index_->search(qv.data() + i * h_.dim, k, 0, l_search, beam, tags.data(), dists.data(), nullptr, true);
      }
      const auto ed = std::chrono::steady_clock::now();
      lat_sum += (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(ed - st).count();
    }
    total_queries_.fetch_add(nq, std::memory_order_relaxed);
    total_query_us_.fetch_add(lat_sum, std::memory_order_relaxed);
    const double avg = (double) lat_sum / (double) nq;
    std::ostringstream oss;
    oss << "OK search_random nq=" << nq << " avg_latency_us=" << avg;
    return oss.str();
  }

  std::string merge_now(uint32_t use_threads) {
    pipeann::Timer t;
    {
      std::unique_lock<std::shared_mutex> lk(state_mu_);
      auto *disk = index_->_disk_index;
      if (disk) {
        while (!disk->bg_tasks.empty()) {
          sleep(1);
        }
        while (disk->bg_tasks_inflight.load(std::memory_order_relaxed) != 0) {
          sleep(1);
        }
        disk->flush_page_cache();
      }
      index_->final_merge_stream_pq_tags(use_threads);
      const uint64_t now = index_->_disk_index ? index_->_disk_index->num_points : next_tag_.load(std::memory_order_relaxed);
      next_tag_.store(now, std::memory_order_relaxed);
      base_tag_.store(now, std::memory_order_relaxed);
    }
    {
      std::unique_lock<std::shared_mutex> lk(map_mu_);
      tag_to_row_.clear();
    }
    bool wal_ok = true;
    std::string wal_err;
    try {
      std::lock_guard<std::mutex> g(wal_mu_);
      wal_.append_merge_marker();
      wal_.clear();
    } catch (const std::exception &e) {
      wal_ok = false;
      wal_err = e.what();
      LOG(WARNING) << "WAL checkpoint/clear after merge failed: " << wal_err;
    }
    const double sec = t.elapsed() / 1.0e6f;
    std::ostringstream oss;
    oss << "OK merge time_ms=" << (sec * 1000.0) << " reloaded_num_points=" << next_tag_.load(std::memory_order_relaxed)
        << " threads=" << use_threads
        << " wal=" << (wal_ok ? "ok" : "warn");
    if (!wal_ok) oss << " wal_err=" << wal_err;
    return oss.str();
  }

  std::string stats() {
    const uint64_t ins = total_inserted_.load(std::memory_order_relaxed);
    const uint64_t del = total_deleted_.load(std::memory_order_relaxed);
    const uint64_t q = total_queries_.load(std::memory_order_relaxed);
    const uint64_t qus = total_query_us_.load(std::memory_order_relaxed);
    const uint64_t r_ins = replayed_inserts_.load(std::memory_order_relaxed);
    const uint64_t r_del = replayed_deletes_.load(std::memory_order_relaxed);
    std::ostringstream oss;
    oss << "OK stats inserted=" << ins << " deleted=" << del << " queries=" << q
        << " avg_query_us=" << ((q > 0) ? ((double) qus / (double) q) : 0.0)
        << " replayed_inserts=" << r_ins
        << " replayed_deletes=" << r_del;
    return oss.str();
  }

  bool should_shutdown() const { return shutdown_.load(std::memory_order_relaxed); }
  void set_shutdown() { shutdown_.store(true, std::memory_order_relaxed); }
  void set_listen_fd(int fd) { listen_fd_ = fd; }
  void close_listener() {
    int fd = listen_fd_.load(std::memory_order_relaxed);
    if (fd >= 0) {
      ::close(fd);
      listen_fd_.store(-1, std::memory_order_relaxed);
    }
  }

  std::string wal_info() {
    std::ostringstream oss;
    oss << "OK wal path=" << wal_.path()
        << " replayed_inserts=" << replayed_inserts_.load(std::memory_order_relaxed)
        << " replayed_deletes=" << replayed_deletes_.load(std::memory_order_relaxed);
    return oss.str();
  }

 private:
  void wait_bg_tasks() {
    auto *disk = index_ ? index_->_disk_index : nullptr;
    if (!disk) return;
    while (!disk->bg_tasks.empty()) sleep(1);
    while (disk->bg_tasks_inflight.load(std::memory_order_relaxed) != 0) sleep(1);
    disk->flush_page_cache();
  }

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
        std::vector<uint64_t> rows{e.row};
        std::vector<T> vecs;
        read_rows<T>(data_bin_, h_, rows, vecs);
        TagT tg = (TagT) e.tag;
        {
          std::shared_lock<std::shared_mutex> lk(state_mu_);
          index_->insert(vecs.data(), tg);
        }
        {
          std::unique_lock<std::shared_mutex> lk(map_mu_);
          tag_to_row_[tg] = e.row;
        }
        max_seen_tag = std::max<uint64_t>(max_seen_tag, e.tag + 1);
        ++replay_ins;
      } else if (e.op == 'D') {
        {
          std::shared_lock<std::shared_mutex> lk(state_mu_);
          index_->lazy_delete((TagT) e.tag);
        }
        {
          std::unique_lock<std::shared_mutex> lk(map_mu_);
          tag_to_row_.erase((TagT) e.tag);
        }
        max_seen_tag = std::max<uint64_t>(max_seen_tag, e.tag + 1);
        ++replay_del;
      }
    }
    wait_bg_tasks();
    next_tag_.store(std::max<uint64_t>(next_tag_.load(std::memory_order_relaxed), max_seen_tag), std::memory_order_relaxed);
    replayed_inserts_.store(replay_ins, std::memory_order_relaxed);
    replayed_deletes_.store(replay_del, std::memory_order_relaxed);
    LOG(INFO) << "[WAL] replay done. path=" << wal_.path()
              << " entries=" << all_entries.size()
              << " last_merge_idx=" << last_merge
              << " replay_inserts=" << replay_ins
              << " replay_deletes=" << replay_del
              << " next_tag=" << next_tag_.load(std::memory_order_relaxed)
              << " pending_inserted=" << tag_to_row_.size();
  }

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

 private:
  std::string data_bin_;
  DataHeader h_{};
  unsigned nthreads_ = 1;
  unsigned beamwidth_ = 4;
  unsigned L_disk_ = 100;

  std::unique_ptr<pipeann::DynamicSSDIndex<T, TagT>> index_;

  std::atomic<uint64_t> next_tag_{0};
  std::atomic<uint64_t> base_tag_{0};
  std::unordered_map<TagT, uint64_t> tag_to_row_;

  std::shared_mutex state_mu_;
  std::shared_mutex map_mu_;
  std::mutex rng_mu_;
  std::mutex wal_mu_;
  std::mt19937_64 rng_;
  LazyWal wal_;

  std::atomic<uint64_t> total_inserted_{0};
  std::atomic<uint64_t> total_deleted_{0};
  std::atomic<uint64_t> total_queries_{0};
  std::atomic<uint64_t> total_query_us_{0};
  std::atomic<uint64_t> replayed_inserts_{0};
  std::atomic<uint64_t> replayed_deletes_{0};

  std::atomic<bool> shutdown_{false};
  std::atomic<int> listen_fd_{-1};
};

template<typename T, typename TagT>
std::string execute_command(LazyWalHotService<T, TagT> &svc, const std::string &line) {
  std::stringstream ss(line);
  std::string cmd;
  ss >> cmd;
  if (cmd.empty()) return "ERR empty command";

  if (cmd == "HELP") {
    return "OK commands: HELP STATUS STATS INSERT_RANDOM n [threads] INSERT_ROWS csv_rows [threads] "
           "DELETE_BASE_RANDOM n DELETE_INSERTED n DELETE_TAGS csv_tags SEARCH_ROW row k [L] [beam] "
           "SEARCH_TAG tag k [L] [beam] SEARCH_RANDOM nq k [L] [beam] WAL_INFO MERGE [threads] QUIT SHUTDOWN";
  }
  if (cmd == "STATUS") return svc.status();
  if (cmd == "STATS") return svc.stats();
  if (cmd == "WAL_INFO") return svc.wal_info();
  if (cmd == "QUIT") return "OK bye";
  if (cmd == "SHUTDOWN") {
    svc.set_shutdown();
    svc.close_listener();
    return "OK shutting down";
  }

  if (cmd == "INSERT_RANDOM") {
    std::string n_s, t_s;
    ss >> n_s >> t_s;
    uint64_t n = 0;
    if (!parse_u64(n_s, n)) return "ERR usage: INSERT_RANDOM n [threads]";
    uint32_t th = (uint32_t) get_env_u64("PIPEANN_SERVICE_DEFAULT_THREADS", 0);
    if (th == 0) th = (uint32_t) get_env_u64("OMP_NUM_THREADS", 1);
    if (!t_s.empty()) {
      uint64_t t64 = 0;
      if (!parse_u64(t_s, t64) || t64 == 0 || t64 > std::numeric_limits<uint32_t>::max()) {
        return "ERR invalid threads";
      }
      th = (uint32_t) t64;
    }
    return svc.insert_random(n, th);
  }

  if (cmd == "INSERT_ROWS") {
    std::string csv, t_s;
    ss >> csv >> t_s;
    if (csv.empty()) return "ERR usage: INSERT_ROWS row1,row2,... [threads]";
    auto toks = split_csv(csv);
    std::vector<uint64_t> rows;
    rows.reserve(toks.size());
    for (const auto &tok : toks) {
      uint64_t r = 0;
      if (!parse_u64(tok, r)) return "ERR invalid row list";
      rows.push_back(r);
    }
    uint32_t th = (uint32_t) get_env_u64("PIPEANN_SERVICE_DEFAULT_THREADS", 0);
    if (th == 0) th = (uint32_t) get_env_u64("OMP_NUM_THREADS", 1);
    if (!t_s.empty()) {
      uint64_t t64 = 0;
      if (!parse_u64(t_s, t64) || t64 == 0 || t64 > std::numeric_limits<uint32_t>::max()) {
        return "ERR invalid threads";
      }
      th = (uint32_t) t64;
    }
    return svc.insert_rows_cmd(rows, th);
  }

  if (cmd == "DELETE_BASE_RANDOM") {
    std::string n_s;
    ss >> n_s;
    uint64_t n = 0;
    if (!parse_u64(n_s, n)) return "ERR usage: DELETE_BASE_RANDOM n";
    return svc.delete_base_random(n);
  }

  if (cmd == "DELETE_INSERTED") {
    std::string n_s;
    ss >> n_s;
    uint64_t n = 0;
    if (!parse_u64(n_s, n)) return "ERR usage: DELETE_INSERTED n";
    return svc.delete_inserted(n);
  }

  if (cmd == "DELETE_TAGS") {
    std::string csv_tags;
    ss >> csv_tags;
    if (csv_tags.empty()) return "ERR usage: DELETE_TAGS tag1,tag2,...";
    auto toks = split_csv(csv_tags);
    std::vector<TagT> tags;
    tags.reserve(toks.size());
    for (const auto &tok : toks) {
      uint64_t tg = 0;
      if (!parse_u64(tok, tg)) return "ERR invalid tag list";
      tags.push_back((TagT) tg);
    }
    return svc.delete_tags_cmd(tags);
  }

  if (cmd == "SEARCH_ROW") {
    std::string row_s, k_s, l_s, b_s;
    ss >> row_s >> k_s >> l_s >> b_s;
    uint64_t row = 0, k = 0;
    if (!parse_u64(row_s, row) || !parse_u64(k_s, k)) return "ERR usage: SEARCH_ROW row k [L] [beam]";
    uint64_t l = get_env_u64("PIPEANN_SERVICE_DEFAULT_L", 100);
    uint64_t b = get_env_u64("PIPEANN_SERVICE_DEFAULT_BEAM", 4);
    if (!l_s.empty()) {
      if (!parse_u64(l_s, l)) return "ERR invalid L";
    }
    if (!b_s.empty()) {
      if (!parse_u64(b_s, b)) return "ERR invalid beam";
    }
    return svc.search_row(row, k, l, (uint32_t) b);
  }

  if (cmd == "SEARCH_TAG") {
    std::string tg_s, k_s, l_s, b_s;
    ss >> tg_s >> k_s >> l_s >> b_s;
    uint64_t tg = 0, k = 0;
    if (!parse_u64(tg_s, tg) || !parse_u64(k_s, k)) return "ERR usage: SEARCH_TAG tag k [L] [beam]";
    uint64_t l = get_env_u64("PIPEANN_SERVICE_DEFAULT_L", 100);
    uint64_t b = get_env_u64("PIPEANN_SERVICE_DEFAULT_BEAM", 4);
    if (!l_s.empty()) {
      if (!parse_u64(l_s, l)) return "ERR invalid L";
    }
    if (!b_s.empty()) {
      if (!parse_u64(b_s, b)) return "ERR invalid beam";
    }
    return svc.search_tag((TagT) tg, k, l, (uint32_t) b);
  }

  if (cmd == "SEARCH_RANDOM") {
    std::string nq_s, k_s, l_s, b_s;
    ss >> nq_s >> k_s >> l_s >> b_s;
    uint64_t nq = 0, k = 0;
    if (!parse_u64(nq_s, nq) || !parse_u64(k_s, k)) return "ERR usage: SEARCH_RANDOM nq k [L] [beam]";
    uint64_t l = get_env_u64("PIPEANN_SERVICE_DEFAULT_L", 100);
    uint64_t b = get_env_u64("PIPEANN_SERVICE_DEFAULT_BEAM", 4);
    if (!l_s.empty()) {
      if (!parse_u64(l_s, l)) return "ERR invalid L";
    }
    if (!b_s.empty()) {
      if (!parse_u64(b_s, b)) return "ERR invalid beam";
    }
    return svc.search_random(nq, k, l, (uint32_t) b);
  }

  if (cmd == "MERGE") {
    std::string t_s;
    ss >> t_s;
    uint64_t t64 = get_env_u64("PIPEANN_SERVICE_DEFAULT_THREADS", 1);
    if (!t_s.empty()) {
      if (!parse_u64(t_s, t64) || t64 == 0 || t64 > std::numeric_limits<uint32_t>::max()) {
        return "ERR invalid threads";
      }
    }
    return svc.merge_now((uint32_t) t64);
  }

  return "ERR unknown command";
}

template<typename T, typename TagT>
void client_loop(int cfd, LazyWalHotService<T, TagT> &svc) {
  (void) send_all(cfd, "OK lazy_wal_hot_service ready. type HELP\n");
  while (true) {
    const std::string line = recv_line(cfd);
    if (line.empty()) {
      break;
    }
    std::string up = line;
    for (auto &ch : up) ch = (char) std::toupper((unsigned char) ch);
    const std::string resp = execute_command(svc, up);
    if (!send_all(cfd, resp + "\n")) {
      break;
    }
    if (up == "QUIT" || up == "SHUTDOWN" || svc.should_shutdown()) {
      break;
    }
  }
  ::close(cfd);
}

template<typename T, typename TagT>
void run_server(const std::string &index_prefix, const std::string &data_bin, unsigned L_disk, unsigned nthreads,
                unsigned beamwidth, unsigned nodes_to_cache, uint16_t port) {
  LazyWalHotService<T, TagT> svc(index_prefix, data_bin, L_disk, nthreads, beamwidth, nodes_to_cache);

  int sfd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (sfd < 0) {
    throw std::runtime_error("socket failed");
  }
  int yes = 1;
  ::setsockopt(sfd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = INADDR_ANY;
  addr.sin_port = htons(port);
  if (::bind(sfd, (sockaddr *) &addr, sizeof(addr)) != 0) {
    ::close(sfd);
    throw std::runtime_error("bind failed");
  }
  if (::listen(sfd, 128) != 0) {
    ::close(sfd);
    throw std::runtime_error("listen failed");
  }
  svc.set_listen_fd(sfd);
  LOG(INFO) << "lazy_wal_hot_service listening on 0.0.0.0:" << port;

  while (!svc.should_shutdown()) {
    sockaddr_in caddr{};
    socklen_t clen = sizeof(caddr);
    int cfd = ::accept(sfd, (sockaddr *) &caddr, &clen);
    if (cfd < 0) {
      if (errno == EINTR) continue;
      if (svc.should_shutdown()) break;
      continue;
    }
    std::thread th(client_loop<T, TagT>, cfd, std::ref(svc));
    th.detach();
  }
  ::close(sfd);
}

}  // namespace

int main(int argc, char **argv) {
  if (argc < 6) {
    std::cerr << "Usage: " << argv[0]
              << " <type(float/int8/uint8)> <index_prefix> <data_bin> <L_disk> <threads> [beamwidth] [nodes_to_cache] [port]\n";
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
  uint16_t port = 18080;
  if (argc > arg_no) {
    beamwidth = (unsigned) std::atoi(argv[arg_no++]);
  }
  if (argc > arg_no) {
    nodes_to_cache = (unsigned) std::atoi(argv[arg_no++]);
  }
  if (argc > arg_no) {
    port = (uint16_t) std::atoi(argv[arg_no++]);
  }

  setenv("PIPEANN_SERVICE_DEFAULT_THREADS", std::to_string(nthreads).c_str(), 1);
  setenv("PIPEANN_SERVICE_DEFAULT_L", std::to_string(L_disk).c_str(), 1);
  setenv("PIPEANN_SERVICE_DEFAULT_BEAM", std::to_string(beamwidth).c_str(), 1);

  try {
    if (type_str == "int8") {
      run_server<int8_t, unsigned>(index_prefix, data_bin, L_disk, nthreads, beamwidth, nodes_to_cache, port);
    } else if (type_str == "uint8") {
      run_server<uint8_t, unsigned>(index_prefix, data_bin, L_disk, nthreads, beamwidth, nodes_to_cache, port);
    } else {
      run_server<float, unsigned>(index_prefix, data_bin, L_disk, nthreads, beamwidth, nodes_to_cache, port);
    }
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << "\n";
    return -1;
  }
  return 0;
}
