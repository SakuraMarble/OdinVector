#include "v2/dynamic_index.h"

#include <index.h>
#include <omp.h>
#include <timer.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <unistd.h>

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
  if (s.empty()) return false;
  char *end = nullptr;
  errno = 0;
  unsigned long long x = std::strtoull(s.c_str(), &end, 10);
  if (errno != 0 || end == nullptr || *end != '\0') return false;
  v = (uint64_t) x;
  return true;
}

std::vector<size_t> parse_csv_counts(const std::string &csv) {
  std::vector<size_t> out;
  std::stringstream ss(csv);
  std::string tok;
  while (std::getline(ss, tok, ',')) {
    if (tok.empty()) continue;
    uint64_t v = 0;
    if (!parse_u64(tok, v)) {
      throw std::runtime_error("Invalid csv number: " + tok);
    }
    out.push_back((size_t) v);
  }
  return out;
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
    if (used.insert(t).second) ids.push_back((IntT) t);
  }
  return ids;
}

std::vector<uint64_t> sample_rows_for_insert(uint64_t npts, size_t n, std::mt19937_64 &rng) {
  if (npts == 0) throw std::runtime_error("Empty data bin");
  if (npts >= n) return sample_unique_in_range<uint64_t>(0, npts, n, rng);
  std::vector<uint64_t> rows;
  rows.reserve(n);
  for (size_t i = 0; i < n; ++i) rows.push_back(rng() % npts);
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
    if (rows[i] >= h.npts) throw std::runtime_error("row id out of range");
    reader.seekg((std::streamoff) header_bytes + (std::streamoff) rows[i] * (std::streamoff) vec_bytes, std::ios::beg);
    reader.read((char *) (out.data() + i * h.dim), (std::streamsize) vec_bytes);
    if (!reader.good()) throw std::runtime_error("Failed to read row from data bin");
  }
}

template<typename T>
void read_one_row(const std::string &bin, const DataHeader &h, uint64_t row, std::vector<T> &out) {
  std::vector<uint64_t> rows{row};
  read_rows<T>(bin, h, rows, out);
}

template<typename T, typename TagT>
void maybe_pre_extend_disk_index(pipeann::DynamicSSDIndex<T, TagT> &index, uint64_t total_after_estimate) {
  const uint64_t pre_extend = get_env_u64("PIPEANN_WAL_PRE_EXTEND", 1);
  if (pre_extend == 0 || total_after_estimate == 0) return;
  auto *disk = index._disk_index;
  if (disk == nullptr || disk->nnodes_per_sector == 0) return;
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

void maybe_set_pq_extra_points(uint64_t extra_points) {
  if (extra_points == 0) return;
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

using WalEntry = v2::LazyOpWalEntry;
using LazyOpWal = v2::LazyOpWal;

std::string resolve_wal_path_for_prefix(const std::string &index_prefix) {
  const char *env = std::getenv("PIPEANN_LAZY_WAL_PATH");
  if (env && env[0] != '\0') return std::string(env);
  return index_prefix + ".lazywal";
}

struct ReplayState {
  std::unordered_map<uint64_t, uint64_t> live_tag_to_row;    // pending inserts not deleted
  std::unordered_map<uint64_t, uint64_t> all_insert_tag_to_row;
  std::unordered_set<uint64_t> deleted_tags;
  uint64_t max_tag = 0;
  uint64_t inserts = 0;
  uint64_t deletes = 0;

  void apply_insert(uint64_t tag, uint64_t row) {
    live_tag_to_row[tag] = row;
    all_insert_tag_to_row[tag] = row;
    deleted_tags.erase(tag);
    max_tag = std::max(max_tag, tag);
    ++inserts;
  }
  void apply_delete(uint64_t tag) {
    live_tag_to_row.erase(tag);
    deleted_tags.insert(tag);
    max_tag = std::max(max_tag, tag);
    ++deletes;
  }
};

template<typename T, typename TagT>
class LazyWalRecoveryTest {
 public:
  LazyWalRecoveryTest(const std::string &index_prefix, const std::string &data_bin, unsigned L_disk, unsigned nthreads,
                      unsigned beamwidth, unsigned nodes_to_cache)
      : index_prefix_(index_prefix),
        data_bin_(data_bin),
        L_disk_(L_disk),
        nthreads_(nthreads),
        beamwidth_(beamwidth),
        rng_(get_env_u64("PIPEANN_WAL_SEED", 12345ULL)),
        wal_(resolve_wal_path_for_prefix(index_prefix)) {
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

    next_tag_ = index_->_disk_index ? index_->_disk_index->num_points : 0;
    maybe_pre_extend_disk_index(*index_, next_tag_ + get_env_u64("PIPEANN_WAL_EXPECTED_INSERT", 0));
  }

  void clear_wal() const {
    wal_.clear();
    std::cout << "OK wal_cleared path=" << wal_.path() << "\n";
  }

  void apply_no_merge(const std::vector<size_t> &insert_counts, const std::vector<size_t> &delete_counts,
                      uint32_t op_threads) {
    ReplayState state;
    replay_wal(state, true, op_threads);
    next_tag_ = std::max<uint64_t>(next_tag_, state.max_tag + 1);
    LOG(INFO) << "[Apply] replay done, wal_path=" << wal_.path() << ", replay_inserts=" << state.inserts
              << ", replay_deletes=" << state.deletes << ", next_tag=" << next_tag_;

    std::vector<uint64_t> live_inserted_tags;
    live_inserted_tags.reserve(1024);

    for (size_t n : insert_counts) {
      if (n == 0) continue;
      if (next_tag_ + n > (uint64_t) std::numeric_limits<TagT>::max()) {
        throw std::runtime_error("Tag overflow in apply_no_merge");
      }
      auto rows = sample_rows_for_insert(h_.npts, n, rng_);
      std::vector<TagT> tags(n);
      for (size_t i = 0; i < n; ++i) tags[i] = (TagT) (next_tag_ + i);

      auto wal_t0 = std::chrono::steady_clock::now();
      for (size_t i = 0; i < n; ++i) {
        wal_.append_insert((uint64_t) tags[i], rows[i]);
      }
      auto wal_t1 = std::chrono::steady_clock::now();
      double wal_ms = (double) std::chrono::duration_cast<std::chrono::microseconds>(wal_t1 - wal_t0).count() / 1.0e3;

      std::vector<T> vecs;
      read_rows<T>(data_bin_, h_, rows, vecs);
      pipeann::Timer t;
      parallel_insert(vecs, tags, op_threads);
      double sec = t.elapsed() / 1.0e6f;
      next_tag_ += n;

      for (size_t i = 0; i < n; ++i) {
        state.apply_insert((uint64_t) tags[i], rows[i]);
        live_inserted_tags.push_back((uint64_t) tags[i]);
      }
      LOG(INFO) << "[ApplyInsert] n=" << n << " wal_ms=" << wal_ms
                << " insert_ms=" << (sec * 1000.0) << " avg_us=" << (sec * 1.0e6 / (double) n)
                << " tps=" << ((sec > 0.0) ? ((double) n / sec) : 0.0) << " threads=" << op_threads
                << " start_tag=" << (uint64_t) tags[0];
    }

    for (size_t n_req : delete_counts) {
      if (n_req == 0 || live_inserted_tags.empty()) continue;
      size_t n = std::min<size_t>(n_req, live_inserted_tags.size());
      std::vector<TagT> del_tags;
      del_tags.reserve(n);
      for (size_t i = 0; i < n; ++i) {
        size_t idx = (size_t) (rng_() % live_inserted_tags.size());
        TagT tg = (TagT) live_inserted_tags[idx];
        del_tags.push_back(tg);
        live_inserted_tags[idx] = live_inserted_tags.back();
        live_inserted_tags.pop_back();
      }

      auto wal_t0 = std::chrono::steady_clock::now();
      for (auto tg : del_tags) {
        wal_.append_delete((uint64_t) tg);
      }
      auto wal_t1 = std::chrono::steady_clock::now();
      double wal_ms = (double) std::chrono::duration_cast<std::chrono::microseconds>(wal_t1 - wal_t0).count() / 1.0e3;

      pipeann::Timer t;
      for (auto tg : del_tags) {
        index_->lazy_delete(tg);
        state.apply_delete((uint64_t) tg);
      }
      double sec = t.elapsed() / 1.0e6f;
      LOG(INFO) << "[ApplyDeleteInserted] n=" << n << " wal_ms=" << wal_ms
                << " delete_ms=" << (sec * 1000.0) << " avg_us=" << (sec * 1.0e6 / (double) n);
    }

    wait_bg_tasks();
    std::cout << "OK apply_no_merge wal_path=" << wal_.path()
              << " pending_live=" << state.live_tag_to_row.size()
              << " total_insert_ops=" << state.inserts
              << " total_delete_ops=" << state.deletes
              << " next_tag=" << next_tag_
              << "\n";
  }

  int verify_after_restart(uint64_t k, uint64_t l_search, uint32_t beam, uint64_t max_live_check,
                           uint64_t max_deleted_check) {
    ReplayState state;
    replay_wal(state, true, nthreads_);
    wait_bg_tasks();

    std::vector<std::pair<uint64_t, uint64_t>> live_items(state.live_tag_to_row.begin(), state.live_tag_to_row.end());
    std::vector<std::pair<uint64_t, uint64_t>> deleted_insert_items;
    deleted_insert_items.reserve(state.deleted_tags.size());
    for (auto tg : state.deleted_tags) {
      auto it = state.all_insert_tag_to_row.find(tg);
      if (it != state.all_insert_tag_to_row.end()) {
        deleted_insert_items.emplace_back(tg, it->second);
      }
    }

    std::shuffle(live_items.begin(), live_items.end(), rng_);
    std::shuffle(deleted_insert_items.begin(), deleted_insert_items.end(), rng_);
    if (max_live_check > 0 && live_items.size() > max_live_check) live_items.resize((size_t) max_live_check);
    if (max_deleted_check > 0 && deleted_insert_items.size() > max_deleted_check) {
      deleted_insert_items.resize((size_t) max_deleted_check);
    }

    uint64_t live_hit = 0, live_miss = 0;
    for (const auto &kv : live_items) {
      auto found = search_contains_tag(kv.second, k, l_search, beam, (TagT) kv.first);
      if (found) ++live_hit;
      else ++live_miss;
    }

    uint64_t del_absent = 0, del_found = 0;
    for (const auto &kv : deleted_insert_items) {
      auto found = search_contains_tag(kv.second, k, l_search, beam, (TagT) kv.first);
      if (found) ++del_found;
      else ++del_absent;
    }

    std::cout << "VERIFY summary wal_path=" << wal_.path()
              << " replay_inserts=" << state.inserts
              << " replay_deletes=" << state.deletes
              << " pending_live=" << state.live_tag_to_row.size()
              << " checked_live=" << live_items.size()
              << " checked_deleted_insert=" << deleted_insert_items.size()
              << " live_hit=" << live_hit
              << " live_miss=" << live_miss
              << " deleted_absent=" << del_absent
              << " deleted_found=" << del_found
              << "\n";

    const bool pass_live = (live_miss == 0);
    const bool pass_del = (del_found == 0);
    if (pass_live && pass_del) {
      std::cout << "VERIFY_PASS\n";
      return 0;
    }
    std::cout << "VERIFY_FAIL\n";
    return 2;
  }

 private:
  void replay_wal(ReplayState &state, bool apply_ops, uint32_t op_threads) {
    auto entries = wal_.load_entries();
    if (entries.empty()) return;

    uint64_t insert_apply = 0, delete_apply = 0;
    for (const auto &e : entries) {
      if (e.type == 'I') {
        state.apply_insert(e.tag, e.row);
        if (apply_ops) {
          std::vector<T> vec;
          read_one_row<T>(data_bin_, h_, e.row, vec);
          TagT tg = (TagT) e.tag;
          std::vector<TagT> tags{tg};
          parallel_insert(vec, tags, op_threads);
          ++insert_apply;
        }
      } else if (e.type == 'D') {
        state.apply_delete(e.tag);
        if (apply_ops) {
          index_->lazy_delete((TagT) e.tag);
          ++delete_apply;
        }
      }
    }
    next_tag_ = std::max<uint64_t>(next_tag_, state.max_tag + 1);
    LOG(INFO) << "[ReplayWal] entries=" << entries.size()
              << " apply_inserts=" << insert_apply
              << " apply_deletes=" << delete_apply
              << " next_tag=" << next_tag_;
  }

  void parallel_insert(const std::vector<T> &vecs, const std::vector<TagT> &tags, uint32_t use_threads) {
    const size_t n = tags.size();
    if (n == 0) return;
    if (vecs.size() != n * h_.dim) {
      throw std::runtime_error("parallel_insert size mismatch");
    }
    if (use_threads <= 1 || n <= 1) {
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

  void wait_bg_tasks() {
    auto *disk = index_ ? index_->_disk_index : nullptr;
    if (!disk) return;
    while (!disk->bg_tasks.empty()) std::this_thread::sleep_for(std::chrono::seconds(1));
    while (disk->bg_tasks_inflight.load(std::memory_order_relaxed) != 0) {
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    disk->flush_page_cache();
  }

  bool search_contains_tag(uint64_t row, uint64_t k, uint64_t l_search, uint32_t beam, TagT tag) {
    std::vector<T> qv;
    read_one_row<T>(data_bin_, h_, row, qv);
    std::vector<TagT> tags((size_t) k);
    std::vector<float> dists((size_t) k);
    index_->search(qv.data(), k, 0, l_search, beam, tags.data(), dists.data(), nullptr, true);
    for (auto tg : tags) {
      if (tg == tag) return true;
    }
    return false;
  }

 private:
  std::string index_prefix_;
  std::string data_bin_;
  unsigned L_disk_ = 100;
  unsigned nthreads_ = 16;
  unsigned beamwidth_ = 4;
  DataHeader h_{};
  uint64_t next_tag_ = 0;
  std::mt19937_64 rng_;
  LazyOpWal wal_;
  std::unique_ptr<pipeann::DynamicSSDIndex<T, TagT>> index_;
};

}  // namespace

int main(int argc, char **argv) {
  if (argc < 9) {
    std::cerr
        << "Usage: " << argv[0]
        << " <type(float/int8/uint8)> <index_prefix> <data_bin> <L_disk> <threads> <beamwidth> <nodes_to_cache> "
           "<mode(clear|apply|verify)> [mode_args...]\n"
        << "  mode=clear\n"
        << "  mode=apply [insert_counts_csv(default=10,100,1000)] [delete_inserted_counts_csv(default=10,100)] "
           "[op_threads(default=threads)]\n"
        << "  mode=verify [k(default=20)] [L_search(default=L_disk)] [beam(default=beamwidth)] "
           "[max_live_check(default=0=all)] [max_deleted_check(default=0=all)]\n";
    return -1;
  }

  int arg_no = 1;
  std::string type_str(argv[arg_no++]);
  std::string index_prefix(argv[arg_no++]);
  std::string data_bin(argv[arg_no++]);
  unsigned L_disk = (unsigned) std::atoi(argv[arg_no++]);
  unsigned nthreads = (unsigned) std::atoi(argv[arg_no++]);
  unsigned beamwidth = (unsigned) std::atoi(argv[arg_no++]);
  unsigned nodes_to_cache = (unsigned) std::atoi(argv[arg_no++]);
  std::string mode(argv[arg_no++]);

  try {
    if (mode == "apply") {
      std::string insert_csv = (arg_no < argc) ? argv[arg_no++] : "10,100,1000";
      std::string delete_csv = (arg_no < argc) ? argv[arg_no++] : "10,100";
      uint64_t op_threads_u64 = (arg_no < argc) ? std::strtoull(argv[arg_no++], nullptr, 10) : nthreads;
      uint32_t op_threads = (uint32_t) std::max<uint64_t>(1, op_threads_u64);
      auto insert_counts = parse_csv_counts(insert_csv);
      auto delete_counts = parse_csv_counts(delete_csv);
      uint64_t expected_insert = 0;
      for (auto v : insert_counts) expected_insert += (uint64_t) v;
      maybe_set_pq_extra_points(expected_insert);
      if (type_str == "int8") {
        LazyWalRecoveryTest<int8_t, unsigned> t(index_prefix, data_bin, L_disk, nthreads, beamwidth, nodes_to_cache);
        t.apply_no_merge(insert_counts, delete_counts, op_threads);
      } else if (type_str == "uint8") {
        LazyWalRecoveryTest<uint8_t, unsigned> t(index_prefix, data_bin, L_disk, nthreads, beamwidth, nodes_to_cache);
        t.apply_no_merge(insert_counts, delete_counts, op_threads);
      } else {
        LazyWalRecoveryTest<float, unsigned> t(index_prefix, data_bin, L_disk, nthreads, beamwidth, nodes_to_cache);
        t.apply_no_merge(insert_counts, delete_counts, op_threads);
      }
      return 0;
    }

    if (mode == "clear") {
      LazyOpWal wal(resolve_wal_path_for_prefix(index_prefix));
      wal.clear();
      std::cout << "OK wal_cleared path=" << wal.path() << "\n";
      return 0;
    }

    if (mode == "verify") {
      uint64_t k = (arg_no < argc) ? std::strtoull(argv[arg_no++], nullptr, 10) : 20ULL;
      uint64_t l_search = (arg_no < argc) ? std::strtoull(argv[arg_no++], nullptr, 10) : (uint64_t) L_disk;
      uint64_t beam = (arg_no < argc) ? std::strtoull(argv[arg_no++], nullptr, 10) : (uint64_t) beamwidth;
      uint64_t max_live = (arg_no < argc) ? std::strtoull(argv[arg_no++], nullptr, 10) : 0ULL;
      uint64_t max_del = (arg_no < argc) ? std::strtoull(argv[arg_no++], nullptr, 10) : 0ULL;
      if (type_str == "int8") {
        LazyWalRecoveryTest<int8_t, unsigned> t(index_prefix, data_bin, L_disk, nthreads, beamwidth, nodes_to_cache);
        return t.verify_after_restart(k, l_search, (uint32_t) beam, max_live, max_del);
      } else if (type_str == "uint8") {
        LazyWalRecoveryTest<uint8_t, unsigned> t(index_prefix, data_bin, L_disk, nthreads, beamwidth, nodes_to_cache);
        return t.verify_after_restart(k, l_search, (uint32_t) beam, max_live, max_del);
      } else {
        LazyWalRecoveryTest<float, unsigned> t(index_prefix, data_bin, L_disk, nthreads, beamwidth, nodes_to_cache);
        return t.verify_after_restart(k, l_search, (uint32_t) beam, max_live, max_del);
      }
    }

    std::cerr << "Unknown mode: " << mode << "\n";
    return -1;
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << "\n";
    return -1;
  }
}
