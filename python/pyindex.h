#include <omp.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <filesystem>
#include <mutex>
#include <random>
#include <shared_mutex>
#include "aux_utils.h"
#include "index.h"
#include "linux_aligned_file_reader.h"
#include "partition_and_pq.h"
#include "ssd_index.h"
#include "utils.h"
#include <sys/sysinfo.h>

namespace py = pybind11;

struct IndexParams {
  py::dtype data_type = py::dtype::of<float>();
  py::dtype tag_type = py::dtype::of<uint32_t>();
  uint32_t data_dim = 0;
  pipeann::Metric metric = pipeann::Metric::L2;
  uint32_t max_nthreads = 32;
  uint32_t max_nbrs = 64;
  uint32_t sampled_nbrs_for_delete = 20;
  uint32_t build_threshold = 100000;

  IndexParams(py::dtype data_type = py::dtype::of<float>(), py::dtype tag_type = py::dtype::of<uint32_t>(),
              uint32_t data_dim = 0, pipeann::Metric metric = pipeann::Metric::L2, uint32_t max_nthreads = 32,
              uint32_t max_nbrs = 64, uint32_t sampled_nbrs_for_delete = 20, uint32_t build_threshold = 100000)
      : data_type(data_type), tag_type(tag_type), data_dim(data_dim), metric(metric), max_nthreads(max_nthreads),
        max_nbrs(max_nbrs), sampled_nbrs_for_delete(sampled_nbrs_for_delete), build_threshold(build_threshold) {
  }
};

class BasePyIndex {
 public:
  BasePyIndex() = default;
  ~BasePyIndex() = default;
};

template<class T>
class PyIndex : public BasePyIndex {
  static constexpr float kMemIndexP = 0.01;
  static constexpr int kMemIndexMaxPts = 2000000;
  using TagT = uint32_t;

 public:
  PyIndex() = delete;

  explicit PyIndex(IndexParams params) : params_(std::move(params)) {
    reader_.reset(new LinuxAlignedFileReader());
    disk_index_.reset(new pipeann::SSDIndex<T, TagT>(params_.metric, reader_, false, true));
    omp_set_num_threads(params_.max_nthreads);

    // experience values.
    disk_index_->l_index = params_.max_nbrs + 32;
    disk_index_->beam_width = 4;
    disk_index_->maxc = 750;
    disk_index_->alpha = 1.2;

    mem_index_params_.Set<unsigned>("R", 32);
    mem_index_params_.Set<unsigned>("L", 64);
    mem_index_params_.Set<unsigned>("C", 750);
    mem_index_params_.Set<float>("alpha", 1.2);
    mem_index_params_.Set<bool>("saturate_graph", 0);
    mem_index_params_.Set<unsigned>("num_threads", params_.max_nthreads);

    // rand value for updating in-memory index.
    std::random_device rd;
    gen = std::mt19937(rd());
  }

  float rand_uniform() {
    std::uniform_real_distribution<float> dist(0, 1);
    return dist(gen);
  }

  ~PyIndex() {
  }

  void load(const std::string &index_prefix) {
    auto mem_index_path = index_prefix + "_mem.index";
    mem_index_.reset(new pipeann::Index<T, TagT>(params_.metric, params_.data_dim, kMemIndexMaxPts, true, false, true));
    mem_index_->load(mem_index_path.c_str());

    auto disk_index_file = index_prefix + "_disk.index";
    if (std::filesystem::exists(disk_index_file)) {
      use_disk_index_ = true;
      disk_index_->load(index_prefix.c_str(), params_.max_nthreads, true, true);
      disk_index_->mem_index_.reset(mem_index_.get());
    } else {
      use_disk_index_ = false;
    }

    cur_index_prefix_ = index_prefix;
  }

  // Use file path to build. For easier interface, refer to PyIndexInterface.
  void build(const std::string &data_path, const std::string &index_prefix, const char *tag_file = nullptr,
             bool build_mem_index = false, uint32_t build_L = 0, uint32_t PQ_bytes = 32, uint32_t memory_use_GB = 0) {
    if (build_L == 0) {
      build_L = params_.max_nbrs + 32;
    }

    if (memory_use_GB == 0) {
      struct sysinfo info;
      sysinfo(&info);
      memory_use_GB = info.totalram / (1024 * 1024 * 1024) / 2;
      LOG(INFO) << "Memory use not specified. Using 1/2 of total memory: " << memory_use_GB << "GB";
    }
    pipeann::build_disk_index_py<T, TagT>(data_path.c_str(), index_prefix.c_str(), params_.max_nbrs, build_L,
                                          memory_use_GB, params_.max_nthreads, PQ_bytes, params_.metric, false,
                                          tag_file);  // create tag later.

    if (build_mem_index) {
      build_mem(data_path, index_prefix);
    }
  }

  void build_mem(const std::string &data_path, const std::string &index_prefix) {
    // sample rate 0.01
    std::string sample_prefix = index_prefix + "_mem_sample";
    gen_random_slice<T>(data_path, sample_prefix, kMemIndexP);

    std::string sample_data_bin = sample_prefix + "_data.bin";
    uint64_t data_num, data_dim;
    pipeann::get_bin_metadata(sample_data_bin, data_num, data_dim);

    std::string sample_id_bin = sample_prefix + "_ids.bin";
    std::ifstream reader;
    reader.open(sample_id_bin, std::ios::binary);
    reader.seekg(2 * sizeof(uint32_t), std::ios::beg);
    uint32_t tags_size = data_num * data_dim;
    std::vector<TagT> tags(data_num);
    reader.read((char *) tags.data(), tags_size * sizeof(uint32_t));
    reader.close();

    auto s = std::chrono::high_resolution_clock::now();
    mem_index_.reset(new pipeann::Index<T, TagT>(params_.metric, data_dim, kMemIndexMaxPts, true, false, true));
    mem_index_.build(sample_data_bin.c_str(), data_num, mem_index_params_, tags);
    std::chrono::duration<double> diff = std::chrono::high_resolution_clock::now() - s;

    LOG(INFO) << "Finish building memory index, indexing time: " << diff.count() << "\n";
    std::string save_path = index_prefix + "_mem.index";
    mem_index_.save(save_path.c_str());
  }

  std::tuple<py::array_t<TagT>, py::array_t<float>> search(py::array_t<T> &query, uint32_t topk, uint32_t L) {
    auto mu = std::shared_lock<std::shared_mutex>(save_mu_);

    auto *query_p = static_cast<T *>(query.request().ptr);
    auto ret_ids = py::array_t<TagT>(topk);
    auto ret_ids_p = static_cast<TagT *>(ret_ids.request().ptr);
    auto ret_dists = py::array_t<float>(topk);
    auto ret_dists_p = static_cast<float *>(ret_dists.request().ptr);
    if (use_disk_index_) {
      // TODO: lock and excluded_nodes in pipe search.
      // mem_L is a value by experience, we use 10 in PipeANN.
      disk_index_->pipe_search(query_p, topk, 10, L, ret_ids_p, ret_dists_p, 32);
    } else {
      mem_index_->search_with_tags(query_p, topk, L, ret_ids_p, ret_dists_p);
    }
    return std::make_tuple(ret_ids, ret_dists);
  }

  void transform_mem_index_to_disk_index() {
    auto mu = std::lock_guard<std::shared_mutex>(save_mu_);
    LOG(INFO) << "Transform memory index to disk index.";
    mem_index_->save_data(cur_index_prefix_ + "_mem_data.bin");
    mem_index_->save_tags(cur_index_prefix_ + "_disk.index.tags");
    // re-build
    this->build(cur_index_prefix_ + "_mem_data.bin", cur_index_prefix_,
                (cur_index_prefix_ + "_disk.index.tags").c_str(), true);
    this->load(cur_index_prefix_);  // here sets use_disk_index_ to true.
    LOG(INFO) << "Transform memory index to disk index done.";
  }

  // bulk add.
  void add(py::array_t<T> &vectors, py::array_t<T> &tags) {
    save_mu_.lock_shared();
#pragma omp parallel for schedule(dynamic)
    for (uint32_t i = 0; i < vectors.shape(0); i++) {
      auto *vector_p = static_cast<T *>(vectors.request().ptr) + i * params_.data_dim;
      auto *tag_p = static_cast<T *>(tags.request().ptr) + i;
      do_insert(vector_p, *tag_p);
    }
    save_mu_.unlock_shared();

    if (!use_disk_index_ && mem_index_->get_num_points() > params_.build_threshold) {
      transform_mem_index_to_disk_index();
    }
  }

  void insert(py::array_t<T> &point, TagT tag) {
    save_mu_.lock_shared();
    auto *point_p = static_cast<T *>(point.request().ptr);
    do_insert(point_p, tag);
    save_mu_.unlock_shared();

    if (!use_disk_index_ && mem_index_->get_num_points() > params_.build_threshold) {
      transform_mem_index_to_disk_index();
    }
  }

  void do_insert(T *point_p, TagT tag) {
    if (use_disk_index_) {
      int target_id = disk_index_->insert_in_place(point_p, tag, &this->deleted_nodes_set_);
      if (rand_uniform() <= kMemIndexP) {  // probably insert into the memory index.
        mem_index_->insert_point(point_p, mem_index_params_, target_id);
      }
    } else {
      mem_index_->insert_point(point_p, mem_index_params_, tag);
    }
  }

  void remove(TagT tag) {
    auto mu = std::shared_lock<std::shared_mutex>(save_mu_);
    if (deleted_nodes_set_.find(tag) == deleted_nodes_set_.end()) {
      deleted_nodes_set_.insert(tag);
      deleted_nodes_.push_back(tag);
    }
    mem_index_->lazy_delete(tag);
  }

  std::string to_string() const {
    return std::string("PyIndex with data type ") + params_.data_type.char_() + " and tag type " +
           params_.tag_type.char_();
  }

  bool save(const std::string &index_prefix /* requires double-version. */) {
    auto mu = std::lock_guard<std::shared_mutex>(save_mu_);

    if (cur_index_prefix_ == "" || cur_index_prefix_ == index_prefix) {
      return false;
    }

    if (this->deleted_nodes_.size() == 0) {
// directly save the memory index is OK.
#ifndef IN_PLACE_RECORD_UPDATE
      static_assert(false, "Require storing page layout.");
#endif
      if (use_disk_index_) {
        disk_index_->write_metadata_and_pq(cur_index_prefix_, cur_index_prefix_, disk_index_->num_points,
                                           disk_index_->medoids[0]);
      }
      mem_index_.save((index_prefix + "_mem.index").c_str());
    } else {
      // There are deleted vectors, go slow path.
      if (use_disk_index_) {
        disk_index_->merge_deletes(cur_index_prefix_, index_prefix, deleted_nodes_, deleted_nodes_set_,
                                   params_.max_nthreads);
        disk_index_->reload(index_prefix.c_str(), 1);  // reload the newest index.
        cur_index_prefix_ = index_prefix;
      }
      LOG(INFO) << "TODO here";
      // TODO(gh): build mem index using the vectors in the index, instead of using original vector data.
      // rebuilding the memory index is advised.
      build_mem(cur_index_prefix_, index_prefix);
      std::string save_path = index_prefix + "_mem.index";
      mem_index_.save(save_path.c_str());
    }
    return true;
  }

 private:
  bool use_disk_index_ = false;
  std::shared_ptr<AlignedFileReader> reader_;
  std::shared_mutex save_mu_;  // save mutex.
  std::string data_path_;
  std::string cur_index_prefix_;
  IndexParams params_;
  std::vector<TagT> deleted_nodes_;
  tsl::robin_set<TagT> deleted_nodes_set_;  // copy of deleted nodes.

  // if vectors are less than the threshold, use mem index instead.
  std::mt19937 gen;
  pipeann::Parameters mem_index_params_;
  std::shared_ptr<pipeann::Index<T, TagT>> mem_index_;
  std::shared_ptr<pipeann::SSDIndex<T, TagT>> disk_index_;
};

class PyIndexInterface {};