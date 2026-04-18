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
#include <cctype>
#include <sstream>
#include <sys/stat.h>
#include <unistd.h>
#include <stdexcept>
#include <iomanip>
#include <cstdlib>
#include "utils.h"         // Log macros
#include "timer.h"         // pipeann::Timer



template<typename T, typename TagT>
static void verify_graph_links(pipeann::SSDIndex<T, TagT> *disk, const std::string &disk_index_path,
                               uint64_t base_count, uint64_t total_after, uint64_t max_samples = 1000) {
    if (disk == nullptr) {
        LOG(WARNING) << "[Verify] Disk index is null, skip verification.";
        return;
    }
    if (base_count == 0 || total_after <= base_count) {
        LOG(INFO) << "[Verify] No new points to verify.";
        return;
    }
    std::ifstream in(disk_index_path, std::ios::binary);
    if (!in.is_open()) {
        LOG(WARNING) << "[Verify] Failed to open disk index: " << disk_index_path;
        return;
    }

    const uint64_t new_count = total_after - base_count;
    const uint64_t sample_old = std::min<uint64_t>(max_samples, base_count);
    const uint64_t sample_new = std::min<uint64_t>(max_samples, new_count);
    const uint64_t stride_old = std::max<uint64_t>(1, base_count / sample_old);
    const uint64_t stride_new = std::max<uint64_t>(1, new_count / sample_new);

    std::vector<char> sector_buf(SECTOR_LEN);
    uint64_t last_sector = std::numeric_limits<uint64_t>::max();

    auto read_sector = [&](uint64_t sector_no) -> bool {
        if (sector_no == last_sector) {
            return true;
        }
        in.seekg((std::streamoff) sector_no * SECTOR_LEN, std::ios::beg);
        if (!in.read(sector_buf.data(), SECTOR_LEN)) {
            return false;
        }
        last_sector = sector_no;
        return true;
    };

    uint64_t old_checked = 0, old_edges_to_new = 0, old_edges_total = 0;
    for (uint64_t idx = 0; idx < base_count && old_checked < sample_old; idx += stride_old) {
        uint64_t id = idx;
        uint64_t sector = disk->node_sector_no((uint32_t) id);
        if (!read_sector(sector)) {
            LOG(WARNING) << "[Verify] Read failed at sector " << sector;
            break;
        }
        char *node_buf = disk->offset_to_node(sector_buf.data(), (uint32_t) id);
        unsigned *nbrs = disk->offset_to_node_nhood(node_buf);
        unsigned nnbrs = *(nbrs - 1);
        if (nnbrs > disk->max_degree) {
            nnbrs = (unsigned) disk->max_degree;
        }
        for (unsigned i = 0; i < nnbrs; ++i) {
            uint32_t nbr = nbrs[i];
            if (nbr >= base_count && nbr < total_after) {
                old_edges_to_new++;
            }
        }
        old_edges_total += nnbrs;
        old_checked++;
    }

    uint64_t new_checked = 0, new_edges_to_old = 0, new_edges_total = 0;
    for (uint64_t idx = 0; idx < new_count && new_checked < sample_new; idx += stride_new) {
        uint64_t id = base_count + idx;
        uint64_t sector = disk->node_sector_no((uint32_t) id);
        if (!read_sector(sector)) {
            LOG(WARNING) << "[Verify] Read failed at sector " << sector;
            break;
        }
        char *node_buf = disk->offset_to_node(sector_buf.data(), (uint32_t) id);
        unsigned *nbrs = disk->offset_to_node_nhood(node_buf);
        unsigned nnbrs = *(nbrs - 1);
        if (nnbrs > disk->max_degree) {
            nnbrs = (unsigned) disk->max_degree;
        }
        for (unsigned i = 0; i < nnbrs; ++i) {
            uint32_t nbr = nbrs[i];
            if (nbr < base_count) {
                new_edges_to_old++;
            }
        }
        new_edges_total += nnbrs;
        new_checked++;
    }

    LOG(INFO) << "[Verify] Sample old nodes: " << old_checked
              << ", edges->new: " << old_edges_to_new << "/" << old_edges_total;
    LOG(INFO) << "[Verify] Sample new nodes: " << new_checked
              << ", edges->old: " << new_edges_to_old << "/" << new_edges_total;

    if (old_edges_to_new == 0) {
        LOG(WARNING) << "[Verify] No old->new edges found in sample. Graph may be disconnected.";
    }
    if (new_edges_to_old == 0) {
        LOG(WARNING) << "[Verify] No new->old edges found in sample. New nodes may be isolated.";
    }
}

// ================= ???? =================
int NUM_INSERT_THREADS = 10;
pipeann::Timer globalTimer;
int begin_time = 0;

// ================= ???? =================

// ??????
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

static std::string to_lower_copy(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return (char)std::tolower(c); });
    return s;
}

static void verify_pq_tags_headers(const std::string &prefix, uint64_t expected_npts, uint32_t expected_chunks,
                                   bool check_tags) {
    const std::string pq_path = prefix + "_pq_compressed.bin";
    std::ifstream pq_in(pq_path, std::ios::binary);
    if (!pq_in.is_open()) {
        LOG(WARNING) << "[Verify] Cannot open PQ file: " << pq_path;
    } else {
        uint32_t npts = 0, ndims = 0;
        pq_in.read((char *) &npts, sizeof(uint32_t));
        pq_in.read((char *) &ndims, sizeof(uint32_t));
        LOG(INFO) << "[Verify] PQ header: npts=" << npts << " ndims=" << ndims
                  << " (expected npts=" << expected_npts << " ndims=" << expected_chunks << ")";
        if (npts != expected_npts || ndims != expected_chunks) {
            LOG(WARNING) << "[Verify] PQ header mismatch.";
        }
    }

    if (!check_tags) {
        return;
    }
    const std::string tags_path = prefix + "_disk.index.tags";
    std::ifstream tag_in(tags_path, std::ios::binary);
    if (!tag_in.is_open()) {
        LOG(WARNING) << "[Verify] Cannot open tags file: " << tags_path;
        return;
    }
    uint32_t tnpts = 0, tndims = 0;
    tag_in.read((char *) &tnpts, sizeof(uint32_t));
    tag_in.read((char *) &tndims, sizeof(uint32_t));
    LOG(INFO) << "[Verify] Tags header: npts=" << tnpts << " ndims=" << tndims
              << " (expected npts=" << expected_npts << " ndims=1)";
    if (tnpts != expected_npts || tndims != 1) {
        LOG(WARNING) << "[Verify] Tags header mismatch.";
    }
}

template<typename T>
static bool append_bin_payload(const std::string &path, const T *payload, uint64_t base_npts, uint64_t add_npts,
                               uint64_t ndims) {
    if (add_npts == 0) {
        return true;
    }
    std::fstream io(path, std::ios::binary | std::ios::in | std::ios::out);
    if (!io.is_open()) {
        return false;
    }
    uint32_t npts_u32 = 0;
    uint32_t ndims_u32 = 0;
    io.read((char *)&npts_u32, sizeof(uint32_t));
    io.read((char *)&ndims_u32, sizeof(uint32_t));
    if (!io.good()) {
        return false;
    }
    if (ndims_u32 != (uint32_t)ndims || npts_u32 != (uint32_t)base_npts) {
        LOG(WARNING) << "Bin header mismatch for " << path << ": file(npts=" << npts_u32 << ", ndims=" << ndims_u32
                     << ") vs expected(npts=" << base_npts << ", ndims=" << ndims << ")";
        return false;
    }
    uint64_t new_npts = base_npts + add_npts;
    if (new_npts > std::numeric_limits<uint32_t>::max()) {
        LOG(ERROR) << "Append would exceed uint32 header limit for " << path << ": " << new_npts;
        return false;
    }

    io.seekp(0, io.beg);
    uint32_t new_npts_u32 = (uint32_t)new_npts;
    io.write((char *)&new_npts_u32, sizeof(uint32_t));
    io.write((char *)&ndims_u32, sizeof(uint32_t));

    const std::streamoff offset =
        (std::streamoff)(2 * sizeof(uint32_t)) + (std::streamoff)base_npts * (std::streamoff)ndims * sizeof(T);
    io.seekp(offset, io.beg);
    io.write((char *)payload, (std::streamsize)(add_npts * ndims * sizeof(T)));
    io.close();
    return true;
}

template<typename TagT>
static void write_tags_full(const std::string &path, uint64_t total_npts) {
    if (total_npts > std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error("Tag count exceeds uint32 header limit: " + std::to_string(total_npts));
    }
    std::ofstream out(path, std::ios::binary | std::ios::out);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to open tags file for write: " + path);
    }
    uint32_t npts_u32 = (uint32_t)total_npts;
    uint32_t ndims_u32 = 1;
    out.write((char *)&npts_u32, sizeof(uint32_t));
    out.write((char *)&ndims_u32, sizeof(uint32_t));

    const size_t kChunk = 1 << 20;
    std::vector<TagT> buf;
    buf.resize(std::min<uint64_t>(kChunk, total_npts));
    uint64_t written = 0;
    while (written < total_npts) {
        const size_t cur = (size_t) std::min<uint64_t>(total_npts - written, buf.size());
        for (size_t i = 0; i < cur; ++i) {
            buf[i] = (TagT)(written + i);
        }
        out.write((char *)buf.data(), (std::streamsize)(cur * sizeof(TagT)));
        written += cur;
    }
    out.close();
}

template<typename TagT>
static bool append_tags_stream(const std::string &path, uint64_t base_npts, uint64_t add_npts) {
    if (add_npts == 0) {
        return true;
    }
    std::fstream io(path, std::ios::binary | std::ios::in | std::ios::out);
    if (!io.is_open()) {
        return false;
    }
    uint32_t npts_u32 = 0;
    uint32_t ndims_u32 = 0;
    io.read((char *)&npts_u32, sizeof(uint32_t));
    io.read((char *)&ndims_u32, sizeof(uint32_t));
    if (!io.good()) {
        return false;
    }
    if (ndims_u32 != 1 || npts_u32 != (uint32_t)base_npts) {
        LOG(WARNING) << "Tags header mismatch for " << path << ": file(npts=" << npts_u32 << ", ndims=" << ndims_u32
                     << ") vs expected(npts=" << base_npts << ", ndims=1)";
        return false;
    }
    uint64_t new_npts = base_npts + add_npts;
    if (new_npts > std::numeric_limits<uint32_t>::max()) {
        LOG(ERROR) << "Append would exceed uint32 header limit for tags file: " << new_npts;
        return false;
    }

    io.seekp(0, io.beg);
    uint32_t new_npts_u32 = (uint32_t)new_npts;
    io.write((char *)&new_npts_u32, sizeof(uint32_t));
    io.write((char *)&ndims_u32, sizeof(uint32_t));

    const std::streamoff offset =
        (std::streamoff)(2 * sizeof(uint32_t)) + (std::streamoff)base_npts * (std::streamoff)sizeof(TagT);
    io.seekp(offset, io.beg);

    const size_t kChunk = 1 << 20;
    std::vector<TagT> buf;
    buf.resize(std::min<uint64_t>(kChunk, add_npts));
    uint64_t written = 0;
    while (written < add_npts) {
        const size_t cur = (size_t) std::min<uint64_t>(add_npts - written, buf.size());
        for (size_t i = 0; i < cur; ++i) {
            buf[i] = (TagT)(base_npts + written + i);
        }
        io.write((char *)buf.data(), (std::streamsize)(cur * sizeof(TagT)));
        written += cur;
    }
    io.close();
    return true;
}

template<typename T, typename TagT>
static void persist_append_only(pipeann::DynamicSSDIndex<T, TagT> &sync_index, const std::string &persist_prefix,
                                uint64_t base_count, uint64_t total_after, bool write_tags) {
#ifndef IN_PLACE_RECORD_UPDATE
    throw std::runtime_error("Append-only persist requires IN_PLACE_RECORD_UPDATE. Rebuild with -DIN_PLACE_RECORD_UPDATE");
#endif
    auto *disk = sync_index._disk_index;
    if (disk == nullptr) {
        throw std::runtime_error("Disk index is null");
    }

    while (!disk->bg_tasks.empty()) {
        sleep(1);
    }
    while (disk->bg_tasks_inflight.load(std::memory_order_relaxed) != 0) {
        sleep(1);
    }
    disk->flush_page_cache();

    if (disk->cur_id != total_after) {
        LOG(WARNING) << "cur_id (" << disk->cur_id << ") != total_after (" << total_after
                     << "). Persisting with total_after.";
    }
    disk->num_points = total_after;

    if (disk->nnodes_per_sector == 0) {
        throw std::runtime_error("nnodes_per_sector is 0; cannot compute disk file size.");
    }

    const uint64_t file_size =
        SECTOR_LEN + ROUND_UP(total_after, disk->nnodes_per_sector) / disk->nnodes_per_sector * SECTOR_LEN;
    const uint64_t medoid = disk->get_init_ids()[0];
    std::vector<uint64_t> meta;
    meta.push_back(total_after);
    meta.push_back((uint64_t)disk->data_dim);
    meta.push_back(medoid);
    meta.push_back(disk->max_node_len);
    meta.push_back(disk->nnodes_per_sector);
    meta.push_back(disk->num_frozen_points);
    meta.push_back(disk->frozen_location);
    meta.push_back(file_size);

    const std::string disk_index_out = persist_prefix + "_disk.index";
    if (!file_exists(disk_index_out)) {
        throw std::runtime_error("Disk index file missing: " + disk_index_out);
    }
    // Update header in-place (do NOT truncate, or we will wipe the graph).
    {
        std::fstream io(disk_index_out, std::ios::binary | std::ios::in | std::ios::out);
        if (!io.is_open()) {
            throw std::runtime_error("Failed to open disk index for update: " + disk_index_out);
        }
        uint32_t npts_u32 = (uint32_t)meta.size();
        uint32_t ndims_u32 = 1;
        io.seekp(0, std::ios::beg);
        io.write((char *)&npts_u32, sizeof(uint32_t));
        io.write((char *)&ndims_u32, sizeof(uint32_t));
        io.write((char *)meta.data(), (std::streamsize)(meta.size() * sizeof(uint64_t)));
        io.flush();
    }
    const uint64_t cur_size = get_file_size(disk_index_out);
    if (cur_size < file_size) {
        (void)truncate(disk_index_out.c_str(), file_size);
    } else if (cur_size > file_size) {
        LOG(WARNING) << "Disk index file size (" << cur_size << ") > expected (" << file_size
                     << "). Keeping existing size to avoid data loss.";
    }

    const uint64_t add_count = total_after - base_count;
    const std::string pq_path = persist_prefix + "_pq_compressed.bin";
    const uint8_t *pq_ptr = disk->data.data() + base_count * disk->n_chunks;
    bool appended = append_bin_payload<uint8_t>(pq_path, pq_ptr, base_count, add_count, disk->n_chunks);
    if (!appended) {
        LOG(WARNING) << "Append failed for PQ file. Rewriting full PQ.";
        pipeann::save_bin<uint8_t>(pq_path, disk->data.data(), total_after, disk->n_chunks);
    }

    if (write_tags) {
        const std::string tags_path = persist_prefix + "_disk.index.tags";
        bool tags_appended = append_tags_stream<TagT>(tags_path, base_count, add_count);
        if (!tags_appended) {
            LOG(WARNING) << "Append failed for tags file. Rewriting full tags as identity mapping.";
            write_tags_full<TagT>(tags_path, total_after);
        }
    }
}

// ================= ???? =================
/*data_load: ????????
  sync_index: ??SSD???????
  start_tag: ?????
  npts: ???????
  dim: ????
  total_done: ????????????????
  total_target: ???????*/
template<typename T, typename TagT>
void insertion_kernel(T *data_load, pipeann::DynamicSSDIndex<T, TagT> &sync_index, TagT start_tag, size_t npts, size_t dim, std::atomic_size_t &total_done, size_t total_target) {
    pipeann::Timer timer;
    std::atomic_size_t done(0);// ???
    const size_t log_every = std::max<size_t>(1000, npts / 20);
    
    // OMP ????
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

// ================= ???? (?????? + ??ID??) =================
template<typename T, typename TagT = uint32_t>
void get_trace(std::string insert_data_bin, uint64_t file_read_offset, uint64_t global_start_tag, uint64_t n, 
               std::vector<TagT> &insert_tags, std::vector<T> &data_load) {
    
    // 1. ???? Tags (Tag Remapping)
    insert_tags.reserve(n);
    for (uint64_t i = 0; i < n; ++i) {
        insert_tags.push_back((TagT)(global_start_tag + i));
    }

    // 2. ????
    std::ifstream reader(insert_data_bin, std::ios::binary | std::ios::ate);
    if (!reader.is_open()) {
        throw std::runtime_error("Error: Cannot open insert data file: " + insert_data_bin);
    }
    
    // ??????
    reader.seekg(0, reader.beg);
    int npts_i32, dim_i32;// ??????????
    reader.read((char *) &npts_i32, sizeof(int));
    reader.read((char *) &dim_i32, sizeof(int));

    size_t data_dim = dim_i32;
    size_t header_size = 2 * sizeof(int);
    
    // ??????
    size_t required_offset_bytes = header_size + file_read_offset * data_dim * sizeof(T);
    size_t required_bytes = n * data_dim * sizeof(T);// ????????????n????????????????????????

    data_load.resize(n * data_dim);
    reader.seekg(required_offset_bytes, reader.beg);
    if (!reader.read((char *) data_load.data(), required_bytes)) {
        throw std::runtime_error("Error: Failed to read vector data. File might be too short.");
    }
}

// ================= ??? =================
template<typename T>
void read_vectors_chunk(std::ifstream &reader, uint64_t file_read_offset, uint64_t n, size_t data_dim,
                        size_t header_size, std::vector<T> &data_load) {
    const size_t vec_bytes = data_dim * sizeof(T);// ????????????? ? ????????
    const size_t required_offset_bytes = header_size + file_read_offset * vec_bytes;// ?????????????????? + ????? ? ???????
    const size_t required_bytes = n * vec_bytes;// ?????????????? ? ???????

    data_load.resize(n * data_dim);
    reader.clear();
    reader.seekg(required_offset_bytes, reader.beg);
    if (!reader.read((char *) data_load.data(), required_bytes)) {
        throw std::runtime_error("Error: Failed to read vector data. File might be too short.");
    }
}

/* insert_data_bin: ???????????        L_disk: ?????????????
vecs_per_step: ?????????                 num_steps: ????
beam_width: ??????                         chunk_size: ???????
nodes_to_cache: ??????                     index_prefix: ??????
dist_cmp: ???????? */

template<typename T, typename TagT>
void run_insert_merge(const std::string &insert_data_bin, const unsigned L_disk, int vecs_per_step, int num_steps,
                      unsigned beam_width, size_t chunk_size, unsigned nodes_to_cache,
                      const std::string &index_prefix, pipeann::Distance<T> *dist_cmp,
                      const std::string &persist_mode, bool write_tags) {
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

    uint64_t total_plan_insert = (uint64_t)vecs_per_step * num_steps;// ?????????
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

    const std::string persist_mode_l = to_lower_copy(persist_mode);
    const bool use_merge_graph = (persist_mode_l == "merge_graph");
    const bool use_merge_full = (persist_mode_l == "merge");
    const bool use_merge_stream = (persist_mode_l == "merge_stream");
    if (!use_merge_graph && !use_merge_full && !use_merge_stream) {
        throw std::runtime_error("PersistMode must be one of: merge | merge_graph | merge_stream");
    }

#ifndef IN_PLACE_RECORD_UPDATE
    if (use_merge_graph) {
        throw std::runtime_error("merge_graph requires IN_PLACE_RECORD_UPDATE. Rebuild with -DIN_PLACE_RECORD_UPDATE");
    }
#endif

    // 1. init SSD index
    pipeann::Parameters paras;
    paras.Set<unsigned>("L_disk", L_disk);
    paras.Set<unsigned>("R_disk", 0);
    paras.Set<float>("alpha_disk", 1.2);
    paras.Set<unsigned>("C", 384);
    paras.Set<unsigned>("beamwidth", beam_width);
    paras.Set<unsigned>("nodes_to_cache", nodes_to_cache);
    paras.Set<unsigned>("num_threads", NUM_INSERT_THREADS);

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

    pipeann::Metric metric = pipeann::Metric::L2;
    pipeann::DynamicSSDIndex<T, TagT> sync_index(paras, index_prefix, index_prefix + "_merge", dist_cmp, metric, 0, false);

    begin_time = globalTimer.elapsed() / 1.0e6f;

    uint64_t base_count = sync_index._disk_index->num_points;// ??????????
    LOG(INFO) << "=== Index Loaded ===";
    if (base_count == 0) {
        LOG(INFO) << "[Warning] Base index is EMPTY! New vectors will start from Tag 0.";
    } else {
        LOG(INFO) << "Current Index Size: " << base_count << " vectors.";
        LOG(INFO) << "New vectors will be appended starting from Tag: " << base_count;
    }

    const uint64_t final_tag = base_count + (uint64_t)vecs_per_step * num_steps - 1;// ???????
    if (final_tag > std::numeric_limits<TagT>::max()) {
        throw std::runtime_error("Tag range exceeds TagT max. Reduce total inserts or use larger TagT.");
    }

    const uint64_t total_after = base_count + (uint64_t) vecs_per_step * num_steps;
    const uint64_t n_chunks = sync_index._disk_index->n_chunks;// ?? PQ ???
    const uint64_t pq_bytes = total_after * n_chunks;// ?? PQ ?????????
    const uint64_t tag_bytes = total_after * sizeof(TagT);// ????????????
    const uint64_t chunk_bytes = (uint64_t) chunk_size * data_dim * sizeof(T);// ????????????
    LOG(INFO) << "[MemEstimate] Total vectors after insert: " << total_after;
    LOG(INFO) << "[MemEstimate] PQ compressed bytes (approx): " << format_bytes((double) pq_bytes)
              << " (n_chunks=" << n_chunks << ")";
    LOG(INFO) << "[MemEstimate] Tags array bytes (min): " << format_bytes((double) tag_bytes);
    LOG(INFO) << "[MemEstimate] Chunk buffer bytes: " << format_bytes((double) chunk_bytes);
    LOG(INFO) << "[MemEstimate] Actual RSS will be higher due to hash maps, page layout, and thread scratch buffers.";

    if (use_merge_graph || use_merge_stream) {
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
    const size_t total_target = (size_t) total_plan_insert;// ???????


    const size_t dim_for_log = data_dim;
    for (int i = 0; i < num_steps; i++) {
        std::vector<T> data_load;
        data_load.reserve((size_t)chunk_size * data_dim);// ?????????????????

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

    if (use_merge_full) {
        // 3. merge to disk (original full merge)
        LOG(INFO) << "=== Merging to Disk (Making changes permanent) ===";
        pipeann::Timer merge_timer;

        sync_index.final_merge(NUM_INSERT_THREADS);

        float merge_time = merge_timer.elapsed() / 1.0e6f;
        LOG(INFO) << "Merge completed in " << merge_time << "s!";
        LOG(INFO) << ">>> All Done. New index saved to: " << index_prefix << "_merge";
        return;
    }

    if (use_merge_stream) {
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
    LOG(INFO) << "=== Full Merge (Stream PQ/Tags) ===";
        pipeann::Timer merge_timer;
        sync_index.final_merge_stream_pq_tags(NUM_INSERT_THREADS);
        float merge_time = merge_timer.elapsed() / 1.0e6f;
        LOG(INFO) << "Merge(stream PQ/tags) completed in " << merge_time << "s!";
        verify_pq_tags_headers(index_prefix + "_merge", total_vectors_now,
                               static_cast<uint32_t>(n_chunks), write_tags);
        LOG(INFO) << ">>> All Done. New index saved to: " << index_prefix << "_merge";
        return;
    }

    // merge_graph: reuse PQ/tags, only rebuild graph
    const std::string persist_prefix = sync_index._disk_index_prefix_in;
    LOG(INFO) << "=== Persist Append-Only (Update PQ/Tags) ===";
    pipeann::Timer persist_timer;
    persist_append_only<T, TagT>(sync_index, persist_prefix, base_count, total_vectors_now, write_tags);
    float persist_time = persist_timer.elapsed() / 1.0e6f;
    LOG(INFO) << "Append-only persist completed in " << persist_time << "s!";

    LOG(INFO) << "=== Graph-Only Merge (No PQ/Tags Rebuild) ===";
    pipeann::Timer graph_timer;
    sync_index.final_merge_graph(NUM_INSERT_THREADS);
    float graph_time = graph_timer.elapsed() / 1.0e6f;
    LOG(INFO) << "Graph-only merge completed in " << graph_time << "s!";
    verify_graph_links<T, TagT>(sync_index._disk_index,
                                sync_index._disk_index_prefix_in + "_disk.index",
                                base_count, total_vectors_now);
    verify_pq_tags_headers(index_prefix + "_merge", total_vectors_now,
                           static_cast<uint32_t>(sync_index._disk_index->n_chunks), write_tags);
    LOG(INFO) << ">>> All Done. New index saved to: " << index_prefix << "_merge";
}

int main(int argc, char **argv) {
    // ??????
    try {
        if (argc < 7) {
            LOG(INFO) << "Usage: ./insert_demo <DataType> <NewDataBin> <L_disk> <VecsPerStep> <Steps> <Threads> <IndexPrefix> "
                      << "[Beamwidth] [ChunkSize] [NodesToCache] [PersistMode] [WriteTags]";
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
        std::string persist_mode = "merge_stream";
        bool write_tags = false;
        // Backward-compatible parsing:
        //  - 1 trailing arg: treat as ChunkSize
        //  - 2 trailing args: Beamwidth, ChunkSize
        //  - 3 trailing args: Beamwidth, ChunkSize, NodesToCache
        //  - 4 trailing args: Beamwidth, ChunkSize, NodesToCache, PersistMode
        //  - 5 trailing args: Beamwidth, ChunkSize, NodesToCache, PersistMode, WriteTags
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
                remaining -= 1;
            }
            if (remaining >= 1) {
                persist_mode = to_lower_copy(std::string(argv[arg_no++]));
                remaining -= 1;
            }
            if (remaining >= 1) {
                write_tags = (std::atoi(argv[arg_no++]) != 0);
                remaining -= 1;
            }
        }

        // [Validation] ???????
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
        if (persist_mode != "merge" && persist_mode != "merge_graph" && persist_mode != "merge_stream") {
            LOG(INFO) << "Error: PersistMode must be one of: merge | merge_graph | merge_stream";
            exit(-1);
        }

#ifndef IN_PLACE_RECORD_UPDATE
        if (persist_mode == "merge_graph") {
            LOG(INFO) << "Error: merge_graph requires IN_PLACE_RECORD_UPDATE. Rebuild with -DIN_PLACE_RECORD_UPDATE";
            exit(-1);
        }
#endif

        LOG(INFO) << "--- Configuration ---";
        LOG(INFO) << "Type: " << type_str;
        LOG(INFO) << "New Data: " << insert_data_bin;
        LOG(INFO) << "Index Prefix: " << index_prefix;
        LOG(INFO) << "Threads: " << NUM_INSERT_THREADS;
        LOG(INFO) << "Beamwidth: " << beam_width;
        LOG(INFO) << "NodesToCache: " << nodes_to_cache;
        LOG(INFO) << "Total Insert: " << (long)vecs_per_step * num_steps;
        LOG(INFO) << "Chunk Size: " << chunk_size;
        LOG(INFO) << "PersistMode: " << persist_mode;
        LOG(INFO) << "WriteTags: " << (write_tags ? "true" : "false");

        if (type_str == "int8") {
            pipeann::DistanceL2Int8 dist_cmp;
            run_insert_merge<int8_t, unsigned>(insert_data_bin, L_disk, vecs_per_step, num_steps, beam_width, chunk_size,
                                               nodes_to_cache, index_prefix, &dist_cmp, persist_mode, write_tags);
        } else if (type_str == "uint8") {
            pipeann::DistanceL2UInt8 dist_cmp;
            run_insert_merge<uint8_t, unsigned>(insert_data_bin, L_disk, vecs_per_step, num_steps, beam_width, chunk_size,
                                                nodes_to_cache, index_prefix, &dist_cmp, persist_mode, write_tags);
        } else {
            pipeann::DistanceL2 dist_cmp;
            run_insert_merge<float, unsigned>(insert_data_bin, L_disk, vecs_per_step, num_steps, beam_width, chunk_size,
                                              nodes_to_cache, index_prefix, &dist_cmp, persist_mode, write_tags);
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
