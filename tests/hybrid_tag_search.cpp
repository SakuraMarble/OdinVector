// Hybrid Tag-Vector Search for PipeANN
// Strategy: use inverted index hit_rate to choose between:
//   - Low hit_rate  (<  threshold): IVF PQ coarse ranking + mmap exact rerank
//   - High hit_rate (>= threshold): Graph beam search + post-filter by tag

#include <cstring>
#include <omp.h>
#include <ssd_index.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <atomic>
#include <algorithm>
#include <limits>
#include <type_traits>
#include <stack>
#include <sstream>
#include <cctype>
#include <immintrin.h>

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include "log.h"
#include "timer.h"
#include "utils.h"
#include "aux_utils.h"
#include "linux_aligned_file_reader.h"

#include "InvertedIndex.hpp"

// ==========================================
// MMap Raw Vector Reader (for Reranking)
// ==========================================
template <typename T>
class RawVectorReader {
public:
    int fd = -1;
    size_t file_size = 0;
    char* data_ptr = nullptr;
    uint32_t num = 0;
    uint32_t dim = 0;

    explicit RawVectorReader(const std::string& path) {
        fd = open(path.c_str(), O_RDONLY);
        if (fd == -1) throw std::runtime_error("Cannot open raw vector file: " + path);

        struct stat sb;
        if (fstat(fd, &sb) == -1) {
            close(fd);
            throw std::runtime_error("Cannot stat raw vector file");
        }
        file_size = sb.st_size;

        data_ptr = (char*)mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (data_ptr == MAP_FAILED) {
            close(fd);
            throw std::runtime_error("mmap failed for raw vector file");
        }

        std::memcpy(&num, data_ptr, sizeof(uint32_t));
        std::memcpy(&dim, data_ptr + sizeof(uint32_t), sizeof(uint32_t));

        size_t expected_size = 2 * sizeof(uint32_t) + (size_t)num * dim * sizeof(T);
        if (file_size != expected_size) {
            std::cout << "Warning: File size mismatch. Expected: " << expected_size
                      << ", Actual: " << file_size << std::endl;
        }

        std::cout << "[RawVectors] Mapped " << num << " vectors, Dim = " << dim
                  << ", sizeof(T) = " << sizeof(T) << " bytes." << std::endl;
    }

    ~RawVectorReader() {
        if (data_ptr && data_ptr != MAP_FAILED) {
            munmap(data_ptr, file_size);
        }
        if (fd != -1) {
            close(fd);
        }
    }

    const T* get_vec(uint32_t i) const {
        if (i >= num) return nullptr;
        size_t offset = 2 * sizeof(uint32_t) + (size_t)i * (size_t)dim * sizeof(T);
        return reinterpret_cast<const T*>(data_ptr + offset);
    }
};

// ==========================================
// Expression Parser (Shunting-yard RPN)
// ==========================================
class ExpressionParser {
public:
    explicit ExpressionParser(InvertedIndexSearcher& index) : _index(index) {}

    roaring::Roaring parse(const std::string& expression) {
        if (expression.empty()) return _index.query_bitmap({});

        bool is_simple_id = true;
        for (char c : expression) {
            if (!std::isdigit(c)) { is_simple_id = false; break; }
        }
        if (is_simple_id) {
            return _index.query_bitmap({std::stoi(expression)});
        }

        std::string rpn = to_rpn(expression);
        return evaluate_rpn(rpn);
    }

private:
    InvertedIndexSearcher& _index;

    int precedence(char op) {
        if (op == '&') return 2;
        if (op == '|') return 1;
        if (op == '-') return 2;
        return 0;
    }

    std::string to_rpn(const std::string& exp) {
        std::string output;
        std::stack<char> ops;
        for (size_t i = 0; i < exp.length(); ++i) {
            char c = exp[i];
            if (std::isspace(c)) continue;
            if (std::isdigit(c)) {
                while (i < exp.length() && std::isdigit(exp[i])) {
                    output += exp[i]; i++;
                }
                output += ' '; i--;
            } else if (c == '(') {
                ops.push(c);
            } else if (c == ')') {
                while (!ops.empty() && ops.top() != '(') {
                    output += ops.top(); output += ' '; ops.pop();
                }
                if (!ops.empty()) ops.pop();
            } else if (c == '&' || c == '|' || c == '-') {
                while (!ops.empty() && precedence(ops.top()) >= precedence(c)) {
                    output += ops.top(); output += ' '; ops.pop();
                }
                ops.push(c);
            }
        }
        while (!ops.empty()) {
            output += ops.top(); output += ' '; ops.pop();
        }
        return output;
    }

    roaring::Roaring evaluate_rpn(const std::string& rpn) {
        std::stack<roaring::Roaring> stk;
        std::stringstream ss(rpn);
        std::string token;
        while (ss >> token) {
            if (std::isdigit(token[0])) {
                stk.push(_index.query_bitmap({std::stoi(token)}));
            } else {
                if (stk.size() < 2) throw std::runtime_error("Invalid expression");
                roaring::Roaring b = stk.top(); stk.pop();
                roaring::Roaring a = stk.top(); stk.pop();
                if (token == "&") a &= b;
                else if (token == "|") a |= b;
                else if (token == "-") a -= b;
                stk.push(a);
            }
        }
        return stk.empty() ? roaring::Roaring() : stk.top();
    }
};

// ==========================================
// AVX2 Distance Functions
// ==========================================
inline float L2_AVX2_Float(const float* a, const float* b, size_t dim) {
    __m256 sum = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 7 < dim; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 diff = _mm256_sub_ps(va, vb);
        sum = _mm256_fmadd_ps(diff, diff, sum);
    }
    __m128 sum_high = _mm256_extractf128_ps(sum, 1);
    __m128 sum_low = _mm256_castps256_ps128(sum);
    __m128 sum128 = _mm_add_ps(sum_low, sum_high);
    __m128 shuf = _mm_movehdup_ps(sum128);
    __m128 sums = _mm_add_ps(sum128, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ps(sums, shuf);
    float total = _mm_cvtss_f32(sums);
    for (; i < dim; ++i) {
        float d = a[i] - b[i];
        total += d * d;
    }
    return total;
}

inline float L2_AVX2_Uint8(const uint8_t* a, const uint8_t* b, size_t dim) {
    __m256i sum = _mm256_setzero_si256();
    size_t i = 0;
    for (; i + 16 <= dim; i += 16) {
        __m128i v_a_128 = _mm_loadu_si128((const __m128i*)(a + i));
        __m128i v_b_128 = _mm_loadu_si128((const __m128i*)(b + i));
        __m256i v_a_256 = _mm256_cvtepu8_epi16(v_a_128);
        __m256i v_b_256 = _mm256_cvtepu8_epi16(v_b_128);
        __m256i diff = _mm256_sub_epi16(v_a_256, v_b_256);
        __m256i sqr = _mm256_madd_epi16(diff, diff);
        sum = _mm256_add_epi32(sum, sqr);
    }
    int32_t buffer[8];
    _mm256_storeu_si256((__m256i*)buffer, sum);
    int32_t total_int = 0;
    for (int k = 0; k < 8; ++k) total_int += buffer[k];
    for (; i < dim; ++i) {
        int d = (int)a[i] - (int)b[i];
        total_int += d * d;
    }
    return (float)total_int;
}

// ==========================================
// Optimized PQ Batch Distance Computation
// Mirrors pq_flash_index.cpp::compute_pq_dists_batch:
//   block-level gather + 4-chunk unrolled lookup + prefetch(T0)
//   + #pragma omp parallel if(n > 10000) for large candidate sets
// ==========================================
template <typename T>
void compute_pq_dists_batch(
    pipeann::SSDIndex<T>* index,
    const T* query,
    const std::vector<uint32_t>& candidate_ids,
    std::vector<float>& dists_out)
{
    const size_t n_chunks = index->n_chunks;
    const size_t n = candidate_ids.size();
    dists_out.resize(n);
    if (n == 0) return;

    // Build query->centroid distance table: 256 * n_chunks floats
    // PipeANN's FixedChunkPQTable::populate_chunk_distances is templated on T,
    // so it handles float/int8/uint8 internally.
    std::vector<float> pq_dists(256 * n_chunks);
    index->pq_table.populate_chunk_distances(query, pq_dists.data());

    constexpr size_t BLOCK_SIZE    = 256;
    constexpr size_t PREFETCH_DIST = 32;

    // This function is called from inside the per-query OMP parallel loop,
    // so we do NOT use nested parallelism here (it would cause thread explosion
    // and hang). The outer loop already provides query-level parallelism.
    // Serial block processing with prefetch + 4-chunk unrolling provides
    // sufficient ILP performance per query.
    std::vector<uint8_t> local_buf(BLOCK_SIZE * n_chunks);

    for (size_t blk = 0; blk < n; blk += BLOCK_SIZE) {
        const size_t blk_end = std::min(blk + BLOCK_SIZE, n);
        const size_t blk_sz  = blk_end - blk;

        // Step 1: Gather PQ codes into local buffer (improves L1 locality)
        for (size_t i = 0; i < blk_sz; ++i) {
            const size_t global_idx = blk + i;
            // Prefetch next entry into L1 cache (T0 = L1, matching reference)
            if (i + PREFETCH_DIST < blk_sz) {
                _mm_prefetch(reinterpret_cast<const char*>(
                    index->data.data() +
                    (size_t)candidate_ids[global_idx + PREFETCH_DIST] * n_chunks),
                    _MM_HINT_T0);
            }
            const uint32_t vid = candidate_ids[global_idx];
            memcpy(local_buf.data() + i * n_chunks,
                   index->data.data() + (size_t)vid * n_chunks,
                   n_chunks);
        }

        // Step 2: Lookup distances, 4-chunk unrolling for ILP
        for (size_t i = 0; i < blk_sz; ++i) {
            const uint8_t* pq_code = local_buf.data() + i * n_chunks;
            float dist = 0.0f;
            size_t chunk = 0;
            for (; chunk + 4 <= n_chunks; chunk += 4) {
                dist += pq_dists[256 * (chunk + 0) + pq_code[chunk + 0]];
                dist += pq_dists[256 * (chunk + 1) + pq_code[chunk + 1]];
                dist += pq_dists[256 * (chunk + 2) + pq_code[chunk + 2]];
                dist += pq_dists[256 * (chunk + 3) + pq_code[chunk + 3]];
            }
            for (; chunk < n_chunks; ++chunk) {
                dist += pq_dists[256 * chunk + pq_code[chunk]];
            }
            dists_out[blk + i] = dist;
        }
    }
}

// ==========================================
// SearchResult helper
// ==========================================
struct SearchResult {
    uint32_t id;
    float dist;
    bool operator<(const SearchResult& other) const { return dist < other.dist; }
};

// ==========================================
// IVF PQ Coarse Ranking + Exact Rerank
// ==========================================
template <typename T>
void ivf_pq_search_with_rerank(
    const T* query, uint32_t query_dim,
    const std::vector<uint32_t>& candidate_ids,
    pipeann::SSDIndex<T>* index,
    const RawVectorReader<T>& raw_reader,
    uint32_t top_L, uint32_t final_K,
    std::vector<uint32_t>& result_ids,
    std::vector<float>& result_dists)
{
    if (candidate_ids.empty()) {
        result_ids.assign(final_K, 0);
        result_dists.assign(final_K, std::numeric_limits<float>::max());
        return;
    }

    // --- Stage 1: PQ coarse ranking ---
    std::vector<float> pq_dist_results;
    compute_pq_dists_batch<T>(index, query, candidate_ids, pq_dist_results);

    size_t num_candidates = candidate_ids.size();
    std::vector<SearchResult> pq_results(num_candidates);
    for (size_t i = 0; i < num_candidates; ++i) {
        pq_results[i] = {candidate_ids[i], pq_dist_results[i]};
    }

    size_t actual_L = std::min((size_t)top_L, pq_results.size());
    if (actual_L < pq_results.size()) {
        std::nth_element(pq_results.begin(),
                         pq_results.begin() + actual_L,
                         pq_results.end());
    }
    std::sort(pq_results.begin(), pq_results.begin() + actual_L);

    // --- Stage 2: Exact rerank via mmap ---
    std::vector<SearchResult> rerank_results(actual_L);

    if constexpr (std::is_same<T, uint8_t>::value) {
        const uint8_t* query_u8 = reinterpret_cast<const uint8_t*>(query);
        for (size_t i = 0; i < actual_L; ++i) {
            uint32_t vid = pq_results[i].id;
            const T* raw_ptr = raw_reader.get_vec(vid);
            if (raw_ptr) {
                float d = L2_AVX2_Uint8(query_u8,
                                        reinterpret_cast<const uint8_t*>(raw_ptr),
                                        query_dim);
                rerank_results[i] = {vid, d};
            } else {
                rerank_results[i] = {vid, std::numeric_limits<float>::max()};
            }
        }
    } else if constexpr (std::is_same<T, float>::value) {
        const float* query_f = reinterpret_cast<const float*>(query);
        for (size_t i = 0; i < actual_L; ++i) {
            uint32_t vid = pq_results[i].id;
            const T* raw_ptr = raw_reader.get_vec(vid);
            if (raw_ptr) {
                float d = L2_AVX2_Float(query_f,
                                        reinterpret_cast<const float*>(raw_ptr),
                                        query_dim);
                rerank_results[i] = {vid, d};
            } else {
                rerank_results[i] = {vid, std::numeric_limits<float>::max()};
            }
        }
    } else {
        // int8 and other types: convert to float
        std::vector<float> qf(query_dim), rf(query_dim);
        for (uint32_t d = 0; d < query_dim; d++) qf[d] = static_cast<float>(query[d]);
        for (size_t i = 0; i < actual_L; ++i) {
            uint32_t vid = pq_results[i].id;
            const T* raw_ptr = raw_reader.get_vec(vid);
            if (raw_ptr) {
                for (uint32_t d = 0; d < query_dim; d++) rf[d] = static_cast<float>(raw_ptr[d]);
                float dist = L2_AVX2_Float(qf.data(), rf.data(), query_dim);
                rerank_results[i] = {vid, dist};
            } else {
                rerank_results[i] = {vid, std::numeric_limits<float>::max()};
            }
        }
    }

    // --- Stage 3: Select final top-K ---
    size_t actual_K = std::min((size_t)final_K, rerank_results.size());
    if (actual_K < rerank_results.size()) {
        std::nth_element(rerank_results.begin(),
                         rerank_results.begin() + actual_K,
                         rerank_results.end());
    }
    std::sort(rerank_results.begin(), rerank_results.begin() + actual_K);

    result_ids.resize(final_K);
    result_dists.resize(final_K);
    for (size_t i = 0; i < actual_K; ++i) {
        result_ids[i] = rerank_results[i].id;
        result_dists[i] = rerank_results[i].dist;
    }
    for (size_t i = actual_K; i < final_K; ++i) {
        result_ids[i] = 0;
        result_dists[i] = std::numeric_limits<float>::max();
    }
}


// ==========================================
// Main search function (templated on data type)
// ==========================================
template <typename T>
int hybrid_tag_search(int argc, char** argv) {
    // ---- Parse arguments ----
    // argv[1]:  data_type (already used for dispatch)
    int idx = 2;
    std::string index_prefix_path(argv[idx++]);
    uint32_t num_threads   = std::atoi(argv[idx++]);
    uint32_t beamwidth     = std::atoi(argv[idx++]);
    std::string query_bin(argv[idx++]);
    std::string truthset_bin(argv[idx++]);
    uint64_t recall_at     = std::atoi(argv[idx++]);
    std::string dist_metric(argv[idx++]);
    uint32_t mem_L         = std::atoi(argv[idx++]);
    std::string ivf_index_path(argv[idx++]);
    std::string raw_vector_path(argv[idx++]);
    std::string query_label_file(argv[idx++]);
    float hit_rate_threshold     = std::atof(argv[idx++]);
    uint32_t ivf_topL_multiplier = std::atoi(argv[idx++]);

    std::vector<uint64_t> Lvec;
    for (int i = idx; i < argc; i++) {
        uint64_t curL = std::atoi(argv[i]);
        if (curL >= recall_at) Lvec.push_back(curL);
    }
    if (Lvec.empty()) {
        std::cout << "No valid L values >= K=" << recall_at << std::endl;
        return -1;
    }

    pipeann::Metric m = (dist_metric == "cosine") ? pipeann::Metric::COSINE : pipeann::Metric::L2;

    // ---- Load query vectors ----
    T* query = nullptr;
    size_t query_num, query_dim;
    pipeann::load_bin<T>(query_bin, query, query_num, query_dim);
    std::cout << "Loaded " << query_num << " queries, dim=" << query_dim << std::endl;

    // ---- Load groundtruth ----
    unsigned* gt_ids = nullptr;
    float* gt_dists  = nullptr;
    size_t gt_num = 0, gt_dim = 0;
    bool calc_recall = false;
    if (truthset_bin != "null" && truthset_bin != "NULL" &&
        file_exists(truthset_bin)) {
        pipeann::load_truthset(truthset_bin, gt_ids, gt_dists, gt_num, gt_dim);
        if (gt_num != query_num) {
            std::cout << "Warning: gt_num=" << gt_num << " != query_num=" << query_num << std::endl;
        }
        calc_recall = true;
    }

    // ---- Load inverted index ----
    InvertedIndexSearcher ivf_searcher;
    ivf_searcher.load(ivf_index_path);
    std::cout << "Inverted index loaded." << std::endl;

    // ---- MMap raw vectors for rerank ----
    RawVectorReader<T> raw_reader(raw_vector_path);
    uint64_t total_num_points = raw_reader.num;

    // ---- Load query label expressions ----
    std::vector<std::string> query_labels;
    {
        std::ifstream qlf(query_label_file);
        if (!qlf.is_open()) {
            std::cerr << "Cannot open query label file: " << query_label_file << std::endl;
            return -1;
        }
        std::string line;
        while (std::getline(qlf, line)) {
            // Strip trailing \r
            if (!line.empty() && line.back() == '\r') line.pop_back();
            query_labels.push_back(line);
        }
    }
    if (query_labels.size() != query_num) {
        std::cerr << "Error: query_labels.size()=" << query_labels.size()
                  << " != query_num=" << query_num << std::endl;
        return -1;
    }

    // ---- Load SSD index ----
    std::shared_ptr<AlignedFileReader> reader = nullptr;
    reader.reset(new LinuxAlignedFileReader());

    std::unique_ptr<pipeann::SSDIndex<T>> index(
        new pipeann::SSDIndex<T>(m, reader, false /*single_file*/, true /*tags*/));

    int res = index->load(index_prefix_path.c_str(), num_threads, true, false);
    if (res != 0) {
        std::cerr << "Failed to load disk index." << std::endl;
        return res;
    }

    if (mem_L != 0) {
        auto mem_index_path = index_prefix_path + "_mem.index";
        std::cout << "Loading memory index from " << mem_index_path << std::endl;
        index->load_mem_index(m, query_dim, mem_index_path);
    }

    omp_set_num_threads(num_threads);

    // ---- Per-L search loop ----
    std::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
    std::cout.precision(2);

std::string recall_string = "Recall@" + std::to_string(recall_at);
    std::cout << std::setw(6) << "L"
              << std::setw(12) << "Beamwidth"
              << std::setw(12) << "QPS"
              << std::setw(14) << "AvgLat(us)"
              << std::setw(12) << "P99.9 Lat"
              << std::setw(12) << "IVF/Graph"
              << std::setw(18) << "IVFcand(avg/max)"
              << std::setw(20) << "Graphcand(avg/max)";
    if (calc_recall) std::cout << std::setw(12) << recall_string;
    std::cout << std::endl;
    std::cout << std::string(120, '=') << std::endl;

    for (uint64_t L : Lvec) {
        
        std::vector<uint32_t> result_tags(recall_at * query_num);
        std::vector<float>    result_dists(recall_at * query_num);

        pipeann::QueryStats* stats = new pipeann::QueryStats[query_num];

        std::atomic<size_t> count_ivf(0), count_graph(0);
        std::atomic<uint64_t> total_cand_ivf(0), total_cand_graph(0);
        std::atomic<uint64_t> max_cand_ivf(0), max_cand_graph(0);

        auto t_start = std::chrono::high_resolution_clock::now();

#pragma omp parallel num_threads(num_threads)
        {
            // Per-thread expression parser to avoid lock contention
            ExpressionParser parser(ivf_searcher);

            // Per-thread reusable buffers
            std::vector<uint32_t> cand_ids;
            std::vector<uint32_t> res_ids;
            std::vector<float>    res_d;

            #pragma omp for schedule(dynamic, 1)
            for (int64_t i = 0; i < (int64_t)query_num; i++) {
                // if(i != 272) continue;
                std::cout << "\nProcessing Query " << i << " / " << query_num << std::endl;
                try {
                    roaring::Roaring candidates = parser.parse(query_labels[i]);
                    uint64_t cand_count = candidates.cardinality();
                    float hit_rate = (float)cand_count / (float)total_num_points;

                    cand_ids.resize(cand_count);
                    candidates.toUint32Array(cand_ids.data());

                    if (hit_rate < hit_rate_threshold) {
                        // === IVF path: PQ coarse ranking + exact rerank ===
                        count_ivf++;
                        total_cand_ivf += cand_count;
                        {
                            uint64_t cur_max = max_cand_ivf.load();
                            while (cand_count > cur_max &&
                                   !max_cand_ivf.compare_exchange_weak(cur_max, cand_count));
                        }

                        uint32_t top_L = std::min((uint64_t)recall_at * ivf_topL_multiplier, cand_count);

                        auto q_start = std::chrono::high_resolution_clock::now();
                        
                        
                        ivf_pq_search_with_rerank<T>(
                            query + (i * query_dim), (uint32_t)query_dim,
                            cand_ids, index.get(), raw_reader,
                            top_L, (uint32_t)recall_at,
                            res_ids, res_d);
                        
                        auto q_end = std::chrono::high_resolution_clock::now();

                        for (size_t j = 0; j < recall_at; j++) {
                            result_tags[i * recall_at + j]  = res_ids[j];
                            result_dists[i * recall_at + j] = res_d[j];
                        }
                        stats[i].total_us = std::chrono::duration<double>(q_end - q_start).count() * 1e6;
                        stats[i].n_ios    = 0;
                    } else {
                        // === Graph path: direct filter_beam_search ==
                        count_graph++;
                        total_cand_graph += cand_count;
                        {
                            uint64_t cur_max = max_cand_graph.load();
                            while (cand_count > cur_max &&
                                !max_cand_graph.compare_exchange_weak(cur_max, cand_count));
                        }

                        // 调用 filter_beam_search
                        index->filter_beam_search(
                            query + (i * query_dim), (uint64_t)recall_at, 
                            mem_L, L,
                            result_tags.data()  + i * recall_at,
                            result_dists.data() + i * recall_at, 
                            beamwidth, cand_ids, stats + i, nullptr, false);

                        // ---------------------------------------------------------
                        // [Debug Information] 增强版：打印结果、GT 距离及标签约束验证
                        // ---------------------------------------------------------
                        #pragma omp critical
                        {
                            std::cout << "\n[Debug Graph Path] Query ID: " << i << std::endl;
                            std::cout << "  - Hit Rate: " << hit_rate << " (" << cand_count << " / " << total_num_points << ")" << std::endl;
                            
                            // 1. 打印检索出的 Top-K 详细信息
                            std::cout << "  - [Retrieved Results] (Rank: ID | Dist | Valid?):" << std::endl;
                            for (size_t j = 0; j < recall_at; j++) {
                                uint32_t res_id = result_tags[i * recall_at + j];
                                float res_dist = result_dists[i * recall_at + j];
                                bool is_valid = candidates.contains(res_id);
                                
                                std::cout << "    #" << j << ": " << res_id << " | " << res_dist 
                                        << " | " << (is_valid ? "YES" : "INVALID!!") << std::endl;
                            }

                            // 2. 打印 Ground Truth 详细信息（包含距离）
                            if (calc_recall && gt_ids) {
                                std::cout << "  - [GroundTruth Data] (Rank: ID | Dist | In_Candidates?):" << std::endl;
                                size_t hit_count = 0;
                                for (size_t j = 0; j < recall_at; j++) {
                                    uint32_t gt_id = gt_ids[i * gt_dim + j];
                                    float gt_dist = (gt_dists != nullptr) ? gt_dists[i * gt_dim + j] : -1.0f;
                                    bool in_cand = candidates.contains(gt_id);
                                    
                                    std::cout << "    #" << j << ": " << gt_id << " | " << gt_dist 
                                            << " | " << (in_cand ? "YES" : "NO") << std::endl;

                                    // 检查是否被成功召回
                                    for (size_t k = 0; k < recall_at; k++) {
                                        if (result_tags[i * recall_at + k] == gt_id) {
                                            hit_count++;
                                            break;
                                        }
                                    }
                                }
                                std::cout << "  - Single Query Recall: " << hit_count << " / " << recall_at << std::endl;
                            }
                            std::cout << "---------------------------------------------------------" << std::endl;
                        }
                    }
                } catch (const std::exception& e) {
#pragma omp critical
                    {
                        std::cerr << "Query " << i << " error: " << e.what() << std::endl << std::flush;
                    }
                    for (size_t j = 0; j < recall_at; j++) {
                        result_tags[i * recall_at + j]  = 0;
                        result_dists[i * recall_at + j] = std::numeric_limits<float>::max();
                    }
                    stats[i].total_us = 0;
                    stats[i].n_ios    = 0;
                } catch (...) {
#pragma omp critical
                    {
                        std::cerr << "Query " << i << " FATAL UNKNOWN EXCEPTION!" << std::endl << std::flush;
                    }
                }
            }
        } // end omp parallel

        auto t_end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(t_end - t_start).count();
        double qps = (double)query_num / elapsed;

        double mean_lat = pipeann::get_mean_stats(
            stats, query_num,
            [](const pipeann::QueryStats& s) { return s.total_us; });
        double p999_lat = pipeann::get_percentile_stats(
            stats, query_num, 0.999f,
            [](const pipeann::QueryStats& s) { return s.total_us; });

        delete[] stats;

        // Ratio string
        std::string ratio = std::to_string(count_ivf.load()) + "/" +
                            std::to_string(count_graph.load());

        std::string ivf_cand_str = "-";
        if (count_ivf > 0) {
            ivf_cand_str = std::to_string(total_cand_ivf / count_ivf) + "/" +
                           std::to_string(max_cand_ivf.load());
        }
        std::string graph_cand_str = "-";
        if (count_graph > 0) {
            graph_cand_str = std::to_string(total_cand_graph / count_graph) + "/" +
                             std::to_string(max_cand_graph.load());
        }

        std::cout << std::setw(6) << L
                  << std::setw(12) << beamwidth
                  << std::setw(12) << qps
                  << std::setw(14) << mean_lat
                  << std::setw(12) << p999_lat
                  << std::setw(12) << ratio
                  << std::setw(18) << ivf_cand_str
                  << std::setw(20) << graph_cand_str;

        if (calc_recall) {
            double recall = pipeann::calculate_recall(
                (unsigned)query_num, gt_ids, gt_dists, (unsigned)gt_dim,
                result_tags.data(), (unsigned)recall_at, (unsigned)recall_at);
            std::cout << std::setw(12) << recall;
        }
        std::cout << std::endl;
    }

    delete[] query;
    if (gt_ids)    delete[] gt_ids;
    if (gt_dists)  delete[] gt_dists;

    return 0;
}

// ==========================================
// Main entry point
// ==========================================
int main(int argc, char** argv) {
    if (argc < 16) {
        std::cout << "Usage: " << argv[0]
                  << " <data_type(float/int8/uint8)>"
                  << " <index_prefix_path>"
                  << " <num_threads>"
                  << " <beamwidth>"
                  << " <query_file.bin>"
                  << " <truthset.bin (use \"null\" for none)>"
                  << " <K>"
                  << " <similarity (l2/cosine)>"
                  << " <mem_L (0=no mem index)>"
                  << " <ivf_index_path>"
                  << " <raw_vector_path>"
                  << " <query_label_file>"
                  << " <hit_rate_threshold>"
                  << " <ivf_topL_multiplier>"
                  << " <L1> [L2] ..."
                  << std::endl;
        return -1;
    }

    std::string data_type(argv[1]);
    if (data_type == "float")
        return hybrid_tag_search<float>(argc, argv);
    else if (data_type == "int8")
        return hybrid_tag_search<int8_t>(argc, argv);
    else if (data_type == "uint8")
        return hybrid_tag_search<uint8_t>(argc, argv);
    else {
        std::cerr << "Unsupported data type: " << data_type
                  << ". Use float, int8, or uint8." << std::endl;
        return -1;
    }
}
