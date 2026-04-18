#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <algorithm>
#include "InvertedIndex.hpp"

// 辅助打印函数，限制最多打印 10 个 ID，防止真实数据刷屏
void print_result(const std::string& step, const std::vector<uint32_t>& res, double cost_ms = -1.0) {
    std::cout << ">>> " << step << " | Matches: " << res.size();
    if (cost_ms >= 0) {
        std::cout << " | Time: " << cost_ms << " ms";
    }

    if (!res.empty()) {
        std::cout << " | IDs: [ ";
        size_t limit = std::min<size_t>(res.size(), 10);
        for (size_t i = 0; i < limit; ++i) {
            std::cout << res[i] << " ";
        }
        if (res.size() > limit) std::cout << "... ";
        std::cout << "]";
    }
    std::cout << std::endl;
}

// 获取当前时间戳的 lambda
auto now = []() { return std::chrono::high_resolution_clock::now(); };
auto get_ms = [](auto start, auto end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
};

void run_dummy_test() {
    std::cout << "\n========== RUNNING DUMMY TEST ==========\n" << std::endl;
    const std::string DUMMY_LABLES_PATH = "dummy_labels.txt";
    const std::string INDEX_PATH = "test_index.bin";
    const std::string MERGED_INDEX_PATH = "test_index_merged.bin";
    const int UNIVERSAL_TAG = 11;

    // 1. 造一份基础数据并 Build
    std::cout << "[Step 1] Building Dummy Index..." << std::endl;
    std::ofstream out(DUMMY_LABLES_PATH);
    out << "1,2\n";    // vid 0: tags [1, 2]
    out << "2,3\n";    // vid 1: tags [2, 3]
    out << "11\n";     // vid 2: tag [11] (Universal)
    out << "1,2,3\n";  // vid 3: tags [1, 2, 3]
    out.close();

    InvertedIndexBuilder builder;
    builder.build(DUMMY_LABLES_PATH, INDEX_PATH, UNIVERSAL_TAG);

    // 2. 初始化 Searcher
    std::cout << "\n[Step 2] Loading Dummy Index..." << std::endl;
    InvertedIndexSearcher searcher;
    searcher.load(INDEX_PATH);
    searcher.set_universal_tag(UNIVERSAL_TAG);

    print_result("Baseline Query [1, 2]", searcher.query({1, 2}));

    std::cout << "\n[Step 3] Inserting vid 100[1,2] and 101[11(Univ)]..." << std::endl;
    searcher.insert(100, {1, 2});
    searcher.insert(101, {11});
    print_result("After Insert Query [1, 2]", searcher.query({1, 2}));

    std::cout << "\n[Step 4] Deleting vid 0 and vid 2..." << std::endl;
    searcher.remove(0);
    searcher.remove(2);
    print_result("After Delete Query [1, 2]", searcher.query({1, 2}));

    std::cout << "\n[Step 5] Merging Index..." << std::endl;
    searcher.merge(MERGED_INDEX_PATH);

    std::cout << "\n[Step 6] Reloading Merged Index..." << std::endl;
    InvertedIndexSearcher new_searcher;
    new_searcher.load(MERGED_INDEX_PATH);
    new_searcher.set_universal_tag(UNIVERSAL_TAG);
    print_result("Post-Merge Load Query [1, 2]", new_searcher.query({1, 2}));
}

void run_real_test(int argc, char** argv) {
    if (argc < 6) {
        std::cerr << "Usage for Real Mode: ./test_dynamic 1 <index_path> <merged_index_path> <universal_tag> <tag1> [tag2...]\n";
        return;
    }

    std::cout << "\n========== RUNNING REAL IVF BATCH TEST ==========\n" << std::endl;

    std::string index_path = argv[2];
    std::string merged_index_path = argv[3];
    int universal_tag = std::stoi(argv[4]);

    std::vector<int> query_tags;
    for (int i = 5; i < argc; ++i) {
        query_tags.push_back(std::stoi(argv[i]));
    }

    std::cout << "Config:\n  Index: " << index_path << "\n  Merged Output: " << merged_index_path
              << "\n  Universal Tag: " << universal_tag << "\n  Query Tags: [ ";
    for (int t : query_tags) std::cout << t << " ";
    std::cout << "]\n\n";

    InvertedIndexSearcher searcher;

    // 1. 加载
    auto t0 = now();
    searcher.load(index_path);
    searcher.set_universal_tag(universal_tag);
    auto t1 = now();
    std::cout << ">>> Index Loaded. Time: " << get_ms(t0, t1) << " ms\n\n";

    // 2. 初始查询 (Baseline)
    t0 = now();
    std::vector<uint32_t> baseline_res = searcher.query(query_tags);
    t1 = now();
    print_result("Baseline Query", baseline_res, get_ms(t0, t1));
    size_t baseline_count = baseline_res.size();

    // 3. 批量动态插入 (Batch Insert)
    const int BATCH_INSERT_SIZE = 50000; // 模拟批量插入 5 万条记录
    const uint32_t FAKE_VID_START = 90000000; // 故意弄个超大ID区间防止冲突

    std::cout << "\n>>> [Action] Batch Inserting " << BATCH_INSERT_SIZE << " fake VIDs (Starting from " << FAKE_VID_START << ")...\n";
    t0 = now();
    for (int i = 0; i < BATCH_INSERT_SIZE; ++i) {
        searcher.insert(FAKE_VID_START + i, query_tags);
    }
    t1 = now();
    std::cout << ">>> Batch Insert Time: " << get_ms(t0, t1) << " ms\n";

    t0 = now();
    std::vector<uint32_t> insert_res = searcher.query(query_tags);
    t1 = now();
    print_result("After Batch Insert Query", insert_res, get_ms(t0, t1));
    std::cout << "    -> Expected Matches: " << baseline_count + BATCH_INSERT_SIZE
              << " | Actual: " << insert_res.size() << "\n";

    // 4. 批量动态删除 (Batch Delete)
    // 从 baseline 中挑出一批真实的 ID 删掉，如果没有那么多就全删
    size_t batch_delete_size = std::min<size_t>(baseline_count, 10000); // 最多删 1 万条真实数据

    std::cout << "\n>>> [Action] Batch Deleting " << batch_delete_size << " REAL VIDs from baseline...\n";
    t0 = now();
    for (size_t i = 0; i < batch_delete_size; ++i) {
        searcher.remove(baseline_res[i]); // 删掉真实的 ID
    }
    t1 = now();
    std::cout << ">>> Batch Delete Time: " << get_ms(t0, t1) << " ms\n";

    t0 = now();
    std::vector<uint32_t> delete_res = searcher.query(query_tags);
    t1 = now();
    print_result("After Batch Delete Query", delete_res, get_ms(t0, t1));
    size_t expected_after_delete = baseline_count + BATCH_INSERT_SIZE - batch_delete_size;
    std::cout << "    -> Expected Matches: " << expected_after_delete
              << " | Actual: " << delete_res.size() << "\n";

    // 5. 异步落盘 Merge
    std::cout << "\n>>> [Action] Merging Index to Disk...\n";
    t0 = now();
    searcher.merge(merged_index_path);
    t1 = now();
    std::cout << ">>> Merge & Save Time: " << get_ms(t0, t1) << " ms\n\n";

    // 6. 重启验证 (Reloading)
    std::cout << ">>> [Action] Reloading Merged Index...\n";
    InvertedIndexSearcher new_searcher;
    t0 = now();
    new_searcher.load(merged_index_path);
    new_searcher.set_universal_tag(universal_tag);
    t1 = now();
    std::cout << ">>> Reload Time: " << get_ms(t0, t1) << " ms\n";

    t0 = now();
    std::vector<uint32_t> final_res = new_searcher.query(query_tags);
    t1 = now();
    print_result("Post-Merge Load Query", final_res, get_ms(t0, t1));
    std::cout << "    -> Expected Matches: " << expected_after_delete
              << " | Actual: " << final_res.size() << "\n";
}

// ================= WAL 宕机恢复测试 =================
void run_wal_test() {
    std::cout << "\n========== RUNNING WAL RECOVERY TEST ==========\n" << std::endl;
    const std::string DUMMY_LABLES_PATH = "dummy_labels.txt";
    const std::string INDEX_PATH = "test_index_wal.bin";
    const std::string WAL_PATH = "transaction.wal";
    const int UNIVERSAL_TAG = 11;

    // 1. 初始化基础索引（模拟旧的基础索引）
    std::ofstream out(DUMMY_LABLES_PATH);
    out << "1,2\n";    // vid 0: tags [1, 2]
    out.close();
    InvertedIndexBuilder builder;
    builder.build(DUMMY_LABLES_PATH, INDEX_PATH, UNIVERSAL_TAG);

    // 2. 模拟日常运行（发生内存写和删），随后突然崩溃
    std::cout << "\n[Simulator] Process started..." << std::endl;
    {
        // 我们用一个大括号控制作用域，让 searcher 对象自动销毁（模拟宕机）
        InvertedIndexSearcher searcher;
        searcher.load(INDEX_PATH);
        searcher.set_universal_tag(UNIVERSAL_TAG);

        // **重点**：开启 WAL
        // 若上次有遗留日志可以先 recover_from_wal，这里我们假设全新开始
        // 先清理可能存在的旧 WAL
        std::ofstream truncate_wal(WAL_PATH, std::ios::trunc);
        truncate_wal.close();

        searcher.open_wal(WAL_PATH);

        std::cout << ">>> Executing dynamic operations..." << std::endl;
        searcher.insert(888, {1, 2}); // 插入内存，同时记入 WAL
        searcher.insert(999, {1, 2, 3});
        searcher.remove(0);           // 删掉原本在基础索引里的记录

        print_result("State BEFORE Crash (Query [1, 2])", searcher.query({1, 2}));

        std::cout << "[Simulator] CRASH! (Process killed, mem index lost)\n" << std::endl;
        // 离开作用域，searcher 析构，内存里的 mem_index 和 tombstone 完全消失！
    }

    // 3. 模拟重启恢复流程
    std::cout << "[Simulator] Process Restarting..." << std::endl;
    InvertedIndexSearcher recover_searcher;

    // a. 加载旧的磁盘基础数据
    recover_searcher.load(INDEX_PATH);
    recover_searcher.set_universal_tag(UNIVERSAL_TAG);

    // b. 从 WAL 恢复丢失的内存数据
    recover_searcher.recover_from_wal(WAL_PATH);

    // c. 接着写新的日志（为未来的操作）
    recover_searcher.open_wal(WAL_PATH);

    // 4. 验证结果
    // 预期：虽然崩溃过，但依然能查到 888 和 999，并且查不到 0
    print_result("State AFTER Recovery (Query [1, 2])", recover_searcher.query({1, 2}));

    std::cout << "\n>>> Test passed if State AFTER matches State BEFORE!" << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: ./test_dynamic <mode>\n"
                  << "  mode 0: Dummy Test\n"
                  << "  mode 1: Real Index Batch Test\n"
                  << "  mode 2: WAL Recovery Test\n";
        return 1;
    }

    int mode = std::stoi(argv[1]);

    if (mode == 0) run_dummy_test();
    else if (mode == 1) run_real_test(argc, argv);
    else if (mode == 2) run_wal_test();
    else std::cerr << "Invalid mode! Use 0, 1, or 2.\n";

    return 0;
}