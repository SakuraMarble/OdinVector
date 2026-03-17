#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <map>
#include <stdexcept>
#include <algorithm>
#include <roaring/roaring.hh> 

const uint32_t INDEX_MAGIC = 0xAA55AA55;

// 定义日志操作类型
enum class WalOp : uint8_t {
    INSERT = 1,
    REMOVE = 2
};

class InvertedIndexBuilder {
public:
    void build(const std::string& input_path, const std::string& output_path, int universal_tag);
private:
    std::map<int, roaring::Roaring> _inverted_index;
    roaring::Roaring _universal_bitmap;
    void write_bitmap(std::ofstream& out, const roaring::Roaring& rb);
};

class InvertedIndexSearcher {
public:
    ~InvertedIndexSearcher() { close_wal(); }

    void load(const std::string& index_path);
    void set_universal_tag(int tag) { _universal_tag = tag; }

    // ================= WAL (预写日志) 接口 =================
    // 开启预写日志文件（追加模式）
    void open_wal(const std::string& wal_path);
    // 从日志中重放数据恢复内存状态
    void recover_from_wal(const std::string& wal_path);
    // 关闭日志
    void close_wal();

    // ================= 动态操作接口 =================
    void insert(uint32_t vid, const std::vector<int>& tags);
    void remove(uint32_t vid);
    void merge(const std::string& new_index_path);

    // ================= 检索接口 =================
    std::vector<uint32_t> query(const std::vector<int>& tags);
    roaring::Roaring query_bitmap(const std::vector<int>& tags);

private:
    std::map<int, roaring::Roaring> _disk_index;
    roaring::Roaring _disk_universal_bitmap;

    std::map<int, roaring::Roaring> _mem_index;
    roaring::Roaring _mem_universal_bitmap;
    roaring::Roaring _deleted_bitmap;

    int _universal_tag = -1;
    
    // WAL 状态
    std::string _wal_path;
    std::ofstream _wal_stream;

    roaring::Roaring read_bitmap(std::ifstream& in);
    void write_bitmap(std::ofstream& out, const roaring::Roaring& rb);

    // 真正的内存操作实现（分离是为了恢复数据时不写 WAL）
    void _apply_insert_to_mem(uint32_t vid, const std::vector<int>& tags);
    void _apply_remove_to_mem(uint32_t vid);

    // 写入 WAL 日志记录
    void _append_wal_insert(uint32_t vid, const std::vector<int>& tags);
    void _append_wal_remove(uint32_t vid);
};