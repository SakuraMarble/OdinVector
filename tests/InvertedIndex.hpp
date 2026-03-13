#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <map>
#include <stdexcept>
#include <roaring/roaring.hh> // 确保安装了 CRoaring 并包含此头文件

// 二进制文件魔数，用于校验
const uint32_t INDEX_MAGIC = 0xAA55AA55;

class InvertedIndexBuilder {
public:
    // 构建索引
    // input_path: 标签文件路径
    // output_path: 输出二进制索引路径
    // universal_tag: 通用标签ID
    void build(const std::string& input_path, const std::string& output_path, int universal_tag);

private:
    std::map<int, roaring::Roaring> _inverted_index;
    roaring::Roaring _universal_bitmap;
    
    // 辅助：写入 Roaring Bitmap 到流
    void write_bitmap(std::ofstream& out, const roaring::Roaring& rb);
};

class InvertedIndexSearcher {
public:
    // 加载索引
    void load(const std::string& index_path);

    // 查询求交集
    // tags: 查询的标签列表
    // 返回: 符合条件的向量ID列表
    std::vector<uint32_t> query(const std::vector<int>& tags);

    // 获取原始 Bitmap (用于后续可能的逻辑)
    roaring::Roaring query_bitmap(const std::vector<int>& tags);

private:
    std::map<int, roaring::Roaring> _inverted_index;
    roaring::Roaring _universal_bitmap;
    
    // 辅助：从流读取 Roaring Bitmap
    roaring::Roaring read_bitmap(std::ifstream& in);
};