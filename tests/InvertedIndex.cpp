#include "InvertedIndex.hpp"

// ================= Builder 实现 =================

void InvertedIndexBuilder::build(const std::string& input_path, const std::string& output_path, int universal_tag) {
    std::ifstream infile(input_path);
    if (!infile.is_open()) {
        throw std::runtime_error("Cannot open input label file: " + input_path);
    }

    std::string line;
    uint32_t vector_id = 0; // 假设行号即为 ID，从 0 开始

    std::cout << "[Build] Starting to parse label file..." << std::endl;

    while (std::getline(infile, line)) {
        if (line.empty()) {
            vector_id++;
            continue;
        }

        std::stringstream ss(line);
        std::string segment;
        std::vector<int> tags;
        bool is_universal = false;

        // 解析 CSV
        while (std::getline(ss, segment, ',')) {
            if (segment.empty() || segment == "\r") continue; // 处理行尾可能的空字符
            try {
                int tag = std::stoi(segment);
                if (tag == universal_tag) {
                    is_universal = true;
                }
                tags.push_back(tag);
            } catch (...) {
                continue; // 忽略非数字
            }
        }

        // 核心逻辑：如果是通用向量，只放入通用 Bitmap
        // 否则，放入对应的 Tag Bitmap
        if (is_universal) {
            _universal_bitmap.add(vector_id);
        } else {
            for (int tag : tags) {
                _inverted_index[tag].add(vector_id);
            }
        }

        vector_id++;
        if (vector_id % 100000 == 0) std::cout << "[Build] Processed " << vector_id << " vectors..." << std::endl;
    }
    
    // Run-Length Encoding 优化（压缩）
    _universal_bitmap.runOptimize();
    for (auto& kv : _inverted_index) {
        kv.second.runOptimize();
    }

    std::cout << "[Build] Parsing done. Writing to disk..." << std::endl;

    // --- 序列化到磁盘 ---
    std::ofstream outfile(output_path, std::ios::binary);
    if (!outfile.is_open()) {
        throw std::runtime_error("Cannot create output file: " + output_path);
    }

    // 1. 写入 Header (Magic)
    outfile.write(reinterpret_cast<const char*>(&INDEX_MAGIC), sizeof(INDEX_MAGIC));

    // 2. 写入通用 Bitmap
    write_bitmap(outfile, _universal_bitmap);

    // 3. 写入 Tag 数量
    uint32_t map_size = _inverted_index.size();
    outfile.write(reinterpret_cast<const char*>(&map_size), sizeof(map_size));

    // 4. 写入具体的 Tag -> Bitmap
    // 格式：[TagID] [BitmapSize] [BitmapData]
    for (const auto& kv : _inverted_index) {
        int tag_id = kv.first;
        outfile.write(reinterpret_cast<const char*>(&tag_id), sizeof(tag_id));
        write_bitmap(outfile, kv.second);
    }

    outfile.close();
    std::cout << "[Build] Index saved to " << output_path << std::endl;
}

void InvertedIndexBuilder::write_bitmap(std::ofstream& out, const roaring::Roaring& rb) {
    // 获取序列化大小（portable format）
    size_t size = rb.getSizeInBytes(); 
    uint32_t size32 = static_cast<uint32_t>(size);
    
    // 写入大小
    out.write(reinterpret_cast<const char*>(&size32), sizeof(size32));
    
    // 写入数据
    std::vector<char> buffer(size);
    rb.write(buffer.data());
    out.write(buffer.data(), size);
}

// ================= Searcher 实现 =================

void InvertedIndexSearcher::load(const std::string& index_path) {
    std::ifstream infile(index_path, std::ios::binary);
    if (!infile.is_open()) {
        throw std::runtime_error("Cannot open index file: " + index_path);
    }

    // 1. 校验 Magic
    uint32_t magic;
    infile.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    if (magic != INDEX_MAGIC) {
        throw std::runtime_error("Invalid index file format (Magic mismatch)");
    }

    // 2. 读取通用 Bitmap
    _universal_bitmap = read_bitmap(infile);

    // 3. 读取 Map 大小
    uint32_t map_size;
    infile.read(reinterpret_cast<char*>(&map_size), sizeof(map_size));

    // 4. 读取所有 Tag Bitmap
    _inverted_index.clear();
    for (uint32_t i = 0; i < map_size; ++i) {
        int tag_id;
        infile.read(reinterpret_cast<char*>(&tag_id), sizeof(tag_id));
        roaring::Roaring rb = read_bitmap(infile);
        _inverted_index[tag_id] = std::move(rb);
    }

    std::cout << "[Load] Index loaded. Universal count: " << _universal_bitmap.cardinality() 
              << ", Total unique tags: " << _inverted_index.size() << std::endl;
}

roaring::Roaring InvertedIndexSearcher::read_bitmap(std::ifstream& in) {
    uint32_t size;
    in.read(reinterpret_cast<char*>(&size), sizeof(size));
    
    std::vector<char> buffer(size);
    in.read(buffer.data(), size);
    
    // 从 buffer 反序列化
    return roaring::Roaring::read(buffer.data());
}

roaring::Roaring InvertedIndexSearcher::query_bitmap(const std::vector<int>& tags) {
    if (tags.empty()) {
        // 如果没有查询标签，是否返回所有数据？
        // 通常逻辑是返回空，或者只返回通用数据。这里假设只返回通用数据。
        return _universal_bitmap;
    }

    // 1. 初始化结果为第一个标签的 Bitmap
    // 注意：如果标签在索引里不存在，它就是一个空的 Bitmap
    roaring::Roaring result;
    bool first = true;

    for (int tag : tags) {
        auto it = _inverted_index.find(tag);
        if (it == _inverted_index.end()) {
            // 如果查的标签不存在，这一个分支的交集必然为空
            // 但不能直接返回空，因为还要最后 OR 通用集
            result = roaring::Roaring(); // Empty
            first = false; 
            break; // A & Empty & ... = Empty
        }

        if (first) {
            result = it->second; // Copy
            first = false;
        } else {
            result &= it->second; // Intersection (AND)
        }
    }

    // 2. 最后合并通用集合 (Universal Set)
    // 逻辑：(TagA & TagB) | Universal
    result |= _universal_bitmap;

    return result;
}

std::vector<uint32_t> InvertedIndexSearcher::query(const std::vector<int>& tags) {
    roaring::Roaring res_bitmap = query_bitmap(tags);
    
    // 转换为 vector
    std::vector<uint32_t> out_ids;
    out_ids.reserve(res_bitmap.cardinality());
    
    // 迭代 bitmap
    for(uint32_t id : res_bitmap) {
        out_ids.push_back(id);
    }
    
    return out_ids;
}