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

        // 核心逻辑：如果是通用向量，只放入通用 Bitmap，否则放入对应的 Tag Bitmap
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
    for (const auto& kv : _inverted_index) {
        int tag_id = kv.first;
        outfile.write(reinterpret_cast<const char*>(&tag_id), sizeof(tag_id));
        write_bitmap(outfile, kv.second);
    }

    outfile.close();
    std::cout << "[Build] Index saved to " << output_path << std::endl;
}

void InvertedIndexBuilder::write_bitmap(std::ofstream& out, const Roaring& rb) {
    size_t size = rb.getSizeInBytes();
    uint32_t size32 = static_cast<uint32_t>(size);
    out.write(reinterpret_cast<const char*>(&size32), sizeof(size32));
    std::vector<char> buffer(size);
    rb.write(buffer.data());
    out.write(buffer.data(), size);
}


// ================= Searcher 核心实现 =================

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

    // 2. 读取通用 Bitmap (落入 Disk 层)
    _disk_universal_bitmap = read_bitmap(infile);

    // 3. 读取 Map 大小
    uint32_t map_size;
    infile.read(reinterpret_cast<char*>(&map_size), sizeof(map_size));

    // 4. 读取所有 Tag Bitmap (落入 Disk 层)
    _disk_index.clear();
    for (uint32_t i = 0; i < map_size; ++i) {
        int tag_id;
        infile.read(reinterpret_cast<char*>(&tag_id), sizeof(tag_id));
        _disk_index[tag_id] = read_bitmap(infile);
    }

    // 5. 初始化/清空 Mem 层和墓碑
    _mem_index.clear();
    _mem_universal_bitmap = Roaring();
    _deleted_bitmap = Roaring();

    std::cout << "[Load] Index loaded. Disk Universal count: " << _disk_universal_bitmap.cardinality()
              << ", Disk Unique tags: " << _disk_index.size() << std::endl;
}

Roaring InvertedIndexSearcher::read_bitmap(std::ifstream& in) {
    uint32_t size;
    in.read(reinterpret_cast<char*>(&size), sizeof(size));
    std::vector<char> buffer(size);
    in.read(buffer.data(), size);
    return Roaring::read(buffer.data());
}

void InvertedIndexSearcher::write_bitmap(std::ofstream& out, const Roaring& rb) {
    size_t size = rb.getSizeInBytes();
    uint32_t size32 = static_cast<uint32_t>(size);
    out.write(reinterpret_cast<const char*>(&size32), sizeof(size32));
    std::vector<char> buffer(size);
    rb.write(buffer.data());
    out.write(buffer.data(), size);
}

Roaring InvertedIndexSearcher::query_bitmap(const std::vector<int>& tags) {
    Roaring result;

    if (tags.empty()) {
        // 合并磁盘与内存的通用集合，剔除已删除的
        result = _disk_universal_bitmap | _mem_universal_bitmap;
        result -= _deleted_bitmap;
        return result;
    }

    bool first = true;
    for (int tag : tags) {
        // 1. 获取包含该 Tag 的完整位图：磁盘 | 内存
        Roaring current_tag_bitmap;

        auto it_disk = _disk_index.find(tag);
        if (it_disk != _disk_index.end()) current_tag_bitmap |= it_disk->second;

        auto it_mem = _mem_index.find(tag);
        if (it_mem != _mem_index.end()) current_tag_bitmap |= it_mem->second;

        // 2. 求交集 (AND)
        if (first) {
            result = current_tag_bitmap;
            first = false;
        } else {
            result &= current_tag_bitmap;
        }
    }

    // 3. 合并通用集合 (Universal Set: 磁盘 + 内存)
    result |= (_disk_universal_bitmap | _mem_universal_bitmap);

    // 4. 剔除所有已删除的向量
    result -= _deleted_bitmap;

    return result;
}

std::vector<uint32_t> InvertedIndexSearcher::query(const std::vector<int>& tags) {
    Roaring res_bitmap = query_bitmap(tags);
    std::vector<uint32_t> out_ids;
    out_ids.reserve(res_bitmap.cardinality());
    for(uint32_t id : res_bitmap) {
        out_ids.push_back(id);
    }
    return out_ids;
}


// ================= WAL 与内存分离逻辑 =================

void InvertedIndexSearcher::_apply_insert_to_mem(uint32_t vid, const std::vector<int>& tags) {
    bool is_universal = false;
    for (int tag : tags) {
        if (tag == _universal_tag) {
            is_universal = true;
            break;
        }
    }

    if (is_universal) {
        _mem_universal_bitmap.add(vid);
    } else {
        for (int tag : tags) {
            _mem_index[tag].add(vid);
        }
    }
    // 从墓碑中移除（如果以前被删过，现在属于重新插入）
    _deleted_bitmap.remove(vid);
}

void InvertedIndexSearcher::_apply_remove_to_mem(uint32_t vid) {
    _deleted_bitmap.add(vid);
}


// ================= WAL 磁盘操作 =================

void InvertedIndexSearcher::open_wal(const std::string& wal_path) {
    _wal_path = wal_path;
    // std::ios::app 保证所有的写操作都追加在文件末尾
    _wal_stream.open(wal_path, std::ios::binary | std::ios::app);
    if (!_wal_stream.is_open()) {
        throw std::runtime_error("Cannot open WAL file: " + wal_path);
    }
}

void InvertedIndexSearcher::close_wal() {
    if (_wal_stream.is_open()) {
        _wal_stream.flush();
        _wal_stream.close();
    }
}

void InvertedIndexSearcher::_append_wal_insert(uint32_t vid, const std::vector<int>& tags) {
    if (!_wal_stream.is_open()) return;
    WalOp op = WalOp::INSERT;
    _wal_stream.write(reinterpret_cast<const char*>(&op), sizeof(op));
    _wal_stream.write(reinterpret_cast<const char*>(&vid), sizeof(vid));

    uint32_t count = tags.size();
    _wal_stream.write(reinterpret_cast<const char*>(&count), sizeof(count));
    for (int tag : tags) {
        _wal_stream.write(reinterpret_cast<const char*>(&tag), sizeof(tag));
    }
    _wal_stream.flush(); // 强制刷入磁盘保证持久化
}

void InvertedIndexSearcher::_append_wal_remove(uint32_t vid) {
    if (!_wal_stream.is_open()) return;
    WalOp op = WalOp::REMOVE;
    _wal_stream.write(reinterpret_cast<const char*>(&op), sizeof(op));
    _wal_stream.write(reinterpret_cast<const char*>(&vid), sizeof(vid));
    _wal_stream.flush();
}

void InvertedIndexSearcher::recover_from_wal(const std::string& wal_path) {
    std::ifstream in(wal_path, std::ios::binary);
    if (!in.is_open()) {
        return; // 没有 WAL 文件是正常的，直接返回
    }

    std::cout << "[WAL] Recovering memory state from WAL..." << std::endl;
    int recover_insert_cnt = 0;
    int recover_remove_cnt = 0;

    while (in.peek() != EOF) {
        WalOp op;
        in.read(reinterpret_cast<char*>(&op), sizeof(op));
        if (in.eof()) break;

        if (op == WalOp::INSERT) {
            uint32_t vid, count;
            in.read(reinterpret_cast<char*>(&vid), sizeof(vid));
            in.read(reinterpret_cast<char*>(&count), sizeof(count));
            std::vector<int> tags(count);
            for (uint32_t i = 0; i < count; ++i) {
                in.read(reinterpret_cast<char*>(&tags[i]), sizeof(int));
            }
            _apply_insert_to_mem(vid, tags);
            recover_insert_cnt++;
        } else if (op == WalOp::REMOVE) {
            uint32_t vid;
            in.read(reinterpret_cast<char*>(&vid), sizeof(vid));
            _apply_remove_to_mem(vid);
            recover_remove_cnt++;
        }
    }
    std::cout << "[WAL] Recovery done! Replayed " << recover_insert_cnt
              << " inserts and " << recover_remove_cnt << " removes." << std::endl;
}


// ================= 对外暴露的动态操作 API =================

void InvertedIndexSearcher::insert(uint32_t vid, const std::vector<int>& tags) {
    _append_wal_insert(vid, tags);
    _apply_insert_to_mem(vid, tags);
}

void InvertedIndexSearcher::remove(uint32_t vid) {
    _append_wal_remove(vid);
    _apply_remove_to_mem(vid);
}


// ================= 异步落盘 Merge =================

void InvertedIndexSearcher::merge(const std::string& new_index_path) {
    std::cout << "[Merge] Starting index compaction..." << std::endl;

    // 1. 合并通用 Bitmap
    Roaring new_universal = (_disk_universal_bitmap | _mem_universal_bitmap) - _deleted_bitmap;
    new_universal.runOptimize();

    // 2. 收集所有存在的 Tag (Disk + Mem)
    std::vector<int> all_tags;
    for (const auto& kv : _disk_index) all_tags.push_back(kv.first);
    for (const auto& kv : _mem_index) all_tags.push_back(kv.first);
    std::sort(all_tags.begin(), all_tags.end());
    all_tags.erase(std::unique(all_tags.begin(), all_tags.end()), all_tags.end());

    // 3. 合并具体的 Tag Bitmap
    std::map<int, Roaring> new_index;
    for (int tag : all_tags) {
        Roaring tb;
        if (_disk_index.count(tag)) tb |= _disk_index[tag];
        if (_mem_index.count(tag))  tb |= _mem_index[tag];

        tb -= _deleted_bitmap;

        // 只有不为空的才写盘
        if (!tb.isEmpty()) {
            tb.runOptimize();
            new_index[tag] = std::move(tb);
        }
    }

    // 4. 序列化落盘
    std::ofstream outfile(new_index_path, std::ios::binary);
    if (!outfile.is_open()) {
        throw std::runtime_error("Cannot create merged index file: " + new_index_path);
    }

    outfile.write(reinterpret_cast<const char*>(&INDEX_MAGIC), sizeof(INDEX_MAGIC));
    write_bitmap(outfile, new_universal);

    uint32_t map_size = new_index.size();
    outfile.write(reinterpret_cast<const char*>(&map_size), sizeof(map_size));

    for (const auto& kv : new_index) {
        int tag_id = kv.first;
        outfile.write(reinterpret_cast<const char*>(&tag_id), sizeof(tag_id));
        write_bitmap(outfile, kv.second);
    }

    outfile.close();
    std::cout << "[Merge] Compaction successful. Saved to " << new_index_path << std::endl;

    // 5. 清理 WAL 记录
    // Merge 完成后，清空当前的 WAL（因为内存数据已经全部写回最新的基础磁盘索引了）
    if (_wal_stream.is_open()) {
        _wal_stream.close();
        // 使用 trunc 模式清空日志
        std::ofstream truncate_wal(_wal_path, std::ios::trunc | std::ios::binary);
        truncate_wal.close();
        // 重新开启全新的日志
        open_wal(_wal_path);
        std::cout << "[WAL] Log file truncated after merge." << std::endl;
    }
}
