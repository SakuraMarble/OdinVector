#pragma once

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <unistd.h>
#include <vector>

namespace v2 {

struct LazyWalEntry {
  char op = '\0';  // I, D, or M.
  uint64_t tag = 0;
  uint64_t row = 0;
};

class LazyWal {
 public:
  explicit LazyWal(std::string path) : path_(std::move(path)) {}

  const std::string &path() const {
    return path_;
  }

  void append_batch(const std::vector<LazyWalEntry> &entries) const {
    if (entries.empty()) {
      return;
    }
    int fd = ::open(path_.c_str(), O_WRONLY | O_CREAT | O_APPEND, 0644);
    if (fd < 0) {
      throw std::runtime_error("open wal failed: " + path_ + ", err=" + std::string(std::strerror(errno)));
    }
    try {
      for (const auto &entry : entries) {
        append_entry(fd, entry);
      }
    } catch (...) {
      ::close(fd);
      throw;
    }
    sync_and_close(fd);
  }

  void append_insert(uint64_t tag, uint64_t row) const {
    append_batch(std::vector<LazyWalEntry>{{'I', tag, row}});
  }

  void append_delete(uint64_t tag) const {
    append_batch(std::vector<LazyWalEntry>{{'D', tag, 0}});
  }

  void append_merge_marker() const {
    append_batch(std::vector<LazyWalEntry>{{'M', 0, 0}});
  }

  void clear() const {
    int fd = ::open(path_.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
      throw std::runtime_error("clear wal failed: " + path_ + ", err=" + std::string(std::strerror(errno)));
    }
    sync_and_close(fd);
  }

  std::vector<LazyWalEntry> load_all() const {
    std::vector<LazyWalEntry> out;
    std::ifstream in(path_);
    if (!in.is_open()) {
      return out;
    }

    std::string line;
    while (std::getline(in, line)) {
      if (line.empty()) {
        continue;
      }
      std::stringstream ss(line);
      char op = '\0';
      ss >> op;
      if (op == 'I') {
        uint64_t tag = 0;
        uint64_t row = 0;
        if (ss >> tag >> row) {
          out.push_back(LazyWalEntry{'I', tag, row});
        }
      } else if (op == 'D') {
        uint64_t tag = 0;
        if (ss >> tag) {
          out.push_back(LazyWalEntry{'D', tag, 0});
        }
      } else if (op == 'M') {
        out.push_back(LazyWalEntry{'M', 0, 0});
      }
    }
    return out;
  }

 private:
  static void write_all(int fd, const std::string &line, const std::string &path) {
    const char *p = line.data();
    size_t left = line.size();
    while (left > 0) {
      ssize_t n = ::write(fd, p, left);
      if (n <= 0) {
        throw std::runtime_error("write wal failed: " + path + ", err=" + std::string(std::strerror(errno)));
      }
      p += n;
      left -= static_cast<size_t>(n);
    }
  }

  void append_entry(int fd, const LazyWalEntry &entry) const {
    if (entry.op == 'I') {
      write_all(fd, "I " + std::to_string(entry.tag) + " " + std::to_string(entry.row) + "\n", path_);
    } else if (entry.op == 'D') {
      write_all(fd, "D " + std::to_string(entry.tag) + "\n", path_);
    } else if (entry.op == 'M') {
      write_all(fd, "M\n", path_);
    }
  }

  void sync_and_close(int fd) const {
    if (::fsync(fd) != 0) {
      ::close(fd);
      throw std::runtime_error("fsync wal failed: " + path_ + ", err=" + std::string(std::strerror(errno)));
    }
    ::close(fd);
  }

  std::string path_;
};

struct LazyOpWalEntry {
  char type = '\0';  // I or D.
  uint64_t tag = 0;
  uint64_t row = 0;
};

class LazyOpWal {
 public:
  explicit LazyOpWal(std::string path) : wal_(std::move(path)) {}

  const std::string &path() const {
    return wal_.path();
  }

  void clear() const {
    wal_.clear();
  }

  void append_insert(uint64_t tag, uint64_t row) const {
    wal_.append_insert(tag, row);
  }

  void append_delete(uint64_t tag) const {
    wal_.append_delete(tag);
  }

  std::vector<LazyOpWalEntry> load_entries() const {
    std::vector<LazyOpWalEntry> out;
    for (const auto &entry : wal_.load_all()) {
      if (entry.op == 'I') {
        out.push_back(LazyOpWalEntry{'I', entry.tag, entry.row});
      } else if (entry.op == 'D') {
        out.push_back(LazyOpWalEntry{'D', entry.tag, 0});
      }
    }
    return out;
  }

 private:
  LazyWal wal_;
};

}  // namespace v2
