// build_ivf_index.cpp
#include <iostream>
#include <string>
#include <chrono>
#include "InvertedIndex.hpp" // 复用之前的头文件

void print_usage(const char* prog_name) {
    std::cerr << "Usage: " << prog_name 
              << " <label_txt_path> <output_index_bin_path> <universal_tag_id>" << std::endl;
    std::cerr << "Example: " << prog_name << " labels.txt inverted_index.bin 11" << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        print_usage(argv[0]);
        return 1;
    }

    std::string input_path = argv[1];
    std::string output_path = argv[2];
    int universal_tag = std::stoi(argv[3]);

    std::cout << ">>> [Build] Config:" << std::endl;
    std::cout << "  Input: " << input_path << std::endl;
    std::cout << "  Output: " << output_path << std::endl;
    std::cout << "  Universal Tag ID: " << universal_tag << std::endl;

    try {
        auto start = std::chrono::high_resolution_clock::now();

        InvertedIndexBuilder builder;
        builder.build(input_path, output_path, universal_tag);

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        std::cout << ">>> [Build] Success! Time elapsed: " << duration << " ms." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << ">>> [Build] Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}