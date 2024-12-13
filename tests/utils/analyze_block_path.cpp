#include <string>
#include <iostream>
#include <fstream>
#include <cassert>
#include <iomanip>

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <random>
#include <limits>
#include <cstring>
#include <queue>
#include <omp.h>
#include <boost/program_options.hpp>
#include <unordered_map>


#include <stdlib.h>

#include "utils.h"


namespace po = boost::program_options;


struct BlockVisited {
    uint64_t block_id;
    float timestamp; // set by who create this record. in us

    BlockVisited() : block_id(0), timestamp(0.0f) {}

    BlockVisited(uint64_t block_id_in, float timestamp_in)
        : block_id(block_id_in), timestamp(timestamp_in) {}
};

// 从文件读取 vector<vector<BlockVisited>>
std::vector<std::vector<BlockVisited>> readFromFile(const std::string &filename) {
    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs) {
        std::cerr << "Failed to open file for reading: " << filename << std::endl;
        return {};
    }

    std::vector<std::vector<BlockVisited>> data;

    size_t outer_size;
    ifs.read(reinterpret_cast<char*>(&outer_size), sizeof(size_t));
    data.resize(outer_size);

    for (size_t i = 0; i < outer_size; ++i) {
        size_t inner_size;
        ifs.read(reinterpret_cast<char*>(&inner_size), sizeof(size_t));
        data[i].clear();
        for (size_t j = 0; j < inner_size; ++j) {
            uint64_t block_id;
            float timestamp;

            ifs.read(reinterpret_cast<char*>(&block_id), sizeof(uint64_t));
            ifs.read(reinterpret_cast<char*>(&timestamp), sizeof(float));
            data[i].push_back(BlockVisited(block_id, timestamp));
            
        }
    }

    ifs.close();
    return data;
}

// 统计每 0.5 秒窗口的块数量和重复块占比
void analyzeData(const std::vector<std::vector<BlockVisited>> &data,float time_window_size_in_s) {
    std::map<int, std::vector<uint64_t>> time_window_blocks; // 时间窗口 -> 块ID列表

    // 将数据分组到时间窗口
    for (const auto &inner_vec : data) {
        for (const auto &block : inner_vec) {
            int window_index = static_cast<int>(block.timestamp / (time_window_size_in_s*1000*1000)); // 时间窗口索引
            time_window_blocks[window_index].push_back(block.block_id);
        }
    }

    // 输出分析结果
    std::cout << std::fixed << std::setprecision(2);
    std::cout << std::setw(20) << "Time Window (s) "<<"| Total Blocks | Duplicate Blocks | Duplicate Ratio (%)\n";
    std::cout << "-----------------------------------------------------------------------\n";

    for (const auto &[window_index, blocks] : time_window_blocks) {
        size_t total_blocks = blocks.size();

        // 统计重复块数量
        std::unordered_map<uint64_t, int> block_counts;
        for (const auto &block_id : blocks) {
            block_counts[block_id]++;
        }

        size_t duplicate_blocks = 0;
        for (const auto &[block_id, count] : block_counts) {
            if (count > 1) {
                duplicate_blocks += (count - 1);
            }
        }

        double duplicate_ratio = (total_blocks > 0)
                                     ? (static_cast<double>(duplicate_blocks) / total_blocks) * 100
                                     : 0.0;

        // 输出结果
        std::cout << std::setw(8) << window_index * time_window_size_in_s << " - "
                  << std::setw(8) << (window_index + 1) * time_window_size_in_s << " | "
                  << std::setw(12) << total_blocks << " | "
                  << std::setw(16) << duplicate_blocks << " | "
                  << std::setw(18) << duplicate_ratio << "\n";
    }
}

int main(int argc, char **argv) {
  std::string data_type, dist_fn, base_file, query_file, gt_file, tags_file;
  std::string block_path_file;
  float time_window_size;

  try {
    po::options_description desc{"Arguments"};

    desc.add_options()("help,h", "Print information on arguments");

    desc.add_options()("block_path_file",
                       po::value<std::string>(&block_path_file)->required(),
                       "File containing the block paths in binary format");
    desc.add_options()("time_window_size",
                       po::value<float>(&time_window_size)->required(),
                       "Time window size for calculating duplicating ratio.");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help")) {
      std::cout << desc;
      return 0;
    }
    po::notify(vm);
  } catch (const std::exception &ex) {
    std::cerr << ex.what() << '\n';
    return -1;
  }
  auto vec = readFromFile(block_path_file);
  analyzeData(vec,time_window_size);

vec.clear();


}

