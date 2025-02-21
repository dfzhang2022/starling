#include <unordered_map>
#include <vector>

// #include "aligned_file_reader.h"
#include "pq_flash_index.h"

using namespace diskann;

// Function to count block_id occurrences
std::unordered_map<uint64_t, int> countBlockIdOccurrences(const std::vector<QueryIO>& ios) {
    std::unordered_map<uint64_t, int> blockIdCount;
    for (const auto& io : ios) {
        for (const auto& read : io.aligned_read_vec) {
            blockIdCount[read.block_id]++;
        }
    }
    return blockIdCount;
  }

  // Function to calculate the total number of block_id occurrences in a QueryIO
  int calculateTotalBlockIdOccurrences(const QueryIO& io, const std::unordered_map<uint64_t, int>& blockIdCount) {
    int total = 0;
    for (const auto& read : io.aligned_read_vec) {
        total += blockIdCount.at(read.block_id);
    }
    return total;
  }

  // Function to sort QueryIO based on block_id occurrences
  void sortQueryIOByBlockIdOccurrences(std::vector<QueryIO>& ios) {
    auto blockIdCount = countBlockIdOccurrences(ios);
    std::sort(ios.begin(), ios.end(), [&blockIdCount](const QueryIO& a, const QueryIO& b) {
        return calculateTotalBlockIdOccurrences(a, blockIdCount) < calculateTotalBlockIdOccurrences(b, blockIdCount);
    });
  }