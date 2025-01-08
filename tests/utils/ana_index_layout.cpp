#include <chrono>
#include <string>
#include <utils.h>
#include <memory>
#include <set>
#include <vector>
#include <iostream>
#include <fstream>
#include <limits>
#include <cstring>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <utility>
#include <omp.h>
#include <cmath>
#include <mutex>
#include <queue>
#include <random>

#include "cached_io.h"
#include "pq_flash_index.h"
#include "aux_utils.h"

#define READ_SECTOR_LEN (size_t) 4096
#define READ_SECTOR_OFFSET(node_id) \
  ((_u64) node_id / nnodes_per_sector  + 1) * READ_SECTOR_LEN + ((_u64) node_id % nnodes_per_sector) * max_node_len;
#define INF 0xffffffff

const std::string partition_index_filename = "_tmp.index";






double calculateDuplicateRatio(const std::vector<unsigned>& vec) {
    // 使用 unordered_map 来统计每个元素出现的次数
    std::unordered_map<unsigned, int> freq_map;

    // 统计元素出现的次数
    for (unsigned num : vec) {
        freq_map[num]++;
    }

    // 计算重复元素的数量
    int duplicateCount = 0;
    for (const auto& entry : freq_map) {
        if (entry.second > 1) {
            duplicateCount += entry.second;  // 累加重复元素的总数量
        }
    }

    // 计算并返回重复元素的比例
    return static_cast<double>(freq_map.size()) / vec.size();
}

double calculateAverageOutDegree(const std::vector<size_t>& out_degree) {
    if (out_degree.empty()) {
        return 0.0;  // 防止空向量除零错误
    }
    
    // 计算总和
    size_t total = std::accumulate(out_degree.begin(), out_degree.end(), static_cast<size_t>(0));
    
    // 计算平均值
    return static_cast<double>(total) / out_degree.size();
}

std::vector<unsigned> getSortedKeysByValue(const std::unordered_map<unsigned, int>& freq_map) {
    // 将 unordered_map 转换为 vector
    std::vector<std::pair<unsigned, int>> freq_vector(freq_map.begin(), freq_map.end());
    
    // 根据第二个值（即 int）进行降序排序
    std::sort(freq_vector.begin(), freq_vector.end(), [](const std::pair<unsigned, int>& a, const std::pair<unsigned, int>& b) {
        return a.second > b.second;  // 降序排序
    });
    
    // 提取排序后的键
    std::vector<unsigned> sorted_keys;
    for (const auto& pair : freq_vector) {
        sorted_keys.push_back(pair.first);
    }

    return sorted_keys;
}

class Block{

public:
    std::vector<unsigned> out_block;
    double duplicateRatio;
    size_t out_degree;
    size_t maximum_block_key;
    size_t maximum_block_value;
    size_t maximum_block_value_10_sum;
    Block(){};
    ~Block(){
        this->out_block.clear();
    };

    void anaylze(){
        // 使用 unordered_map 来统计每个元素出现的次数
        std::unordered_map<unsigned, int> freq_map;

        // 统计元素出现的次数
        for (unsigned num : out_block) {
            freq_map[num]++;
        }

        // 计算重复元素的数量
        int duplicateCount = 0;
        for (const auto& entry : freq_map) {
            if (entry.second > 1) {
                duplicateCount += entry.second;  // 累加重复元素的总数量
            }
        }

        // 计算并返回重复元素的比例
        this->duplicateRatio =  static_cast<double>(freq_map.size()) / out_block.size();
        this->out_degree = freq_map.size();

        
        
        std::vector<std::pair<unsigned, int>> freq_vector(freq_map.begin(), freq_map.end());
    
        // 根据第二个值（即 int）进行降序排序
        std::sort(freq_vector.begin(), freq_vector.end(), [](const std::pair<unsigned, int>& a, const std::pair<unsigned, int>& b) {
            return a.second > b.second;  // 降序排序
        });
        
        // 提取排序后的键
        std::vector<unsigned> sorted_keys;
        for (const auto& pair : freq_vector) {
            sorted_keys.push_back(pair.first);
        }

        // 提取排序后的键
        std::vector<unsigned> sorted_values;
        for (const auto& pair : freq_vector) {
            sorted_values.push_back(pair.second);
        }
        this->maximum_block_key = sorted_keys[0];
        this->maximum_block_value = sorted_values[0];


        this->maximum_block_value_10_sum = 0;
        for(size_t i = 0; i<10;i++){
            this->maximum_block_value_10_sum += sorted_values[i];
        }

    }
};


void read_ssd_layout(const char* layout_file_name){
    _u64                                _block_num;
    std::vector<Block>  block_affinity_graph;
    

    diskann::cout<<"layout file name : "<<layout_file_name<<std::endl;


    std::ifstream part(layout_file_name);
    part.read((char*) &_block_num, sizeof(_u64));
    block_affinity_graph.resize(_block_num);
    for(size_t i = 0 ; i < _block_num;i++){
        if (i % 3000000 == 0) {
            diskann::cout << "read has done " << (float) i / _block_num
                            << std::endl;
            diskann::cout.flush();
        }
        unsigned s;
        part.read((char*) &s, sizeof(unsigned));
        block_affinity_graph[i].out_block.resize(s);
        part.read((char*) block_affinity_graph[i].out_block.data(), sizeof(unsigned) * s);
    }
    part.close();
    
    std::vector<double> ratio(_block_num);
    std::vector<size_t> out_degree(_block_num);
    std::vector<size_t> max_out_num(_block_num);
    std::vector<size_t> max_out_num_top10_sum(_block_num);
    // 释放内存
    #pragma omp parallel for
    for(size_t i = 0 ; i < _block_num;i++){
        // ratio[i] = calculateDuplicateRatio(block_affinity_graph[i]);
        block_affinity_graph[i].anaylze();
        ratio[i] = block_affinity_graph[i].duplicateRatio;
        out_degree[i] = block_affinity_graph[i].out_degree;
        max_out_num[i] = block_affinity_graph[i].maximum_block_value;
        max_out_num_top10_sum[i] = block_affinity_graph[i].maximum_block_value_10_sum;
        // if(i<10){
        //     diskann::cout<<ratio<<std::endl;
        // }
        // block_affinity_graph[i].clear();
    }
    block_affinity_graph.clear();



    size_t p10 = 0,p50 = 0,p80 = 0,p90 = 0;
    for(size_t i = 0 ; i < _block_num;i++){
        double tmp = ratio[i];
        // if(i<10){
        //     diskann::cout<<tmp<<std::endl;
        //     diskann::cout<<"max conn num = "<<max_out_num[i]<<std::endl;
        //     std::cout << "Max out-block_num top 10 sum : " << max_out_num_top10_sum[i] << std::endl;
        // }
        
        if(tmp < 0.1){
            p10++;
        }
        if(tmp < 0.5){
            p50++;
        }
        if(tmp < 0.8){
            p80++;
        }
        if(tmp < 0.9){
            p90++;
        }
    }

    diskann::cout<<" p10 = "<< static_cast<double>(p10)/_block_num<<std::endl;
    diskann::cout<<" p50 = "<< static_cast<double>(p50)/_block_num<<std::endl;
    diskann::cout<<" p80 = "<< static_cast<double>(p80)/_block_num<<std::endl;
    diskann::cout<<" p90 = "<< static_cast<double>(p90)/_block_num<<std::endl;

    double avg = calculateAverageOutDegree(out_degree);
    std::cout << "Average out-degree: " << avg << std::endl;
    avg = calculateAverageOutDegree(max_out_num);
    std::cout << "Average out-block_num: " << avg << std::endl;
    avg = calculateAverageOutDegree(max_out_num_top10_sum);
    std::cout << "Average out-block_num top 10 sum : " << avg << std::endl;

}

int main(int argc, char** argv){
  std::string layout_file_name1 = "bigann_100m_M200_R80_L100_B5/result/result_outdegree_PageSearch0.bin";
  std::string layout_file_name2 = "bigann_100m_M200_R80_L100_B5/result/result_outdegree_PageSearch1.bin";
  read_ssd_layout(layout_file_name1.c_str());
  read_ssd_layout(layout_file_name2.c_str());
  return 0;
}