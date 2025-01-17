//
// Created by Songlin Wu on 2022/6/30.
//
#include <chrono>
#include <string>
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
#include <set>
#include <algorithm>
#include <utility>
#include <omp.h>
#include <cmath>
#include <mutex>
#include <queue>
#include <random>

#include <boost/program_options.hpp>
#include <omp.h>


#include <atomic>
#include <cstring>
#include <iomanip>
#include <omp.h>
#include <pq_flash_index.h>
#include <set>
#include <string.h>
#include <time.h>
#include <boost/program_options.hpp>
#include "aux_utils.h"
#include "utils.h"

// #include "linux_aligned_file_reader.h"

namespace po = boost::program_options;

#define READ_SECTOR_LEN (size_t) 4096
#define READ_SECTOR_OFFSET(node_id) \
  ((_u64) node_id / nnodes_per_sector  + 1) * READ_SECTOR_LEN + ((_u64) node_id % nnodes_per_sector) * max_node_len;
#define INF 0xffffffff

// #define disk_bytes_per_point 128

const std::string partition_index_filename = "_tmp.index";
const std::string affinity_filename = "_affinity.bin";

// 辅助函数，用于将无符号整数写入到文件流中
void writeUnsigned(std::ofstream& outFile, unsigned num) {
    outFile.write(reinterpret_cast<const char*>(&num), sizeof(unsigned));
}

// 辅助函数，用于从文件流中读取无符号整数
unsigned readUnsigned(std::ifstream& inFile) {
    unsigned num;
    inFile.read(reinterpret_cast<char*>(&num), sizeof(unsigned));
    return num;
}

// 辅助函数，用于将一个std::set<unsigned>写入到文件流中（序列化集合）
void writeSet(std::ofstream& outFile, const std::set<unsigned>& s) {
    size_t size = s.size();
    writeUnsigned(outFile, size);
    for (const auto& element : s) {
        writeUnsigned(outFile, element);
    }
}

// 辅助函数，用于从文件流中读取并还原一个std::set<unsigned>（反序列化集合）
std::set<unsigned> readSet(std::ifstream& inFile) {
    size_t size = readUnsigned(inFile);
    std::set<unsigned> result;
    for (size_t i = 0; i < size; ++i) {
        unsigned element = readUnsigned(inFile);
        result.insert(element);
    }
    return result;
}

// 函数用于将整个unordered_map结构写入到文件流中（序列化映射）
void writeUnorderedMap(std::ofstream& outFile, const std::unordered_map<unsigned, std::set<unsigned>>& mapping) {
    size_t size = mapping.size();
    writeUnsigned(outFile, size);
    for (const auto& [key, value] : mapping) {
        writeUnsigned(outFile, key);
        writeSet(outFile, value);
    }
}

// 函数用于从文件流中读取并还原整个unordered_map结构（反序列化映射）
std::unordered_map<unsigned, std::set<unsigned>> readUnorderedMap(std::ifstream& inFile) {
    std::unordered_map<unsigned, std::set<unsigned>> result;
    size_t size = readUnsigned(inFile);
    for (size_t i = 0; i < size; ++i) {
        unsigned key = readUnsigned(inFile);
        std::set<unsigned> value = readSet(inFile);
        result[key] = value;
    }
    return result;
}

// 保存unordered_map结构到指定文件路径的函数
void saveToFile(const std::string& affinity_path, const std::unordered_map<unsigned, std::set<unsigned>>& mapping) {
    std::ofstream outFile(affinity_path, std::ios::binary);
    if (outFile.is_open()) {
        writeUnorderedMap(outFile, mapping);
        outFile.close();
    } else {
        std::cerr << "无法打开文件: " << affinity_path << " 进行写入操作" << std::endl;
    }
}

// 从指定文件路径读取并还原unordered_map结构的函数
std::unordered_map<unsigned, std::set<unsigned>> loadFromFile(const std::string& affinity_path) {
    std::unordered_map<unsigned, std::set<unsigned>> result;
    std::ifstream inFile(affinity_path, std::ios::binary);
    if (inFile.is_open()) {
        result = readUnorderedMap(inFile);
        inFile.close();
    } else {
        std::cerr << "无法打开文件: " << affinity_path << " 进行读取操作" << std::endl;
    }
    return result;
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

class Block{

public:
    std::vector<unsigned> out_block;
    double duplicateRatio;
    size_t out_degree;
    size_t maximum_block_key;
    size_t maximum_block_value;
    size_t maximum_block_value_10_sum;
    std::vector<unsigned> top_10_block_id;
    Block(){};
    ~Block(){
        this->out_block.clear();
        this->top_10_block_id.clear();
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
        this->top_10_block_id.clear();
        for(size_t i = 0; i<10;i++){
            this->maximum_block_value_10_sum += sorted_values[i];
            this->top_10_block_id.push_back(sorted_keys[i]);
        }

    }
};


template<typename T>
// Read index and
void stat_block_outdegree(
  diskann::Metric& metric, const std::string& index_path_prefix,
  const std::string& result_output_prefix,
  const std::string& disk_file_path,
  const unsigned num_threads,
  const bool use_page_search=true){
  _u64                               C; // 单个块内部有多少个节点
  _u64                               _partition_nums; // 分区数量 也等于 磁盘上的块数
  _u64                               _nd; // 索引中的点数量
  _u64                               max_node_len; // 最大单节点长度
  std::vector<std::vector<unsigned>> layout;
  std::vector<std::vector<unsigned>> _partition;
  std::unordered_map<unsigned,std::set<unsigned>> mapping;

  std::vector<unsigned> id2page_; // 在重排后的索引中，使用点id找到对应的物理块号
  std::vector<std::vector<unsigned>> gp_layout_;
  
  std::string indexname = disk_file_path;
  std::string partition_name = index_path_prefix+"_partition.bin";

  std::string pq_table_bin = index_path_prefix + "_pq_pivots.bin";
  std::string pq_compressed_vectors =
        index_path_prefix + "_pq_compressed.bin";
  std::string medoids_file = index_path_prefix + "_medoids.bin";

  std::cout<<"Index file name is: "<<indexname<<std::endl;
  std::cout<<"Partition file name is: "<<partition_name<<std::endl;
  std::ifstream part(partition_name);
  part.read((char*) &C, sizeof(_u64));
  part.read((char*) &_partition_nums, sizeof(_u64));
  part.read((char*) &_nd, sizeof(_u64));
  std::cout << "Partition meta: C: " << C << " partition_nums:" << _partition_nums
            << " _nd:" << _nd << std::endl;

  if(use_page_search){
    std::cout<<"Executing on page_search index"<<std::endl;
  }else{
    std::cout<<"Executing on beam_search index"<<std::endl;
  }



  auto meta_pair = diskann::get_disk_index_meta(indexname.c_str());
  _u64 actual_index_size = get_file_size(indexname.c_str());
  _u64 expected_file_size, expected_npts;

  if (meta_pair.first) {
      // new version
      expected_file_size = meta_pair.second.back();
      expected_npts = meta_pair.second.front();
  } else {
      expected_file_size = meta_pair.second.front();
      expected_npts = meta_pair.second[1];
  }
  if (expected_file_size != actual_index_size) {
    diskann::cout << "File size mismatch for " << indexname
                  << " (size: " << actual_index_size << ")"
                  << " with meta-data size: " << expected_file_size << std::endl;
    exit(-1);
  }
  if (expected_npts != _nd) {
    diskann::cout << "expect _nd: " << _nd
                  << " actual _nd: " << expected_npts << std::endl;
    exit(-1);
  }

  max_node_len = meta_pair.second[3];
  unsigned nnodes_per_sector = meta_pair.second[4];

  if (SECTOR_LEN / max_node_len != C) {
    diskann::cout << "nnodes per sector: " << SECTOR_LEN / max_node_len << " C: " << C
                  << std::endl;
    exit(-1);
  }


  size_t pq_file_dim, pq_file_num_centroids;
  diskann::get_bin_metadata(pq_table_bin, pq_file_num_centroids, pq_file_dim,
                     METADATA_SIZE);
  if (pq_file_num_centroids != 256) {
    diskann::cout << "Error. Number of PQ centroids is not 256. Exitting."
                  << std::endl;
    // return -1;
  }
  unsigned disk_bytes_per_point = pq_file_dim * sizeof(T);
  std::cout << "disk bytes per point: "<<disk_bytes_per_point << std::endl;



  layout.resize(_partition_nums);
  for (unsigned i = 0; i < _partition_nums; i++) {
    unsigned s;
    part.read((char*) &s, sizeof(unsigned));
    layout[i].resize(s);
    part.read((char*) layout[i].data(), sizeof(unsigned) * s);
  }
  id2page_.resize(_nd);
  part.read((char *) id2page_.data(), sizeof(unsigned) * _nd);

  // this time, we load all index into mem;
  std::cout << "nnodes per sector "<<nnodes_per_sector << std::endl;
  _u64 file_size = READ_SECTOR_LEN + READ_SECTOR_LEN * ((expected_npts + nnodes_per_sector - 1) / nnodes_per_sector);
  std::unique_ptr<char[]> mem_index =
      std::make_unique<char[]>(file_size);
  std::ifstream diskann_reader(indexname);
  diskann_reader.read(mem_index.get(),file_size);

  _u64 block_nums = ((expected_npts + nnodes_per_sector - 1) / nnodes_per_sector);
  std::cout << "Amount of blocks " << block_nums << std::endl;

  // 定义writer
  _u64            read_blk_size = 64 * 1024 * 1024;
  _u64            write_blk_size = read_blk_size;
  std::string output_outdegree_path = result_output_prefix+"_outdegree";
  if(use_page_search){
    output_outdegree_path = output_outdegree_path+"_PageSearch"+std::to_string(1)+".bin";
  }
  else{
    output_outdegree_path = output_outdegree_path+"_PageSearch"+std::to_string(0)+".bin";
  }
  std::cout<<"Output file: "<<output_outdegree_path<<std::endl;
  cached_ofstream diskann_writer(output_outdegree_path, write_blk_size);


  std::unique_ptr<char[]> sector_buf = std::make_unique<char[]>(SECTOR_LEN);
  // std::unique_ptr<char[]> node_buf = std::make_unique<char[]>(max_node_len);


  diskann_writer.write((char*) &block_nums,sizeof(_u64));

  std::vector<Block>  block_affinity_graph;
  block_affinity_graph.resize(block_nums);
  for (unsigned i = 0; i < block_nums; i++) {
    if (i % 3000000 == 0) {
      diskann::cout << "calc affinity has done " << (float) i / block_nums
                    << std::endl;
      diskann::cout.flush();
    }
    // auto set = mapping.find(i);
    // if(set == mapping.end()){
    //   mapping[i] = std::set<unsigned>();
    // }
    

    std::vector<unsigned> vec_out_blocks;

    // memset(sector_buf.get(), 0, SECTOR_LEN);
    uint64_t sector_index = (1 + i) * READ_SECTOR_LEN;
    // memcpy((char*) sector_buf.get(),
    //          (char*) mem_index.get() + sector_index, SECTOR_LEN);

    char* disk_sector_buf = mem_index.get() + sector_index;
    
    // 对每一个块内部的所有节点以及他们的邻居进行合并展开
    for(unsigned j = 0;j<nnodes_per_sector;j++){
      char *node_disk_buf =
            OFFSET_TO_NODE(disk_sector_buf,j);
      unsigned *node_buf = OFFSET_TO_NODE_NHOOD(node_disk_buf);
      _u64      nnbrs = (_u64)(*node_buf);
      unsigned *node_nbrs = (node_buf + 1);
      // if(i ==0 || i==1){
      //   // 打印一个向量的样例
      //   T* node_coord = OFFSET_TO_NODE_COORDS(node_disk_buf);
      //   for(unsigned k = 0;k < disk_bytes_per_point;k++){
      //     T sub_vec = (T)(*(node_coord+k*sizeof(T)));
      //     printf("%d:{%d} ",k,sub_vec);
      //     // std::cout<<k<<":{"<<sub_vec<<"} ";
      //   }
      //   std::cout<<std::endl;
      //   std::cout<<"nnbrs:"<<nnbrs<<std::endl;
      // }
      for(_u64 nbr_idx = 0; nbr_idx < nnbrs;nbr_idx++){
        unsigned *tmp_nbr = (node_nbrs + nbr_idx);
        // TODO 这里需要考虑如果是beam_search直接计算块号就可以，如果是page_search 那么使用id_page
        unsigned nbr_block_id = 0;
        if(use_page_search){
          nbr_block_id = id2page_[(*tmp_nbr)];
        }else{
          nbr_block_id = NODE_SECTOR_NO((*tmp_nbr));
        }
        vec_out_blocks.push_back(nbr_block_id);
        block_affinity_graph[i].out_block.push_back(nbr_block_id);
        // mapping[i].insert(nbr_block_id);
      }
      
    }

    unsigned vec_size = vec_out_blocks.size();
    diskann_writer.write((char*) &vec_size,sizeof(unsigned));
    diskann_writer.write((char*) vec_out_blocks.data(),sizeof(unsigned)* vec_size);

    // block_affinity_graph[i].out_block.resize(vec_size);
    // for(size_t i  = 0;i<vec_size;i++){
    //   block_affinity_graph[i].out_block.push_back(vec_out_blocks[i]);
    // }

    mapping[i].clear();
    // vec_out_blocks.clear();
  }
  mem_index.reset();  // 显式释放mem_index所占内存


  // std::cout<<"cnt1: "<<cnt1<<",cnt2: "<<cnt2<<",cnt3: "<<cnt3<<std::endl;
  
  
  // saveToFile(output_outdegree_path,mapping);

  // save to file
  {
    // _u64            read_blk_size = 64 * 1024 * 1024;
    // _u64            write_blk_size = read_blk_size;
    // std::string affinity_path(indexname);
    // affinity_path = affinity_path.substr(0, affinity_path.find_last_of('.')) + affinity_filename;
    
    // std::cout << "Affinity path: "<< affinity_path << std::endl;
    // cached_ofstream diskann_writer(partition_path, write_blk_size);
    // saveToFile(affinity_path, mapping);
  }
  std::cout<<"Begin to analyze."<<std::endl;
  std::vector<double> ratio(block_nums);
  std::vector<size_t> out_degree(block_nums);
  std::vector<size_t> max_out_num(block_nums);
  std::vector<size_t> max_out_num_top10_sum(block_nums);
  std::vector<std::vector<size_t>> top_10_block_id_for_each(block_nums);


  // 释放内存
  #pragma omp parallel for
  for(size_t i = 0 ; i < block_nums;i++){
      // ratio[i] = calculateDuplicateRatio(block_affinity_graph[i]);
      block_affinity_graph[i].anaylze();
      ratio[i] = block_affinity_graph[i].duplicateRatio;
      out_degree[i] = block_affinity_graph[i].out_degree;
      max_out_num[i] = block_affinity_graph[i].maximum_block_value;
      max_out_num_top10_sum[i] = block_affinity_graph[i].maximum_block_value_10_sum;
      for(int j = 0;j<10;j++){
        top_10_block_id_for_each[i].push_back(block_affinity_graph[i].top_10_block_id[j]);
      }
      // if(i<10){
      //     diskann::cout<<ratio<<std::endl;
      // }
      // block_affinity_graph[i].clear();
  }
  block_affinity_graph.clear();


  size_t p10 = 0,p50 = 0,p80 = 0,p90 = 0;
  for(size_t i = 0 ; i < block_nums;i++){
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

  diskann::cout<<" p10 = "<< static_cast<double>(p10)/block_nums<<std::endl;
  diskann::cout<<" p50 = "<< static_cast<double>(p50)/block_nums<<std::endl;
  diskann::cout<<" p80 = "<< static_cast<double>(p80)/block_nums<<std::endl;
  diskann::cout<<" p90 = "<< static_cast<double>(p90)/block_nums<<std::endl;

  double avg = calculateAverageOutDegree(out_degree);
  std::cout << "Average out-degree: " << avg << std::endl;
  avg = calculateAverageOutDegree(max_out_num);
  std::cout << "Average out-block_num: " << avg << std::endl;
  avg = calculateAverageOutDegree(max_out_num_top10_sum);
  std::cout << "Average out-block_num top 10 sum : " << avg << std::endl;


  std::string output_blockprefetch_path = index_path_prefix+"_block_prefetch";
  if(use_page_search){
    output_blockprefetch_path = output_blockprefetch_path+"_PS"+std::to_string(1)+".bin";
  }
  else{
    output_blockprefetch_path = output_blockprefetch_path+"_PS"+std::to_string(0)+".bin";
  }
  std::cout<<"Output block prefetch file: "<<output_blockprefetch_path<<std::endl;
  cached_ofstream blockprefetch_writer(output_blockprefetch_path, write_blk_size);
  blockprefetch_writer.write((char*) &block_nums,sizeof(_u64));
  for (unsigned i = 0; i < block_nums; i++) {
    blockprefetch_writer.write((char*)top_10_block_id_for_each[i].data(), sizeof(unsigned)* 10);
  }


  return;
}

// Write DiskANN sector data according to graph-partition layout 
// The new index data

int main(int argc, char** argv){
  
  // char* indexName = argv[1];
  // char* partitonName = argv[2];


  std::string data_type, dist_fn, index_path_prefix, result_path_prefix, disk_file_path,
              block_out_degree_save_path;
  unsigned              num_threads;
  bool                  use_page_search = true;

  po::options_description desc{"Arguments"};
  try {
    desc.add_options()("help,h", "Print information on arguments");
    desc.add_options()("data_type",
                       po::value<std::string>(&data_type)->required(),
                       "data type <int8/uint8/float>");
    desc.add_options()("dist_fn", po::value<std::string>(&dist_fn)->required(),
                       "distance function <l2/mips/fast_l2>");
    desc.add_options()("index_path_prefix",
                       po::value<std::string>(&index_path_prefix)->required(),
                       "Path prefix to the index");
    desc.add_options()("result_path",
                       po::value<std::string>(&result_path_prefix)->required(),
                       "Path prefix for saving results of the queries");

    
    desc.add_options()(
        "num_threads,T",
        po::value<uint32_t>(&num_threads)->default_value(omp_get_num_procs()),
        "Number of threads used for building index (defaults to "
        "omp_get_num_procs())");
    desc.add_options()("use_page_search", po::value<bool>(&use_page_search)->default_value(1),
                       "Use 1 for page search (default), 0 for DiskANN beam search");
    desc.add_options()("disk_file_path", po::value<std::string>(&disk_file_path)->required(),
                       "The path of the disk file (_disk.index in the original DiskANN)");
    desc.add_options()("block_out_degree_save_path", po::value<std::string>(&block_out_degree_save_path)->required(),
                       "frequency file save path");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help")) {
      std::cout << desc;
      return 0;
    }
    po::notify(vm);
  } catch (const std::exception& ex) {
    std::cerr << ex.what() << '\n';
    return -1;
  }

  diskann::Metric metric;
  if (dist_fn == std::string("mips")) {
    metric = diskann::Metric::INNER_PRODUCT;
  } else if (dist_fn == std::string("l2")) {
    metric = diskann::Metric::L2;
  } else if (dist_fn == std::string("cosine")) {
    metric = diskann::Metric::COSINE;
  } else {
    std::cout << "Unsupported distance function. Currently only L2/ Inner "
                 "Product/Cosine are supported."
              << std::endl;
    return -1;
  }
  // ${INDEX_PREFIX_PATH}_partition.bin
  stat_block_outdegree<uint8_t>(metric,index_path_prefix, 
            result_path_prefix, 
            disk_file_path,
            num_threads,
            use_page_search);


  // relayout(indexName, partitonName);
  // stat_affinity(indexName, partitonName);
  return 0;
}