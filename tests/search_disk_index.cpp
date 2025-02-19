// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

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
#include "index.h"
#include "math_utils.h"
#include "memory_mapper.h"
#include "partition_and_pq.h"
#include "timer.h"
#include "utils.h"
#include "percentile_stats.h"

#ifndef _WINDOWS
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include "linux_aligned_file_reader.h"
#else
#ifdef USE_BING_INFRA
#include "bing_aligned_file_reader.h"
#else
#include "windows_aligned_file_reader.h"
#endif
#endif

#define WARMUP false


#define READ_SECTOR_LEN (size_t) 4096



namespace po = boost::program_options;
using namespace diskann;

void print_stats(std::string category, std::vector<float> percentiles,
                 std::vector<float> results) {
  diskann::cout << std::setw(20) << category << ": " << std::flush;
  for (uint32_t s = 0; s < percentiles.size(); s++) {
    diskann::cout << std::setw(8) << percentiles[s] << "%";
  }
  diskann::cout << std::endl;
  diskann::cout << std::setw(22) << " " << std::flush;
  for (uint32_t s = 0; s < percentiles.size(); s++) {
    diskann::cout << std::setw(9) << results[s];
  }
  diskann::cout << std::endl;
}

void writeIndexToSPDK(std::string indexname, ssdps::SpdkWrapper* reader){
  auto meta_pair = diskann::get_disk_index_meta(indexname);
  _u64 actual_index_size = get_file_size(indexname);
  _u64 expected_file_size, expected_npts;
  _u64                               max_node_len;

  std::cout<<"Copy index file: "<<indexname<<" to spdk."<<std::endl;

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
  max_node_len = meta_pair.second[3];
  unsigned nnodes_per_sector = meta_pair.second[4];
  
  _u64 file_size = READ_SECTOR_LEN + READ_SECTOR_LEN * ((expected_npts + nnodes_per_sector - 1) / nnodes_per_sector);
  std::cout<<"file_size:  "<<file_size<<std::endl;
  std::unique_ptr<char[]> mem_index =
      std::make_unique<char[]>(file_size);
  std::ifstream diskann_reader(indexname);
  diskann_reader.read(mem_index.get(),file_size);

  unsigned batch_size = 1024;
  unsigned sector_size = ((expected_npts + nnodes_per_sector - 1) / nnodes_per_sector) + 1;
  std::cout<<"sector_size:  "<<sector_size<<std::endl;
  unsigned wrt_idx = 0;

  char *buf_2 = (char *)spdk_zmalloc(READ_SECTOR_LEN* batch_size, READ_SECTOR_LEN, NULL,
    SPDK_ENV_SOCKET_ID_ANY, SPDK_MALLOC_DMA);
  
  while(wrt_idx<sector_size){
    unsigned wrt_size_iter = std::min(sector_size - wrt_idx , batch_size);
    memcpy(buf_2,mem_index.get()+wrt_idx*READ_SECTOR_LEN,wrt_size_iter*READ_SECTOR_LEN);
    int res = memcmp(buf_2,mem_index.get()+wrt_idx*READ_SECTOR_LEN,wrt_size_iter*READ_SECTOR_LEN);
    if(res!=0){
      std::cout<<res<<" "<<wrt_idx<<std::endl;
    }
    reader->SyncWrite(buf_2,wrt_size_iter*READ_SECTOR_LEN,wrt_idx,0);
    wrt_idx += wrt_size_iter;
  }
  std::cout<<"Correction check."<<std::endl;
  wrt_idx = 0;
  while(wrt_idx<sector_size){
    unsigned wrt_size_iter = std::min(sector_size - wrt_idx , batch_size);
    reader->SyncRead(buf_2,wrt_size_iter*READ_SECTOR_LEN,wrt_idx,0);
    int res = memcmp(buf_2,mem_index.get()+wrt_idx*READ_SECTOR_LEN,wrt_size_iter*READ_SECTOR_LEN);
    if(res!=0){
      std::cout<<res<<" "<<wrt_idx<<std::endl;
    }
    wrt_idx += wrt_size_iter;
  }

  spdk_free(buf_2);
  std::cout<<"Done."<<std::endl;
}

template<typename T>
int search_disk_index(
    diskann::Metric& metric, const std::string& index_path_prefix,
    const std::string& mem_index_path,
    const std::string& result_output_prefix, const std::string& query_file,
    const std::string& gt_file, 
    const std::string& disk_file_path,
     const std::vector<unsigned>& Lvec,
     SearchParams& params) {


  unsigned num_threads = params.num_threads;
  unsigned recall_at = params.recall_at;
  unsigned beamwidth = params.beam_width;
  unsigned num_nodes_to_cache = params.num_nodes_to_cache;
  _u32 search_io_limit=params.io_limit;
  _u32 mem_L = params.mem_L;

  // page search
  bool use_page_search = params.use_page_search;
  bool use_pipeline=params.use_pipeline;
  float use_ratio=params.use_ratio;
  bool use_reorder_data = params.use_reorder_data;
  bool use_sq = params.use_sq;

  bool use_coro = params.use_coro;
  bool pure_io = params.pure_io;



  diskann::cout << "Search parameters: #threads: " << num_threads << ", ";
  if (beamwidth <= 0)
    diskann::cout << "beamwidth to be optimized for each L value" << std::flush;
  else
    diskann::cout << " beamwidth: " << beamwidth << std::flush;
  if (search_io_limit == std::numeric_limits<_u32>::max())
    diskann::cout << "." << std::endl;
  else
    diskann::cout << ", io_limit: " << search_io_limit << "." << std::endl;

  std::string warmup_query_file = index_path_prefix + "_sample_data.bin";

  // load query bin
  T*        query = nullptr;
  unsigned* gt_ids = nullptr;
  float*    gt_dists = nullptr;
  size_t    query_num, query_dim, query_aligned_dim, gt_num, gt_dim;
  diskann::load_aligned_bin<T>(query_file, query, query_num, query_dim,
                               query_aligned_dim);

  bool calc_recall_flag = false;
  if (gt_file != std::string("null") && gt_file != std::string("NULL") &&
      file_exists(gt_file)) {
    diskann::load_truthset(gt_file, gt_ids, gt_dists, gt_num, gt_dim);
    if (gt_num != query_num) {
      diskann::cout
          << "Error. Mismatch in number of queries and ground truth data"
          << std::endl;
    }
    calc_recall_flag = true;
  }

  std::shared_ptr<AlignedFileReader> reader = nullptr;
// #ifdef _WINDOWS
// #ifndef USE_BING_INFRA
//   reader.reset(new WindowsAlignedFileReader());
// #else
//   reader.reset(new diskann::BingAlignedFileReader());
// #endif
// #else
  reader.reset(new LinuxAlignedFileReader()); // Only Linux
// #endif

  std::shared_ptr<ssdps::SpdkWrapper> spdk_reader = ssdps::SpdkWrapper::create(1);
  spdk_reader->Init();
  // writeIndexToSPDK(disk_file_path,spdk_reader.get());

  if(use_sq && !std::is_same<T, float>::value){
    std::cout << "erro, only support float sq" << std::endl;
    exit(-1);
  }
  std::unique_ptr<diskann::PQFlashIndex<T>> _pFlashIndex(
      new diskann::PQFlashIndex<T>(reader,spdk_reader, use_page_search, metric, use_sq));
  // _pFlashIndex->set_ncoroutines(MAX_COROUTINE);
  _pFlashIndex->set_ncoroutines(params.coro_size);
  _pFlashIndex->load_search_params(params);

  int res = _pFlashIndex->load(num_threads, index_path_prefix.c_str(), disk_file_path);

  if (res != 0) {
    return res;
  }

  _pFlashIndex->set_query_aligned_dim(query_aligned_dim);
  

  size_t load_mem = getCurrentRSS();

  // load in-memory navigation graph
  if (mem_L) {
    _pFlashIndex->load_mem_index(metric, query_dim, mem_index_path, num_threads, mem_L);
  }

  

  // cache bfs levels
  std::vector<uint32_t> node_list;
  diskann::cout << "Caching " << num_nodes_to_cache
                << " BFS nodes around medoid(s)" << std::endl;
  //_pFlashIndex->cache_bfs_levels(num_nodes_to_cache, node_list);
  if (num_nodes_to_cache > 0){
    if(use_sq){
      std::cout << "not support sq cache, please use mem index" << std::endl;
      exit(-1);
    }
    _pFlashIndex->generate_cache_list_from_sample_queries(
        warmup_query_file, 15, 6, num_nodes_to_cache, num_threads, node_list, use_page_search, mem_L);
    _pFlashIndex->load_cache_list(node_list);
  }
  
  node_list.clear();
  node_list.shrink_to_fit();

  size_t cache_mem = getCurrentRSS();

  omp_set_num_threads(num_threads);

  uint64_t warmup_L = 20;
  uint64_t warmup_num = 0, warmup_dim = 0, warmup_aligned_dim = 0;
  T*       warmup = nullptr;

  if (WARMUP) {
    if (file_exists(warmup_query_file)) {
      diskann::load_aligned_bin<T>(warmup_query_file, warmup, warmup_num,
                                   warmup_dim, warmup_aligned_dim);
    } else {
      warmup_num = (std::min)((_u32) 150000, (_u32) 15000 * num_threads);
      warmup_dim = query_dim;
      warmup_aligned_dim = query_aligned_dim;
      diskann::alloc_aligned(((void**) &warmup),
                             warmup_num * warmup_aligned_dim * sizeof(T),
                             8 * sizeof(T));
      std::memset(warmup, 0, warmup_num * warmup_aligned_dim * sizeof(T));
      std::random_device              rd;
      std::mt19937                    gen(rd());
      std::uniform_int_distribution<> dis(-128, 127);
      for (uint32_t i = 0; i < warmup_num; i++) {
        for (uint32_t d = 0; d < warmup_dim; d++) {
          warmup[i * warmup_aligned_dim + d] = (T) dis(gen);
        }
      }
    }
    diskann::cout << "Warming up index... " << std::flush;
    std::vector<uint64_t> warmup_result_ids_64(warmup_num, 0);
    std::vector<float>    warmup_result_dists(warmup_num, 0);

#pragma omp parallel for schedule(dynamic, 1)
    for (_s64 i = 0; i < (int64_t) warmup_num; i++) {
      _pFlashIndex->cached_beam_search(warmup + (i * warmup_aligned_dim), 1,
                                       warmup_L,
                                       warmup_result_ids_64.data() + (i * 1),
                                       warmup_result_dists.data() + (i * 1), 4);
    }
    diskann::cout << "..done" << std::endl;
  }

  diskann::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
  diskann::cout.precision(2);

  std::string recall_string = "Recall@" + std::to_string(recall_at);
  diskann::cout << std::setw(6) << "L" 
                << std::setw(12) << "Beamwidth"
                << std::setw(16) << "QPS"
                << std::setw(16) << "Mean Latency"
                << std::setw(16) << "P99.9 Latency"
                << std::setw(16) << "P90 Latency"
                << std::setw(16) << "IOps"
                << std::setw(16) << "Mean IOs" 
                << std::setw(16) << "Mean IO (us)"
                << std::setw(16) << "CPU (us)"
                << std::setw(16) << "Mean hops"
                << std::setw(16) << "Mean cache_hits"
                << std::setw(16) << "Aff. cache n"
                << std::setw(20) << "B4 Load In-Mem"
                << std::setw(20) << "After Load Cache"
                << std::setw(15) << "Peak Mem(MB)";
  if (calc_recall_flag) {
    diskann::cout << std::setw(16) << recall_string << std::endl;
  } else
    diskann::cout << std::endl;
  diskann::cout
      << "==============================================================="
         "==============================================================="
         "==============================================================="
         "============================================================"
      << std::endl;

  std::vector<std::vector<uint32_t>> query_result_ids(Lvec.size());
  std::vector<std::vector<float>>    query_result_dists(Lvec.size());

  uint32_t optimized_beamwidth = 2;

  for (uint32_t test_id = 0; test_id < Lvec.size(); test_id++) {
    _u64 L = Lvec[test_id];

    if (L < recall_at) {
      diskann::cout << "Ignoring search with L:" << L
                    << " since it's smaller than K:" << recall_at << std::endl;
      continue;
    }

    if (beamwidth <= 0) {
      diskann::cout << "Tuning beamwidth.." << std::endl;
      optimized_beamwidth =
          optimize_beamwidth(_pFlashIndex, warmup, warmup_num,
                             warmup_aligned_dim, L, optimized_beamwidth);
    } else
      optimized_beamwidth = beamwidth;

    query_result_ids[test_id].resize(recall_at * query_num);
    query_result_dists[test_id].resize(recall_at * query_num);

    auto stats = new diskann::QueryStats[query_num];

    std::vector<uint64_t> query_result_ids_64(recall_at * query_num);
    auto                  s = std::chrono::high_resolution_clock::now();
    // query_num = SEARCH_QUERY;
    if(!params.query_num){
      query_num = params.query_num;
    }
    // _pFlashIndex->verbose_ = true;
    // Using branching outside the for loop instead of inside and 
    // std::function/std::mem_fn for less switching and function calling overhead
    std::cout<<"Query num is "<<query_num<<std::endl;
    if (pure_io) {
      if (use_coro) {
        _pFlashIndex->pure_io_search(query, query_num, optimized_beamwidth,
                                     search_io_limit, stats);

      } else {
        _pFlashIndex->pure_libaio_search(query, query_num, optimized_beamwidth,
                                         search_io_limit, stats);
      }
    } else {
      if (use_page_search) {
        if (use_coro) {
          _pFlashIndex->bqann_search(
              query, query_num, recall_at, mem_L, L, query_result_ids_64.data(),
              query_result_dists[test_id].data(), optimized_beamwidth,
              search_io_limit, use_reorder_data, use_ratio, stats);
        } else {
          bool pipeline = use_pipeline;
          _pFlashIndex->starling_search(
              query, query_num, recall_at, mem_L, L, query_result_ids_64.data(),
              query_result_dists[test_id].data(), optimized_beamwidth,
              search_io_limit, use_reorder_data, use_ratio, pipeline, stats);
        }
      } else {
#pragma omp parallel for schedule(dynamic, 1)
        for (_s64 i = 0; i < (int64_t) query_num; i++) {
          _pFlashIndex->cached_beam_search(
              query + (i * query_aligned_dim), recall_at, L,
              query_result_ids_64.data() + (i * recall_at),
              query_result_dists[test_id].data() + (i * recall_at),
              optimized_beamwidth, search_io_limit, use_reorder_data, stats + i,
              mem_L);
        }
      }
    }
    auto                          e = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = e - s;
    float qps = (1.0 * query_num) / (1.0 * diff.count());

    diskann::convert_types<uint64_t, uint32_t>(query_result_ids_64.data(),
                                               query_result_ids[test_id].data(),
                                               query_num, recall_at);

    auto mean_latency = diskann::get_mean_stats<float>(
        stats, query_num,
        [](const diskann::QueryStats& stats) { return stats.total_us; });

    auto latency_999 = diskann::get_percentile_stats<float>(
        stats, query_num, 0.999,
        [](const diskann::QueryStats& stats) { return stats.total_us; });
    auto latency_90 = diskann::get_percentile_stats<float>(
        stats, query_num, 0.90,
        [](const diskann::QueryStats& stats) { return stats.total_us; });

    auto mean_ios = diskann::get_mean_stats<unsigned>(
        stats, query_num,
        [](const diskann::QueryStats& stats) { return stats.n_ios; });

    auto sum_ios = diskann::get_sum_stats<unsigned>(
        stats, query_num,
        [](const diskann::QueryStats& stats) { return stats.n_ios; });
    
    auto mean_ious = diskann::get_mean_stats<float>(
        stats, query_num,
        [](const diskann::QueryStats &stats) { return stats.io_us; });

    auto mean_cpus = diskann::get_mean_stats<float>(
        stats, query_num,
        [](const diskann::QueryStats& stats) { return stats.cpu_us; });
    auto mean_bubble_time_us = diskann::get_mean_stats<float>(
        stats, query_num,
        [](const diskann::QueryStats& stats) { return stats.bubble_time_us; });

    auto mean_coro_us = diskann::get_mean_stats<float>(
        stats, query_num,
        [](const diskann::QueryStats& stats) { return stats.executing_in_coro_us; });
        

    auto mean_hops = diskann::get_mean_stats<unsigned>(
        stats, query_num,
        [](const diskann::QueryStats& stats) { return stats.n_hops; });
    auto mean_cache_hits = diskann::get_mean_stats<unsigned>(
        stats, query_num,
        [](const diskann::QueryStats& stats) { return stats.n_cache_hits; });
    auto n_aff_cache_nodes = diskann::get_mean_stats<unsigned>(
        stats, query_num,
        [](const diskann::QueryStats& stats) { return stats.n_affinity_cache; });
    
    float iops = (1.0 * sum_ios) / (1.0 * diff.count());

    float recall = 0;
    if (calc_recall_flag) {
      recall = diskann::calculate_recall(query_num, gt_ids, gt_dists, gt_dim,
                                         query_result_ids[test_id].data(),
                                         recall_at, recall_at);
    }

    diskann::cout << std::setw(6) << L
                  << std::setw(12) << optimized_beamwidth
                  << std::setw(16) << qps
                  << std::setw(16) << mean_latency
                  << std::setw(16) << latency_999
                  << std::setw(16) << latency_90
                  << std::setw(16) << iops
                  << std::setw(16) << mean_ios
                  << std::setw(16) << mean_ious
                  << std::setw(16) << mean_cpus
                  << std::setw(16) << mean_hops
                  << std::setw(16) << mean_cache_hits
                  << std::setw(16) << n_aff_cache_nodes
                  << std::setw(20) << load_mem
                  << std::setw(20) << cache_mem
                  << std::setw(15) << getProcessPeakRSS();
    if (calc_recall_flag) {
      diskann::cout << std::setw(16) << recall << std::endl;
    } else
      diskann::cout << std::endl;
    diskann::cout << "L" 
                << ","<< "#Threads"
                << ","<< "Beamwidth"
                << ","<< "QPS"
                << ","<< "Mean Latency"
                << ","<< "P90 Latency"
                << ","<< "P99.9 Latency"
                << ","<< "IOps"
                << ","<< "Mean IOs" 
                << ","<< "Mean IO (us)"
                << ","<< "CPU (us)"
                << ","<< "Mean coro exe time (us)"
                << ","<< "Mean hops"
                << ","<< "Mean cache_hits"
                << ","<< "Aff. cache n"
                << ","<< "B4 Load In-Mem"
                << ","<< "After Load Cache"
                << ","<< "Peak Mem(MB)";
  if (calc_recall_flag) {
    diskann::cout << "," << recall_string << std::endl;
  } else
    diskann::cout << std::endl;

  diskann::cout << L
                  << "," << num_threads
                  << "," << optimized_beamwidth
                  << ","<< qps
                  << ","<< mean_latency
                  << ","<< latency_90
                  << ","<< latency_999
                  << ","<< iops
                  << ","<< mean_ios
                  << ","<< mean_ious
                  << ","<< mean_cpus
                  << ","<< mean_coro_us
                  << ","<< mean_hops
                  << ","<< mean_cache_hits
                  << ","<< n_aff_cache_nodes
                  << ","<< load_mem
                  << ","<< cache_mem
                  << ","<< getProcessPeakRSS();
    if (calc_recall_flag) {
      diskann::cout << "," << recall << std::endl;
    } else
      diskann::cout << std::endl;

    diskann::cout <<"Bubble time proportion is:"<< mean_bubble_time_us / mean_latency << std::endl;

    {
      // save block path
      std::string  block_path_prefix =
                  result_output_prefix + "_block_path" +"_L" + std::to_string(L)+"_PS"+std::to_string(use_page_search)+ "_B"+ std::to_string(optimized_beamwidth) +"_T"+std::to_string(num_threads);
      std::string block_path_with_timestamp = block_path_prefix+"_withts.bin";
      std::string block_path_no_timestamp = block_path_prefix+"_nots.txt";;
      std::ofstream outFile(block_path_with_timestamp, std::ios::binary);
      std::ofstream outFile_no_ts(block_path_no_timestamp, std::ios::binary);
      if (!outFile) {
          std::cerr << "Failed to open file for writing: " << block_path_with_timestamp << std::endl;
      }else if(!outFile_no_ts){
          std::cerr << "Failed to open file for writing: " << block_path_no_timestamp << std::endl;
      }else{
        // 写入 query 数量
        // size_t query_num = queries.size();
        outFile.write(reinterpret_cast<const char*>(&query_num), sizeof(query_num));
        outFile_no_ts<<std::to_string(query_num)<<std::endl;
        // 遍历每个查
        for (_s64 i = 0; i < (int64_t) query_num; i++) {
          size_t block_num = stats[i].block_visited_queue.size();
          outFile.write(reinterpret_cast<const char*>(&block_num), sizeof(block_num));
          outFile_no_ts<<std::to_string(block_num);
          // 写入每个 BlockVisited
            for (const auto& block : stats[i].block_visited_queue) {
                // 写入 block_id
                outFile.write(reinterpret_cast<const char*>(&block.block_id), sizeof(block.block_id));
                outFile_no_ts<<" "<<std::to_string(block.block_id);

                // 将 timestamp 转换为 float（秒数）并写入
                std::chrono::duration<double> diff = block.timestamp - s;
                float timestamp_float =  diff.count();
                outFile.write(reinterpret_cast<const char*>(&timestamp_float), sizeof(timestamp_float));
            }
          outFile_no_ts<<std::endl;
        }
        outFile.close();
        outFile_no_ts.close();
      }
      
    }

    delete[] stats;
  }

  diskann::cout << "Done searching. Now saving results " << std::endl;
  _u64 test_id = 0;
  for (auto L : Lvec) {
    if (L < recall_at)
      continue;

    std::string cur_result_path =
        result_output_prefix + "_" + std::to_string(L) + "_idx_uint32.bin";
    diskann::save_bin<_u32>(cur_result_path, query_result_ids[test_id].data(),
                            query_num, recall_at);

    cur_result_path =
        result_output_prefix + "_" + std::to_string(L) + "_dists_float.bin";
    diskann::save_bin<float>(cur_result_path,
                             query_result_dists[test_id++].data(), query_num,
                             recall_at);
  }

  diskann::aligned_free(query);
  if (warmup != nullptr)
    diskann::aligned_free(warmup);
  return 0;
}

int main(int argc, char** argv) {
  std::string data_type, dist_fn, index_path_prefix, result_path_prefix,
      query_file, gt_file, disk_file_path, mem_index_path;
  unsigned              num_threads, K, W, num_nodes_to_cache, search_io_limit;
  unsigned              mem_L;
  std::vector<unsigned> Lvec;
  bool                  use_reorder_data = false;
  bool                  use_page_search = true;
  bool                  use_pipeline = true;
  bool                  page_expansion = true;
  bool                  use_coro = false;
  float                 use_ratio = 1.0;
  bool                  pure_io = false;
  unsigned query_num = 0;
  bool use_sq = false;
  unsigned coro_size = 0;

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
    desc.add_options()("query_file",
                       po::value<std::string>(&query_file)->required(),
                       "Query file in binary format");
    desc.add_options()(
        "gt_file",
        po::value<std::string>(&gt_file)->default_value(std::string("null")),
        "ground truth file for the queryset");
    desc.add_options()("recall_at,K", po::value<uint32_t>(&K)->required(),
                       "Number of neighbors to be returned");
    desc.add_options()("query_num", po::value<uint32_t>(&query_num)->default_value(0),
                       "Set if you want execute only first# query. 0 is default.");
    desc.add_options()("search_list,L",
                       po::value<std::vector<unsigned>>(&Lvec)->multitoken(),
                       "List of L values of search");
    desc.add_options()("beamwidth,W", po::value<uint32_t>(&W)->default_value(2),
                       "Beamwidth for search. Set 0 to optimize internally.");
    desc.add_options()(
        "num_nodes_to_cache",
        po::value<uint32_t>(&num_nodes_to_cache)->default_value(0),
        "Beamwidth for search");
    desc.add_options()("search_io_limit",
                       po::value<uint32_t>(&search_io_limit)
                           ->default_value(std::numeric_limits<_u32>::max()),
                       "Max #IOs for search");
    desc.add_options()(
        "num_threads,T",
        po::value<uint32_t>(&num_threads)->default_value(omp_get_num_procs()),
        "Number of threads used for building index (defaults to "
        "omp_get_num_procs())");
    desc.add_options()("use_reorder_data",
                       po::bool_switch()->default_value(false),
                       "Include full precision data in the index. Use only in "
                       "conjuction with compressed data on SSD.");
    desc.add_options()("use_sq",
                       po::value<bool>(&use_sq)->default_value(0),
                       "Use SQ-compressed disk vector.");
    desc.add_options()("mem_L", po::value<unsigned>(&mem_L)->default_value(0),
                       "The L of the in-memory navigation graph while searching. Use 0 to disable");
    desc.add_options()("use_page_search", po::value<bool>(&use_page_search)->default_value(1),
                       "Use 1 for page search (default), 0 for DiskANN beam search");
    desc.add_options()("use_pipeline", po::value<bool>(&use_pipeline)->default_value(1),
                       "Use 1 for pipeline (default), 0 for not pipeline");
    desc.add_options()("page_expansion", po::value<bool>(&page_expansion)->default_value(1),
                       "Use 1 for using page_expansion in search (default), 0 for node_expansion");
    desc.add_options()("use_coro", po::value<bool>(&use_coro)->default_value(0),
                       "Use 1 for using coroutine in IO, 0 for using coroutine (default).");
    desc.add_options()("coro_size", po::value<unsigned>(&coro_size)->default_value(1),
                       "coro size per thread");
    desc.add_options()("pure_io", po::value<bool>(&pure_io)->default_value(0),
                       "Use 1 for testing pure IO, 0 for not (default).");
    desc.add_options()("use_ratio", po::value<float>(&use_ratio)->default_value(1.0f),
                       "The percentage of how many vectors in a page to search each time");
    desc.add_options()("disk_file_path", po::value<std::string>(&disk_file_path)->required(),
                       "The path of the disk file (_disk.index in the original DiskANN)");
    desc.add_options()("mem_index_path", po::value<std::string>(&mem_index_path)->default_value(""),
                       "The prefix path of the mem_index");

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

  if (use_ratio < 0 || use_ratio > 1.0f) {
    std::cout << "use_ratio should be in the range [0, 1] (inclusive)." << std::endl;
    return -1;
  }

  if ((data_type != std::string("float")) &&
      (metric == diskann::Metric::INNER_PRODUCT)) {
    std::cout << "Currently support only floating point data for Inner Product."
              << std::endl;
    return -1;
  }
  if ((data_type != std::string("float")) &&
      (use_sq)) {
    std::cout << "Currently support only float sq"
              << std::endl;
    return -1;
  }

  if (use_reorder_data && data_type != std::string("float")) {
    std::cout << "Error: Reorder data for reordering currently only "
                 "supported for float data type."
              << std::endl;
    return -1;
  }
  if(!use_page_search && use_sq){
    std::cout << "Currently not support diskann + sq" << std::endl;
    return -1;
  }

  diskann::SearchParams params;
  params.num_threads = num_threads;
  params.recall_at = K;
  params.beam_width = W;
  params.num_nodes_to_cache = num_nodes_to_cache;
  params.io_limit = search_io_limit;
  params.mem_L = mem_L;

  params.use_page_search = use_page_search;
  params.use_pipeline = use_pipeline;
  params.use_ratio = use_ratio;
  params.use_reorder_data = use_reorder_data;
  params.use_sq = use_sq;

  params.use_coro = use_coro;
  params.coro_size = coro_size;
  params.pure_io = pure_io;

  params.query_num = query_num;

  try {
    if (data_type == std::string("float"))
      return search_disk_index<float>(
          metric, index_path_prefix, mem_index_path, result_path_prefix,
          query_file, gt_file, disk_file_path,Lvec, params);
    else if (data_type == std::string("int8"))
      return search_disk_index<int8_t>(
          metric, index_path_prefix, mem_index_path, result_path_prefix,
          query_file, gt_file, disk_file_path, Lvec, params);
    else if (data_type == std::string("uint8"))
      return search_disk_index<uint8_t>(
          metric, index_path_prefix, mem_index_path, result_path_prefix,
          query_file, gt_file, disk_file_path, Lvec, params);
    else {
      std::cerr << "Unsupported data type. Use float or int8 or uint8"
                << std::endl;
      return -1;
    }
  } catch (const std::exception& e) {
    std::cout << std::string(e.what()) << std::endl;
    diskann::cerr << "Index search failed." << std::endl;
    return -1;
  }
}
