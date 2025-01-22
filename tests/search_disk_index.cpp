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

namespace po = boost::program_options;

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

template<typename T>
int search_disk_index(
    diskann::Metric& metric, const std::string& index_path_prefix,
    const std::string& mem_index_path,
    const std::string& result_output_prefix, const std::string& query_file,
    const std::string& gt_file, 
    const std::string& disk_file_path,
    const unsigned num_threads, const unsigned recall_at,
    const unsigned beamwidth, const unsigned num_nodes_to_cache,
    const _u32 search_io_limit, const std::vector<unsigned>& Lvec,
    const _u32 mem_L,
    const bool use_page_search=true,
    const bool use_coro = false,
    const float use_ratio=1.0,
    const bool use_reorder_data = false,
    const bool use_sq = false) {
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
#ifdef _WINDOWS
#ifndef USE_BING_INFRA
  reader.reset(new WindowsAlignedFileReader());
#else
  reader.reset(new diskann::BingAlignedFileReader());
#endif
#else
  reader.reset(new LinuxAlignedFileReader());
#endif

  if(use_sq && !std::is_same<T, float>::value){
    std::cout << "erro, only support float sq" << std::endl;
    exit(-1);
  }
  std::unique_ptr<diskann::PQFlashIndex<T>> _pFlashIndex(
      new diskann::PQFlashIndex<T>(reader, use_page_search, metric, use_sq));

  int res = _pFlashIndex->load(num_threads, index_path_prefix.c_str(), disk_file_path);

  if (res != 0) {
    return res;
  }

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
    // query_num = 1;
    // Using branching outside the for loop instead of inside and 
    // std::function/std::mem_fn for less switching and function calling overhead
    if (use_page_search) {
      if(use_sq){
  #pragma omp parallel for schedule(dynamic, 1)
        for (_s64 i = 0; i < (int64_t) query_num; i++) {
          _pFlashIndex->page_search_sq(
              query + (i * query_aligned_dim), recall_at, mem_L, L,
              query_result_ids_64.data() + (i * recall_at),
              query_result_dists[test_id].data() + (i * recall_at),
              optimized_beamwidth, search_io_limit, use_reorder_data, use_ratio, stats + i);
        }
      }else{
        if (use_coro) {
          std::cout << "Using coro" << std::endl;
#pragma omp parallel for schedule(dynamic, 1)
          for (_s64 i = 0; i < (int64_t) query_num; i++) {
            _pFlashIndex->async_search(
                query + (i * query_aligned_dim), recall_at, mem_L, L,
                query_result_ids_64.data() + (i * recall_at),
                query_result_dists[test_id].data() + (i * recall_at),
                optimized_beamwidth, search_io_limit, use_reorder_data,
                use_ratio, stats + i);
          }
          } else {
            // // 获取当前线程的 ID
            // int thread_id = omp_get_thread_num();

            // // 设置线程亲和性，绑定到特定的核心
            // // 假设每个线程绑定到 CPU 核心 0, 1, 2, ..., n-1
            // cpu_set_t cpuset;
            // CPU_ZERO(&cpuset);
            // CPU_SET(thread_id%num_threads, &cpuset);

            // // 设置当前线程的 CPU 亲和性
            // int ret = sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);
            // if (ret != 0) {
            //     std::cerr << "Error setting thread affinity!" << std::endl;
            // }
            bool pipeline = false;
            if (pipeline) {
#pragma omp parallel for schedule(dynamic, 1)
              for (_s64 i = 0; i < (int64_t) query_num; i++) {
                _pFlashIndex->page_search(
                    query + (i * query_aligned_dim), recall_at, mem_L, L,
                    query_result_ids_64.data() + (i * recall_at),
                    query_result_dists[test_id].data() + (i * recall_at),
                    optimized_beamwidth, search_io_limit, use_reorder_data,
                    use_ratio, stats + i);
              }
              } else {
#pragma omp parallel for schedule(dynamic, 1)
              for (_s64 i = 0; i < (int64_t) query_num; i++) {
                _pFlashIndex->page_search_no_pipeline(
                    query + (i * query_aligned_dim), recall_at, mem_L, L,
                    query_result_ids_64.data() + (i * recall_at),
                    query_result_dists[test_id].data() + (i * recall_at),
                    optimized_beamwidth, search_io_limit, use_reorder_data,
                    use_ratio, stats + i);
              }
              }
            }
      }
    } else {
      if(use_sq){
        std::cout << "diskann current not support sq..." << std::endl;
        exit(-1);
      }
#pragma omp parallel for schedule(dynamic, 1)
      for (_s64 i = 0; i < (int64_t) query_num; i++) {
        _pFlashIndex->cached_beam_search(
            query + (i * query_aligned_dim), recall_at, L,
            query_result_ids_64.data() + (i * recall_at),
            query_result_dists[test_id].data() + (i * recall_at),
            optimized_beamwidth, search_io_limit, use_reorder_data, stats + i, mem_L);
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
    
    auto mean_ious = diskann::get_mean_stats<float>(
        stats, query_num,
        [](const diskann::QueryStats &stats) { return stats.io_us; });

    auto mean_cpus = diskann::get_mean_stats<float>(
        stats, query_num,
        [](const diskann::QueryStats& stats) { return stats.cpu_us; });

    auto mean_hops = diskann::get_mean_stats<unsigned>(
        stats, query_num,
        [](const diskann::QueryStats& stats) { return stats.n_hops; });
    auto mean_cache_hits = diskann::get_mean_stats<unsigned>(
        stats, query_num,
        [](const diskann::QueryStats& stats) { return stats.n_cache_hits; });
    auto n_aff_cache_nodes = diskann::get_mean_stats<unsigned>(
        stats, query_num,
        [](const diskann::QueryStats& stats) { return stats.n_affinity_cache; });

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
                << ","<< "P99.9 Latency"
                << ","<< "P90 Latency"
                << ","<< "Mean IOs" 
                << ","<< "Mean IO (us)"
                << ","<< "CPU (us)"
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
                  << ","<< latency_999
                  << ","<< latency_90
                  << ","<< mean_ios
                  << ","<< mean_ious
                  << ","<< mean_cpus
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
  bool                  page_expansion = true;
  bool                  use_coro = false;
  float                 use_ratio = 1.0;
  bool use_sq = false;

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
    desc.add_options()("page_expansion", po::value<bool>(&page_expansion)->default_value(1),
                       "Use 1 for using page_expansion in search (default), 0 for node_expansion");
    desc.add_options()("use_coro", po::value<bool>(&use_coro)->default_value(0),
                       "Use 1 for using coroutine in IO, 0 for using coroutine (default).");
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

  try {
    if (data_type == std::string("float"))
      return search_disk_index<float>(metric, index_path_prefix,
                                      mem_index_path,
                                      result_path_prefix, query_file, gt_file,
                                      disk_file_path,
                                      num_threads, K, W, num_nodes_to_cache,
                                      search_io_limit, Lvec, mem_L, use_page_search, use_coro, use_ratio, use_reorder_data, use_sq);
    else if (data_type == std::string("int8"))
      return search_disk_index<int8_t>(metric, index_path_prefix,
                                       mem_index_path,
                                       result_path_prefix, query_file, gt_file,
                                       disk_file_path,
                                       num_threads, K, W, num_nodes_to_cache,
                                       search_io_limit, Lvec, mem_L, use_page_search, use_coro, use_ratio, use_reorder_data);
    else if (data_type == std::string("uint8"))
      return search_disk_index<uint8_t>(
          metric, index_path_prefix, mem_index_path, result_path_prefix, query_file, gt_file,
          disk_file_path, num_threads, K, W, num_nodes_to_cache, search_io_limit, Lvec, mem_L,
          use_page_search, use_coro, use_ratio, use_reorder_data);
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
