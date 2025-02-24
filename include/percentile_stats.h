// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <functional>
#include <algorithm>
#include <chrono>
#ifdef _WINDOWS
#include <numeric>
#endif
#include <string>
#include <vector>

#include "distance.h"
#include "parameters.h"

namespace diskann {

  struct BlockVisited{
    uint64_t block_id;
    std::chrono::high_resolution_clock::time_point timestamp; // set by who create this record. in us

    BlockVisited() : block_id(0), timestamp(std::chrono::high_resolution_clock::now()) {
    }

    BlockVisited(uint64_t block_id_in, std::chrono::high_resolution_clock::time_point timestamp_in)
     : block_id(block_id_in), timestamp(timestamp_in) {
    }
  };
  struct QueryStats {
    float total_us = 0;              // total time to process query in micros
    float io_us = 0;                 // total time spent in IO
    float cpu_us = 0;                // total time spent in CPU
    float executing_in_coro_us = 0;  // total time spent in coro
    float bubble_time_us = 0;


    float mean_io_time;             // 单个io请求端到端时间
    float mean_io_push_queue_time;  // #生成IO请求，提交到队列的时间 = ts_beign
    float mean_io_submit_time;      // #通过spdk提交io时间 = now - ts_begin
    float mean_io_complete_time;    // #通过spdk完成io的时间 = now - ts_begin
    float mean_io_resume_time;      // 对应的coro恢复执行的时间 = now - ts_begin

    unsigned n_4k = 0;              // # of 4kB reads
    unsigned n_8k = 0;              // # of 8kB reads
    unsigned n_12k = 0;             // # of 12kB reads
    unsigned n_ios = 0;             // total # of IOs issued
    unsigned n_io_returns = 0;      // total # of IOs uring returned
    unsigned read_size = 0;         // total # of bytes read
    unsigned n_cmps_saved = 0;      // # cmps saved
    unsigned n_cmps = 0;            // # cmps
    unsigned n_cache_hits = 0;      // # cache_hits
    unsigned n_hops = 0;            // # search hops
    unsigned n_affinity_cache = 0;  // # affinity nodes
    


    std::vector<BlockVisited> block_visited_queue;
  };

  struct ThreadStats {
    float total_us = 0;  // total time to executing one thread in micross
    float io_us = 0;     // total time spent in IO
    float cpu_us = 0;    // total time spent in CPU
    float executing_in_coro_us = 0;    // total time spent in coro
    float io_submit_us = 0;
    float io_reap_us = 0;

    float compute_us = 0;    // total time spent in distinct

    float scheduler_total_us = 0;
    float scheduler_cpu_us = 0;
    float scheduler_wait_us = 0;

    float wait_ring_lock_us = 0;
    float awaiter_time_us = 0;
    float awaiter_middle_time_us = 0;


    unsigned n_ios = 0;
    unsigned n_hops = 0;        // # search hops
    
    
  };

  template<typename T>
  inline T get_percentile_stats(
      QueryStats *stats, uint64_t len, float percentile,
      const std::function<T(const QueryStats &)> &member_fn) {
    std::vector<T> vals(len);
    for (uint64_t i = 0; i < len; i++) {
      vals[i] = member_fn(stats[i]);
    }

    std::sort(vals.begin(), vals.end(),
              [](const T &left, const T &right) { return left < right; });

    auto retval = vals[(uint64_t)(percentile * len)];
    vals.clear();
    return retval;
  }

  template<typename T>
  inline double get_mean_stats(
      QueryStats *stats, uint64_t len,
      const std::function<T(const QueryStats &)> &member_fn) {
    double avg = 0;
    for (uint64_t i = 0; i < len; i++) {
      avg += (double) member_fn(stats[i]);
    }
    return avg / len;
  }

  template<typename T>
  inline double get_mean_vec(std::vector<T> &vec) {
    double avg = 0;
    size_t len = vec.size();
    for (size_t i = 0; i < len; i++) {
      avg += (double) vec[i];
    }
    return avg / len;
  }

  template<typename T>
  inline double get_sum_stats(
      QueryStats *stats, uint64_t len,
      const std::function<T(const QueryStats &)> &member_fn) {
    double sum = 0;
    for (uint64_t i = 0; i < len; i++) {
      sum += (double) member_fn(stats[i]);
    }
    return sum;
  }

  // The following two functions are used when getting statistics while range searching on only queries with
  // non-zero gt lengths
  template<typename T>
  inline T get_percentile_stats_gt(
      QueryStats *stats, uint64_t len, float percentile,
      const std::function<T(const QueryStats &)> &member_fn, std::vector<std::vector<uint32_t>> &gt) {
    std::vector<T> vals;
    for (uint64_t i = 0; i < len; i++) {
      if (gt[i].size()) vals.push_back(member_fn(stats[i]));
    }

    std::sort(vals.begin(), vals.end(),
              [](const T &left, const T &right) { return left < right; });

    auto retval = vals[(uint64_t)(percentile * vals.size())];
    vals.clear();
    return retval;
  }

  template<typename T>
  inline double get_mean_stats_gt(
      QueryStats *stats, uint64_t len,
      const std::function<T(const QueryStats &)> &member_fn, std::vector<std::vector<uint32_t>> &gt) {
    uint32_t cnt = 0;
    double avg = 0;
    for (uint64_t i = 0; i < len; i++) {
      if (gt[i].size()) {
        ++cnt;
        avg += (double) member_fn(stats[i]);
      }
    }
    return avg / cnt;
  }
}  // namespace diskann
