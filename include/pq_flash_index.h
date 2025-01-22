// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once
#include <cassert>
#include <sstream>
#include <stack>
#include <string>
#include "tsl/robin_map.h"
#include "tsl/robin_set.h"

#include "aligned_file_reader.h"
#include "concurrent_queue.h"
#include "neighbor.h"
#include "parameters.h"
#include "percentile_stats.h"
#include "pq_table.h"
#include "utils.h"
#include "windows_customizations.h"
#include "index.h"
#include "pq_flash_index_utils.h"

#define MAX_GRAPH_DEGREE 512
#define MAX_N_CMPS 16384
#define SECTOR_LEN (_u64) 4096
#define MAX_N_SECTOR_READS 128
#define MAX_PQ_CHUNKS 256

#define FULL_PRECISION_REORDER_MULTIPLIER 3

namespace diskann {
  void boost_example();
  template<typename T>
  struct QueryScratch {
    T *  coord_scratch = nullptr;  // MUST BE AT LEAST [MAX_N_CMPS * data_dim]
    _u64 coord_idx = 0;            // index of next [data_dim] scratch to use

    char *sector_scratch =
        nullptr;          // MUST BE AT LEAST [MAX_N_SECTOR_READS * SECTOR_LEN]
    _u64 sector_idx = 0;  // index of next [SECTOR_LEN] scratch to use

    float *aligned_pqtable_dist_scratch =
        nullptr;  // MUST BE AT LEAST [256 * NCHUNKS]
    float *aligned_dist_scratch =
        nullptr;  // MUST BE AT LEAST diskann MAX_DEGREE
    _u8 *aligned_pq_coord_scratch =
        nullptr;  // MUST BE AT LEAST  [N_CHUNKS * MAX_DEGREE]
    T *    aligned_query_T = nullptr;
    float *aligned_query_float = nullptr;

    // TODO 为每个线程单独分配prefetch的存储空间
    // prefetch_nhood_cache
    unsigned *                                    affinity_nhood_cache_buf = nullptr;
    tsl::robin_map<_u32, std::pair<_u32, _u32 *>> affinity_nhood_cache;

    // prefetch_coord_cache
    T *                       affinity_coord_cache_buf = nullptr;
    tsl::robin_map<_u32, T *> affinity_coord_cache;

    tsl::robin_set<_u64> *visited = nullptr;
    tsl::robin_set<unsigned> *page_visited = nullptr;

    void reset() {
      coord_idx = 0;
      sector_idx = 0;
      visited->clear();  // does not deallocate memory.
      page_visited->clear();
    }

    void refresh_and_init_affinity_cache(size_t num_cached_nodes,_u64 max_degree, _u64 aligned_dim){
        if (this->affinity_nhood_cache_buf != nullptr){
            delete [] this->affinity_nhood_cache_buf;
            diskann::aligned_free(this->affinity_coord_cache_buf);
            this->affinity_nhood_cache.clear();
            this->affinity_coord_cache.clear();
        }

        affinity_nhood_cache_buf = new unsigned[num_cached_nodes * (max_degree + 1)];
        memset(affinity_nhood_cache_buf, 0, num_cached_nodes * (max_degree + 1));

        _u64 coord_cache_buf_len = num_cached_nodes * aligned_dim;
        diskann::alloc_aligned((void **) &affinity_coord_cache_buf,
                           coord_cache_buf_len * sizeof(T), 8 * sizeof(T));
        memset(affinity_coord_cache_buf, 0, coord_cache_buf_len * sizeof(T));
    }
  };

  template<typename T>
  struct ThreadData {
    QueryScratch<T> scratch;
    IOContext       ctx;
  };
  
  template<typename T>
  struct PageSearchPersistData {
    std::vector<Neighbor> ret_set;
    NeighborVec kicked;
    std::vector<Neighbor> full_ret_set;
    unsigned cur_list_size;
    ThreadData<T> thread_data;
    float query_norm;
  };

  template<typename T>
  class PQFlashIndex {
   public:
    DISKANN_DLLEXPORT PQFlashIndex(
        std::shared_ptr<AlignedFileReader> &fileReader,
        const bool use_page_search,
        diskann::Metric                     metric = diskann::Metric::L2, bool use_sq = false);
    DISKANN_DLLEXPORT ~PQFlashIndex();

    // load id to page id and graph partition layout
    DISKANN_DLLEXPORT void load_partition_data(const std::string &index_prefix,
        const _u64 nnodes_per_sector = 0, const _u64 num_points = 0);

    // load affinity blocks to each block.
    DISKANN_DLLEXPORT void load_affinity_data(const std::string &index_prefix);

#ifdef EXEC_ENV_OLS
    DISKANN_DLLEXPORT int load(diskann::MemoryMappedFiles &files,
                               uint32_t num_threads, const char *index_prefix);
#else
    // load compressed data, and obtains the handle to the disk-resident index
    DISKANN_DLLEXPORT int  load(uint32_t num_threads, const char *index_prefix,
        const std::string& disk_index_path);
#endif

    DISKANN_DLLEXPORT void load_mem_index(Metric metric, const size_t query_dim,
        const std::string &mem_index_path, const _u32 num_threads,
        const _u32 mem_L);

    DISKANN_DLLEXPORT void load_cache_list(std::vector<uint32_t> &node_list);


    // Prefetch related_
    DISKANN_DLLEXPORT void init_prefetch_cache_list(const _u64 prefetch_block_size);

#ifdef EXEC_ENV_OLS
    DISKANN_DLLEXPORT void generate_cache_list_from_sample_queries(
        MemoryMappedFiles &files, std::string sample_bin, _u64 l_search,
        _u64 beamwidth, _u64 num_nodes_to_cache, uint32_t nthreads,
        std::vector<uint32_t> &node_list);
#else
    DISKANN_DLLEXPORT void generate_cache_list_from_sample_queries(
        std::string sample_bin, _u64 l_search, _u64 beamwidth,
        _u64 num_nodes_to_cache, uint32_t num_threads,
        std::vector<uint32_t> &node_list, bool use_pagesearch, const _u32 mem_L);
#endif

    DISKANN_DLLEXPORT void cache_bfs_levels(_u64 num_nodes_to_cache,
                                            std::vector<uint32_t> &node_list);

    DISKANN_DLLEXPORT void cached_beam_search(
        const T *query, const _u64 k_search, const _u64 l_search, _u64 *res_ids,
        float *res_dists, const _u64 beam_width,
        const bool use_reorder_data = false, QueryStats *stats = nullptr, const _u32 mem_L = 0);

    DISKANN_DLLEXPORT void cached_beam_search(
        const T *query, const _u64 k_search, const _u64 l_search, _u64 *res_ids,
        float *res_dists, const _u64 beam_width, const _u32 io_limit,
        const bool use_reorder_data = false, QueryStats *stats = nullptr, const _u32 mem_L = 0);

    DISKANN_DLLEXPORT void generate_node_nbrs_freq(
        const std::string& freq_save_path,
        const size_t query_num,
        const T *query, const size_t query_aligned_dim,
        const _u64 k_search, const _u64 l_search, _u64 *res_ids,
        float *res_dists, const _u64 beam_width, const _u32 io_limit,
        const bool use_reorder_data = false, QueryStats *stats = nullptr, const _u32 mem_L = 0);

    DISKANN_DLLEXPORT void page_search(
        const T *query, const _u64 k_search, const _u32 mem_L, const _u64 l_search, _u64 *res_ids,
        float *res_dists, const _u64 beam_width, const _u32 io_limit,
        const bool use_reorder_data = false, const float use_ratio = 1.0f, QueryStats *stats = nullptr);
    DISKANN_DLLEXPORT void page_search_no_pipeline(
        const T *query, const _u64 k_search, const _u32 mem_L, const _u64 l_search, _u64 *res_ids,
        float *res_dists, const _u64 beam_width, const _u32 io_limit,
        const bool use_reorder_data = false, const float use_ratio = 1.0f, QueryStats *stats = nullptr);
    DISKANN_DLLEXPORT void page_search_sq(
        const T *query, const _u64 k_search, const _u32 mem_L, const _u64 l_search, _u64 *res_ids,
        float *res_dists, const _u64 beam_width, const _u32 io_limit,
        const bool use_reorder_data = false, const float use_ratio = 1.0f, QueryStats *stats = nullptr);
    DISKANN_DLLEXPORT void async_search(
        const T *query, const _u64 k_search, const _u32 mem_L, const _u64 l_search, _u64 *res_ids,
        float *res_dists, const _u64 beam_width, const _u32 io_limit,
        const bool use_reorder_data = false, const float use_ratio = 1.0f, QueryStats *stats = nullptr);

    DISKANN_DLLEXPORT _u32 range_search_iter_knn(const T *query1, const double range,
                                        const _u32          mem_L,
                                        const _u64          min_l_search,
                                        const _u64          max_l_search,
                                        std::vector<_u64> & indices,
                                        std::vector<float> &distances,
                                        const _u64          min_beam_width,
                                        const float         page_search_use_ratio = 1.0f,
                                        QueryStats *        stats = nullptr);

    DISKANN_DLLEXPORT _u32 custom_range_search(const T *query1,
                                        const double range,
                                        const _u32          mem_L,
                                        const _u64          knn_min_l_search,
                                        const _u64          max_l_search,
                                        std::vector<_u64> & indices,
                                        std::vector<float> &distances,
                                        const _u32          beam_width,
                                        const float         page_search_use_ratio,
                                        const _u32          kicked_size,
                                        const _u32          custom_round_num,
                                        QueryStats *        stats = nullptr);

    DISKANN_DLLEXPORT _u32 custom_range_search_iter_page_search(
                                     const T *query1, const double range,
                                     const _u32          mem_L,
                                     std::vector<unsigned> &upper_mem_tags,
                                     std::vector<float> &upper_mem_dis,
                                     const _u64          min_l_search,
                                     const _u64          max_l_search,
                                     std::vector<_u64> & indices,
                                     std::vector<float> &distances,
                                     const _u64          min_beam_width,
                                     const float         page_search_use_ratio,
                                     const _u32          kicked_size,
                                     QueryStats *        stats = nullptr);

    std::shared_ptr<AlignedFileReader> &reader;

   protected:
    DISKANN_DLLEXPORT void use_medoids_data_as_centroids();
    DISKANN_DLLEXPORT void setup_thread_data(_u64 nthreads);
    DISKANN_DLLEXPORT void destroy_thread_data();

   private:
    // index info
    // nhood of node `i` is in sector: [i / nnodes_per_sector]
    // offset in sector: [(i % nnodes_per_sector) * max_node_len]
    // nnbrs of node `i`: *(unsigned*) (buf)
    // nbrs of node `i`: ((unsigned*)buf) + 1
    _u64 max_node_len = 0, nnodes_per_sector = 0, max_degree = 0;

    // Data used for searching with re-order vectors
    _u64 ndims_reorder_vecs = 0, reorder_data_start_sector = 0,
         nvecs_per_sector = 0;

    diskann::Metric metric = diskann::Metric::L2;

    // used only for inner product search to re-scale the result value
    // (due to the pre-processing of base during index build)
    float max_base_norm = 0.0f;

    // data info
    _u64 num_points = 0;
    _u64 num_frozen_points = 0;
    _u64 frozen_location = 0;
    _u64 data_dim = 0;
    _u64 disk_data_dim = 0;  // will be different from data_dim only if we use
                             // PQ for disk data (very large dimensionality)
    _u64 aligned_dim = 0;
    _u64 disk_bytes_per_point = 0;

    std::string                        disk_index_file;
    std::vector<std::pair<_u32, _u32>> node_visit_counter;

    // PQ data
    // n_chunks = # of chunks ndims is split into
    // data: _u8 * n_chunks
    // chunk_size = chunk size of each dimension chunk
    // pq_tables = float* [[2^8 * [chunk_size]] * n_chunks]
    _u8 *             data = nullptr;
    _u64              n_chunks;
    FixedChunkPQTable pq_table;

    // distance comparator
    std::shared_ptr<Distance<T>>     dist_cmp;
    std::shared_ptr<Distance<float>> dist_cmp_float;

    // for very large datasets: we use PQ even for the disk resident index
    bool              use_disk_index_pq = false;
    _u64              disk_pq_n_chunks = 0;
    FixedChunkPQTable disk_pq_table;

    // medoid/start info

    // graph has one entry point by default,
    // we can optionally have multiple starting points
    uint32_t *medoids = nullptr;
    // defaults to 1
    size_t num_medoids;
    // by default, it is empty. If there are multiple
    // centroids, we pick the medoid corresponding to the
    // closest centroid as the starting point of search
    float *centroid_data = nullptr;

    // nhood_cache
    unsigned *                                    nhood_cache_buf = nullptr;
    tsl::robin_map<_u32, std::pair<_u32, _u32 *>> nhood_cache;

    // coord_cache
    T *                       coord_cache_buf = nullptr;
    tsl::robin_map<_u32, T *> coord_cache;

    // thread-specific scratch
    ConcurrentQueue<ThreadData<T>> thread_data;
    _u64                           max_nthreads;
    bool                           load_flag = false;
    bool                           count_visited_nodes = false;
    bool                           count_visited_nbrs = false;
    bool                           reorder_data_exists = false;
    _u64                           reoreder_data_offset = 0;

    // in-memory navigation graph
    std::unique_ptr<Index<T, uint32_t>> mem_index_;
    std::vector<unsigned> memid2diskid_;

    // page search
    bool use_page_search_ = true;
    std::vector<unsigned> id2page_;
    std::vector<std::vector<unsigned>> gp_layout_;


    // affinity prefetch
    // bool use_affinity_ = true;
    bool use_affinity_ = false;
    _u64 affinity_size_ = 1;
    std::vector<std::vector<unsigned>> affinity_prefetch_dict_;


    // 是否打印详细输出 注意 可能会造成线程间输出混乱
    bool verbose_ = false;

    bool use_sq_ = false;
    float* mins = nullptr;
    float* frac = nullptr;
    // tsl::robin_map<_u32, char*> page_cache_;
    // tsl::robin_map<_u32, char*> node_cache_;

    // count the visit frequency of the neighbors
    // the length of the vector is equal to the total number of vectors in the base
    // idx = node_id, the key represents the neighbor id, value is its count
    std::vector<std::unordered_map<_u32, _u32>> nbrs_freq_counter_;

    void init_node_visit_counter();

    void page_search_interim(
      const _u64 k_search, const _u32 mem_L, const _u64 l_search, _u64 *indices,
      float *distances, const _u64 beam_width, const _u32 io_limit,
      const bool use_reorder_data = false, const float use_ratio = 1.0f, QueryStats *stats = nullptr,
      PageSearchPersistData<T>* persist_data = nullptr);

#ifdef EXEC_ENV_OLS
    // Set to a larger value than the actual header to accommodate
    // any additions we make to the header. This is an outer limit
    // on how big the header can be.
    static const int HEADER_SIZE = SECTOR_LEN;
    char *           getHeaderBytes();
#endif
  };
}  // namespace diskann
