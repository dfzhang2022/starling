#include <immintrin.h>
#include <iomanip> 
#include <cstdlib>
#include <cstring>
#include "logger.h"
#include "percentile_stats.h"
#include "pq_flash_index.h"
#include "timer.h"

#include "file.h"
#include "io_uring.h"
#include "overlap_utils.h"

namespace diskann {
  void cb(void *ctx, const struct spdk_nvme_cpl *cpl) {
    if ((spdk_nvme_cpl_is_error(cpl))) {
      std::cout << "I/O error status: "
                 << spdk_nvme_cpl_get_status_string(&cpl->status);
    }else{
    std::atomic<int> *p = (std::atomic<int> *)ctx;
    p->fetch_add(1);
  }

  }
  void cb_in_queryIO_way(void *ctx, const struct spdk_nvme_cpl *cpl) {
    if ((spdk_nvme_cpl_is_error(cpl))) {
      std::cout << "I/O error status: "
                 << spdk_nvme_cpl_get_status_string(&cpl->status);
    }else{
    QueryIO *p = (QueryIO *)ctx;
    p->completed++;
    if(p->completed == p->io_num){
      p->complete_from_spdk();
    }
    p->ptr_to_atomic_flag->fetch_add(1);
  }

  }


  void thread_stat_print(ThreadStats* thread_stat, bool is_coro = false){
    std::cout << "in coro:"
              << (thread_stat->executing_in_coro_us) / thread_stat->total_us
              << ", in awaiter :"
              << thread_stat->awaiter_time_us / thread_stat->total_us
              << ", submit wait:"
              << thread_stat->awaiter_middle_time_us / thread_stat->total_us
              << ", sche_all: "
              << thread_stat->scheduler_total_us / thread_stat->total_us
              << ", cpu usage:"
              << (thread_stat->executing_in_coro_us) / thread_stat->total_us
              << std::endl;
  }


  double calculateBlockIdFrequency(const std::vector<AlignedRead>& reads, int& distinctNum, int& allNum) {
    // Use an unordered_map to count the occurrences of each block_id
    std::unordered_map<uint64_t, uint64_t> blockIdCount;
  
    // Iterate over all AlignedRead elements and count the occurrences of block_id
    for (const auto& read : reads) {
      if(blockIdCount.find(read.block_id)==blockIdCount.end()){
        blockIdCount[read.block_id] = 1;
      }else{
        blockIdCount[read.block_id]++;
      }
    }
  
    // Get the total number of reads
    size_t totalReads = reads.size();
    if (totalReads == 0) {
        // std::cout << "No reads to process." << std::endl;
        return 0;
    }

    size_t diffReadNum = blockIdCount.size();
    if(diffReadNum<totalReads){
      // std::cout << "map#"<<diffReadNum<<" ,totalReads: "<<totalReads << std::endl;
    }
  
    // Output the count and repetition rate for each block_id
    // for (const auto& entry : blockIdCount) {
    //     uint64_t blockId = entry.first;
    //     uint64_t count = entry.second;
    //     double repetitionRate = static_cast<double>(count) / totalReads;
        
    //     // Print the block_id, its count, and the repetition rate as a percentage
    //     std::cout << "Block ID: " << blockId
    //               << " | Count: " << count
    //               << " | Repetition Rate: " << repetitionRate * 100.0 << "%" << std::endl;
    // }
  
    distinctNum = diffReadNum;
    allNum = totalReads;
    return (totalReads - diffReadNum)/totalReads;
  }
  template<typename T>
  void LibaioIORegisterAwaiter<T>::await_suspend(
      cppcoro::coroutine_handle<> handle) {
    // TODO 增加thread id -> ring#的逻辑
    this->ext_data_.handle = handle;
    int            thread_id_local = this->ext_data_.thread_id;
    int            coro_id_local = this->ext_data_.coro_idx;
    int            cnt = this->aligned_read_vec_.size();
    diskann::Timer ring_mutex_timer, awaiter_timer, new_timer;
    ring_mutex_timer.reset();
    awaiter_timer.reset();
    new_timer.reset();


    //设置回调函数指针
    pq_flash_index_->set_handle(thread_id_local, coro_id_local, handle);
    
    new_timer.reset();
    int returned = pq_flash_index_->libaio_submit(this->aligned_read_vec_, thread_id_local, coro_id_local);
    // pq_flash_index_->register_io(thread_id_local, coro_id_local, cnt);
    // int returned = pq_flash_index_->batch_push(this->aligned_read_vec_, thread_id_local);
    thread_stat_->awaiter_middle_time_us += new_timer.elapsed();
    if(returned!=cnt){
      std::cout<<"[Worker Thread] push "<< cnt<<" but return "<<returned<<std::endl;
    }


    // if (true) {
    if (false) {
      std::cout << "[Worker Thread]Awaiter Issued num: " << returned
                << ", thread: " << thread_id_local
                << ", coro: " << coro_id_local << std::endl;
    }

    thread_stat_->awaiter_time_us += awaiter_timer.elapsed();
  }
  template<typename T>
  void NewIORegisterAwaiter<T>::await_suspend(
      cppcoro::coroutine_handle<> handle) {
    // TODO 增加thread id -> ring#的逻辑
    this->ext_data_.handle = handle;
    int            thread_id_local = this->ext_data_.thread_id;
    int            coro_id_local = this->ext_data_.coro_idx;
    int            cnt = this->aligned_read_vec_.size();
    diskann::Timer ring_mutex_timer, awaiter_timer, new_timer;
    ring_mutex_timer.reset();
    awaiter_timer.reset();
    new_timer.reset();

    pq_flash_index_->set_handle(thread_id_local, coro_id_local, handle);
    pq_flash_index_->register_io(thread_id_local, coro_id_local, cnt);
    new_timer.reset();
    int returned = pq_flash_index_->batch_push(this->aligned_read_vec_, thread_id_local);
    thread_stat_->awaiter_middle_time_us += new_timer.elapsed();
    if(returned!=cnt){
      std::cout<<"[Worker Thread] push "<< cnt<<" but return "<<returned<<std::endl;
    }

    // if (true) {
    if (false) {
      std::cout << "[Worker Thread]Awaiter Issued num: " << returned
                << ", thread: " << thread_id_local
                << ", coro: " << coro_id_local << std::endl;
    }

    thread_stat_->awaiter_time_us += awaiter_timer.elapsed();
  }
  template<typename T>
  void IORegisterAwaiter<T>::await_suspend(cppcoro::coroutine_handle<> handle) {

    // TODO 增加thread id -> ring#的逻辑
      this->ext_data_.handle = handle;
      int thread_id_local = this->ext_data_.thread_id;
      int coro_id_local = this->ext_data_.coro_idx;
      int cnt = 0;
      diskann::Timer ring_mutex_timer, awaiter_timer, new_timer;
      ring_mutex_timer.reset();
      awaiter_timer.reset();
      // pq_flash_index_->ring_mutex_lock(this->ext_data_.thread_id,this->ext_data_.coro_idx);
      // pq_flash_index_->ring_mutex_lock(thread_id_local, coro_id_local);
      thread_stat_->wait_ring_lock_us += ring_mutex_timer.elapsed();
      io_uring* ring_ptr = this->pq_flash_index_->get_iouring(thread_id_local, coro_id_local);
      // std::cout<<"aligned_read_vec_.size()"<<aligned_read_vec_.size()<<std::endl;
      for (size_t i = 0; i < this->aligned_read_vec_.size(); i++) {
        // std::cout << "Issue io:"<<i<< std::endl;
        AlignedRead  *tmp_ptr = &aligned_read_vec_[i];
        tmp_ptr->thread_id = thread_id_local;
        tmp_ptr->coro_id = coro_id_local;
        // pq_flash_index_->ring_mutex_lock(thread_id_local, coro_id_local);
        
        io_uring_sqe *sqe =
            io_uring_get_sqe(ring_ptr);
        
        // pq_flash_index_->ring_mutex_unlock(thread_id_local, coro_id_local);
        // if (sqe == nullptr) {
        //   // throw BQANN::SubmissionQueueFullError{};
        //   std::cout<<"SubmissionQueueFullError"<<std::endl;
        //   continue;
        // }
        while(sqe == nullptr){
          sqe =
            io_uring_get_sqe(ring_ptr);
        }
        
        // std::cout << "Issue io:"<<i<< std::endl;
        io_uring_prep_read(sqe, this->pq_flash_index_->get_index_fd(),
                           tmp_ptr->buf, tmp_ptr->len, tmp_ptr->offset);
        // std::cout << "[B]Issue io:"<<i<< std::endl;
        io_uring_sqe_set_data(sqe, tmp_ptr);
        
        // std::cout << "[C]Issue io:"<<i<< std::endl;
        cnt++;
        // std::cout<<"bbb"<<std::endl;
      }
      // pq_flash_index_->handles_map[thread_id_local][coro_id_local] = handle;
      pq_flash_index_->set_handle(thread_id_local,coro_id_local,handle);
      pq_flash_index_->register_io(thread_id_local, coro_id_local,cnt);
      // pq_flash_index_->ring_mutex_lock(thread_id_local, coro_id_local);
      new_timer.reset();
      // io_uring_submit(ring_ptr);
      int res = io_uring_submit(ring_ptr);
      if (res!=cnt) {
          // printf("submit: %s\n", strerror(-res));
          // assert(0);
          std::cout<<"io_uring_submit submitted less: "<<res<<std::endl;
      }
      thread_stat_->awaiter_middle_time_us+= new_timer.elapsed();
      // pq_flash_index_->ring_mutex_unlock(thread_id_local, coro_id_local);
      thread_stat_->awaiter_time_us += awaiter_timer.elapsed();
    }


  void printFirst128Chars(AlignedRead &req) {
    if (req.buf == nullptr) {
      std::cout << "The pointer (char*)(req.buf) is not initialized."
                << std::endl;
      return;
    }
    char* charBuf = static_cast<char*>(req.buf);  // 强制转换为char*类型
    // 确保有足够的字符可以打印
    size_t availableChars = std::min(static_cast<size_t>(128), req.len);
    for (size_t i = 0; i < availableChars; ++i) {
      std::cout<< std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(static_cast<unsigned char>(charBuf[i])) << " ";
    }
    std::cout << std::endl;
  }
  static cppcoro::task<void> AsyncBatchRead(
      std::vector<AlignedRead> &aligned_read_vec, const BQANN::File &data_file,
      BQANN::IOUring &ring, BQANN::Countdown &countdown, int coro_id) {
    // std::cout<<"In AsyncBatchRead():
    // aligned_read_vec_.size()"<<aligned_read_vec.size()<<std::endl; for(int i
    // = 0;i<1;i++){
    co_await data_file.AsyncBatchReadBlock(ring, aligned_read_vec, coro_id);
    // std::cout<<"Done "<<i<<" batch read in coro id:"<<coro_id<<std::endl;
    // }
    // std::cout<<"Done all batch read in coro id:"<<coro_id<<std::endl;
    countdown.Decrement();
  }
  
  template<typename T>
  static cppcoro::task<void> tmp_task(PQFlashIndex<T>          *index,std::vector<AlignedRead> &aligned_read_vec, int thread_id,
                      int coro_id) {
    // std::cout<<"In AsyncBatchRead():
    // aligned_read_vec_.size()"<<aligned_read_vec.size()<<std::endl; for(int i
    // = 0;i<1;i++){
    co_await IORegisterAwaiter<T>(index, aligned_read_vec,thread_id,coro_id);
    // std::cout<<"Done "<<i<<" batch read in coro id:"<<coro_id<<std::endl;
    // }
    // std::cout<<"Done all batch read in coro id:"<<coro_id<<std::endl;
    // countdown.Decrement();
  }
  

  template class IORegisterAwaiter<_u8>;
  template class IORegisterAwaiter<_s8>;
  template class IORegisterAwaiter<float>;


    template<typename T>
  cppcoro::task<void> PQFlashIndex<T>::query_coro(
      const T *query1, const size_t query_num, const _u64 k_search,
      const _u32 mem_L, const _u64 l_search, _u64 *indices, float *distances,
      const _u64 beam_width, const _u32 io_limit, const bool use_reorder_data,
      const float use_ratio, QueryStats *stats, int thread_id, int coro_id,
      BQANN::Countdown &countdown, ThreadStats* thread_stat) {
        // std::cout << "Get into coro."<< std::endl;
    QueryScratch<T> scratch = this->coro_data.pop();
    while (scratch.sector_scratch == nullptr) {
      this->coro_data.wait_for_push_notify();
      scratch = this->coro_data.pop();
    }
    Timer thread_timer;
    thread_timer.reset();
    // std::cout << "Get coro data."<< std::endl;
    // int q_id = thread_id + coro_id;
    // Continuously obtain new query IDs
    for (size_t q_id = thread_id * max_ncoroutines + coro_id;;
         q_id = q_id + max_nthreads * max_ncoroutines) {

      if (q_id >= query_num) {

        this->coro_data.push(scratch);
        this->coro_data.push_notify_all();
        if (verbose_) {
          std::cout << "Coro Exit." << thread_id << "," << coro_id << std::endl;
        }
        thread_stat->executing_in_coro_us += (double) thread_timer.elapsed();
        countdown.Decrement();
        co_return;
      }
      // if(q_id%1000 == 0)
      // std::cout<<"Executing Q#"<<q_id<<", "<<thread_id<<", "<<coro_id<<std::endl;


    

      const T     *this_coro_query = query1 + (q_id * this->query_aligned_dim);
      _u64  *this_coro_result_ids_64 = indices + (q_id * k_search);
      float *this_coro_result_distances = distances + (q_id * k_search);
      QueryStats  *this_coro_stats = stats + q_id;

      // copy query to coroutine specific aligned and allocated memory (for
      // distance calculations we need aligned data)
      float        query_norm = 0;
      const T     *query = scratch.aligned_query_T;
      const float *query_float = scratch.aligned_query_float;
      
      uint32_t query_dim = metric == diskann::Metric::INNER_PRODUCT
                               ? this->data_dim - 1
                               : this->data_dim;

      for (uint32_t i = 0; i < query_dim; i++) {
        scratch.aligned_query_float[i] = this_coro_query[i];
        scratch.aligned_query_T[i] = this_coro_query[i];
        query_norm += query1[i] * this_coro_query[i];
      }

      auto query_scratch = &(scratch);

      // reset query
      query_scratch->reset();

      // pointers to buffers for data
      T *data_buf = query_scratch->coord_scratch;
      // _mm_prefetch((char *) data_buf, _MM_HINT_T1);

      // sector scratch
      char *sector_scratch = query_scratch->sector_scratch;
      _u64 &sector_scratch_idx = query_scratch->sector_idx;

      // query <-> PQ chunk centers distances
      float *pq_dists = query_scratch->aligned_pqtable_dist_scratch;
      pq_table.populate_chunk_distances(query_float, pq_dists);

      // query <-> neighbor list
      float *dist_scratch = query_scratch->aligned_dist_scratch;
      _u8   *pq_coord_scratch = query_scratch->aligned_pq_coord_scratch;

      Timer query_timer, io_timer, cpu_timer, coro_timer;
      query_timer.reset();
      coro_timer.reset();
      std::vector<Neighbor>     retset(l_search + 1);
      tsl::robin_set<_u64>     &visited = *(query_scratch->visited);
      tsl::robin_set<unsigned> &page_visited = *(query_scratch->page_visited);
      unsigned                  cur_list_size = 0;

      std::vector<Neighbor> full_retset;
      full_retset.reserve(4096);
      _u32  best_medoid = 0;
      float best_dist = (std::numeric_limits<float>::max)();
      std::vector<SimpleNeighbor> medoid_dists;
      for (_u64 cur_m = 0; cur_m < num_medoids; cur_m++) {
        float cur_expanded_dist = dist_cmp_float->compare(
            query_float, centroid_data + aligned_dim * cur_m,
            (unsigned) aligned_dim);
        if (cur_expanded_dist < best_dist) {
          best_medoid = medoids[cur_m];
          best_dist = cur_expanded_dist;
        }
      }

      // lambda to batch compute query<-> node distances in PQ space
      auto compute_pq_dists = [this, pq_coord_scratch, pq_dists](
                                  const unsigned *ids, const _u64 n_ids,
                                  float *dists_out) {
        pq_flash_index_utils::aggregate_coords(
            ids, n_ids, this->data, this->n_chunks, pq_coord_scratch);
        pq_flash_index_utils::pq_dist_lookup(
            pq_coord_scratch, n_ids, this->n_chunks, pq_dists, dists_out);
      };

      // 将id的点强制push进入full_retset
      auto compute_extact_dists_and_push = [&](const char    *node_buf,
                                               const unsigned id) -> float {
        T *node_fp_coords_copy = data_buf;
        memcpy(node_fp_coords_copy, node_buf, disk_bytes_per_point);
        float cur_expanded_dist = dist_cmp->compare(query, node_fp_coords_copy,
                                                    (unsigned) aligned_dim);
        full_retset.push_back(Neighbor(id, cur_expanded_dist, true));
        return cur_expanded_dist;
      };

      auto compute_and_push_nbrs = [&](const char *node_buf, unsigned &nk) {
        unsigned *node_nbrs = OFFSET_TO_NODE_NHOOD(node_buf);
        unsigned  nnbrs = *(node_nbrs++);
        unsigned  nbors_cand_size = 0;
        for (unsigned m = 0; m < nnbrs; ++m) {
          if (visited.find(node_nbrs[m]) == visited.end()) {
            node_nbrs[nbors_cand_size++] = node_nbrs[m];
            visited.insert(node_nbrs[m]);
          }
        }
        if (nbors_cand_size) {
          compute_pq_dists(node_nbrs, nbors_cand_size, dist_scratch);
          for (unsigned m = 0; m < nbors_cand_size; ++m) {
            const int   nbor_id = node_nbrs[m];
            const float nbor_dist = dist_scratch[m];
            if (this_coro_stats != nullptr) {
              this_coro_stats->n_cmps++;
            }
            if (nbor_dist >= retset[cur_list_size - 1].distance &&
                (cur_list_size == l_search))
              continue;
            Neighbor nn(nbor_id, nbor_dist, true);
            // Return position in sorted list where nn inserted
            auto r = InsertIntoPool(retset.data(), cur_list_size, nn);
            if (cur_list_size < l_search)
              ++cur_list_size;
            // nk logs the best position in the retset that was updated due to
            // neighbors of n.
            if (r < nk)
              nk = r;
          }
        }
      };

      // 这个是原本就有的结构
      // 计算 node_ids[] 若干个节点的pq距离并加入到retset中
      auto compute_and_add_to_retset = [&](const unsigned *node_ids,
                                           const _u64      n_ids) {
        compute_pq_dists(node_ids, n_ids, dist_scratch);
        for (_u64 i = 0; i < n_ids; ++i) {
          retset[cur_list_size].id = node_ids[i];
          retset[cur_list_size].distance = dist_scratch[i];
          retset[cur_list_size++].flag = true;
          visited.insert(node_ids[i]);
        }
      };

      if (mem_L) {
        std::vector<unsigned> mem_tags(mem_L);
        std::vector<float>    mem_dists(mem_L);
        std::vector<T *>      res = std::vector<T *>();
        mem_index_->search_with_tags(query, mem_L, mem_L, mem_tags.data(),
                                     mem_dists.data(), nullptr, res);
        compute_and_add_to_retset(
            mem_tags.data(), std::min((unsigned) mem_L, (unsigned) l_search));
      } else {
        compute_and_add_to_retset(&best_medoid, 1);
      }

      std::sort(retset.begin(), retset.begin() + cur_list_size);

      unsigned num_ios = 0;
      unsigned k = 0;

      std::vector<float>
          io_push_queue_time_vec;  // 生成IO请求，提交到队列的时间 = ts_beign
      std::vector<float>
          io_submit_time_vec;  // 通过spdk提交io时间 = now - ts_begin
      std::vector<float>
          io_complete_time_vec;  // 通过spdk完成io的时间 = now - ts_begin
      std::vector<float>
          io_resume_time_vec;  // 对应的coro恢复执行的时间 = now - ts_begin
      std::vector<float> io_single_time_vec;

      // cleared every iteration
      std::vector<unsigned> frontier;
      frontier.reserve(2 * beam_width);
      std::vector<std::pair<unsigned, char *>> frontier_nhoods;
      frontier_nhoods.reserve(2 * beam_width);
      std::vector<std::pair<unsigned, char *>> prefetch_frontier_nhoods;
      prefetch_frontier_nhoods.reserve(2 * beam_width * affinity_size_);

      std::vector<AlignedRead> frontier_read_reqs;
      frontier_read_reqs.reserve(2 * beam_width);
      std::vector<std::pair<unsigned, std::pair<unsigned, unsigned *>>>
          cached_nhoods;
      cached_nhoods.reserve(2 * beam_width);

      std::vector<std::pair<unsigned, std::pair<unsigned, unsigned *>>>
          aff_cached_nhoods;
      aff_cached_nhoods.reserve(2 * beam_width);

      std::vector<unsigned> last_io_ids;
      last_io_ids.reserve(2 * beam_width);
      std::vector<char> last_pages(SECTOR_LEN * beam_width * 2);


      while (k < cur_list_size && num_ios < io_limit) {
        if (this->verbose_) {
          std::cout << cur_list_size << "," << k
                    << ", fullret_size: " << full_retset.size() << std::endl;
          std::cout << retset[k].print() << std::endl;
        }
        unsigned nk = cur_list_size;
        // clear iteration state
        frontier.clear();
        frontier_nhoods.clear();
        frontier_read_reqs.clear();
        prefetch_frontier_nhoods.clear();
        cached_nhoods.clear();
        sector_scratch_idx = 0;
        // find new beam
        _u32 marker = k;
        _u32 num_seen = 0;

        // Log the id of block be visited.
        std::vector<BlockVisited> block_visited_in_this_iter;
        // distribute cache and disk-read nodes
        while (marker < cur_list_size && frontier.size() < beam_width &&
               num_seen < beam_width) {
          const unsigned pid = id2page_[retset[marker].id];
          if (page_visited.find(pid) == page_visited.end() &&
              retset[marker].flag) {
            num_seen++;
            auto iter = nhood_cache.find(retset[marker].id);
            if (iter != nhood_cache.end()) {
              cached_nhoods.push_back(
                  std::make_pair(retset[marker].id, iter->second));
              if (this_coro_stats != nullptr) {
                this_coro_stats->n_cache_hits++;
              }
            } else {
              frontier.push_back(retset[marker].id);
              page_visited.insert(pid);
            }
            retset[marker].flag = false;
          }
          marker++;
        }

        // read nhoods of frontier ids
        if (!frontier.empty()) {
          std::vector<_u64> prefetch_block_ids;
          if (this_coro_stats != nullptr)
            this_coro_stats->n_hops++;
          for (_u64 i = 0; i < frontier.size(); i++) {
            auto                    id = frontier[i];
            _u64                    block_id = static_cast<_u64>(id2page_[id]);
            std::pair<_u32, char *> fnhood;
            fnhood.first = id;
            fnhood.second = sector_scratch + sector_scratch_idx * SECTOR_LEN;
            sector_scratch_idx++;
            frontier_nhoods.push_back(fnhood);
            // AlignedRead* tmp_ptr = new AlignedRead((static_cast<_u64>(id2page_[id] + 1)) * SECTOR_LEN, SECTOR_LEN,
            // fnhood.second, block_id, thread_id, coro_id);
            // frontier_read_reqs.emplace_back(*tmp_ptr);
            frontier_read_reqs.emplace_back((static_cast<_u64>(id2page_[id] + 1)) * SECTOR_LEN, SECTOR_LEN,
            fnhood.second, block_id+1, thread_id, coro_id);
            if (this_coro_stats != nullptr) {
              this_coro_stats->n_4k++;
              this_coro_stats->n_ios++;
              block_visited_in_this_iter.push_back(BlockVisited(
                  block_id+1, std::chrono::high_resolution_clock::now()));
            }
            num_ios++;
          }
          io_timer.reset();

          // TODO use coroutine to issue io
          if (verbose_) {
            size_t io_size = frontier_read_reqs.size();
            std::cout << "Begin" << std::endl;
            std::cout << io_size << std::endl;
          }

          if (this_coro_stats != nullptr) {
              this_coro_stats->executing_in_coro_us += (double) coro_timer.elapsed();
          }
          thread_stat->executing_in_coro_us += (double) thread_timer.elapsed();

          // int io_return_num = co_await diskann::IORegisterAwaiter<T>((PQFlashIndex<T>*)(this), frontier_read_reqs,thread_id,coro_id,thread_stat);
          // int io_return_num = co_await diskann::NewIORegisterAwaiter<T>((PQFlashIndex<T>*)(this), frontier_read_reqs,thread_id,coro_id,thread_stat);
          int io_return_num = co_await diskann::LibaioIORegisterAwaiter<T>((PQFlashIndex<T>*)(this), frontier_read_reqs,thread_id,coro_id,thread_stat);

          this->query_io_per_coro[thread_id][coro_id].resume();

          
          io_single_time_vec.push_back(io_timer.elapsed());
          io_push_queue_time_vec.push_back(this->query_io_per_coro[thread_id][coro_id].io_begin_time);
          io_submit_time_vec.push_back(this->query_io_per_coro[thread_id][coro_id].io_submit_time);
          io_complete_time_vec.push_back(this->query_io_per_coro[thread_id][coro_id].io_complete_time);
          io_resume_time_vec.push_back(this->query_io_per_coro[thread_id][coro_id].io_resume_time);


          this_coro_stats->n_io_returns += io_return_num;

          coro_timer.reset();
          thread_timer.reset();
          
          // std::cout << "After io."<< std::endl;
          if (this->count_visited_nodes) {
#pragma omp critical
            {
              auto &cnt = this->node_visit_counter[retset[marker].id].second;
              ++cnt;
            }
          }
          // reader->get_events(ctx, n_ops);
          if (verbose_) {
            std::cout << io_timer.elapsed() << std::endl;
            std::cout << "End" << std::endl;
          }

          if (this_coro_stats != nullptr) {
            this_coro_stats->io_us += (double) io_timer.elapsed();
            for (auto item : block_visited_in_this_iter) {
              this_coro_stats->block_visited_queue.push_back(BlockVisited(
                  item.block_id, std::chrono::high_resolution_clock::now()));
            }
            block_visited_in_this_iter.clear();
          }
        }
        cpu_timer.reset();

        // process cached nhoods
        for (auto &cached_nhood : cached_nhoods) {
          auto      id = cached_nhood.first;
          auto      global_cache_iter = coord_cache.find(cached_nhood.first);
          T        *node_fp_coords_copy = global_cache_iter->second;
          unsigned  nnr = cached_nhood.second.first;
          unsigned *cnhood = cached_nhood.second.second;
          std::vector<char> node_tmp(max_node_len);
          char*      node_buf = node_tmp.data();
          memcpy(node_buf, node_fp_coords_copy, disk_bytes_per_point);
          memcpy((node_buf + disk_bytes_per_point), &nnr, sizeof(unsigned));
          memcpy((node_buf + disk_bytes_per_point + sizeof(unsigned)), cnhood,
                 sizeof(unsigned) * nnr);
          compute_extact_dists_and_push(node_buf, id);
          compute_and_push_nbrs(node_buf, nk);
        }
        if (this_coro_stats != nullptr) {
          this_coro_stats->cpu_us += (double) cpu_timer.elapsed();
        }
        // TODO 记录这部分io时间和计算时间到底是谁等谁

        cpu_timer.reset();
        // compute only the desired vectors in the pages - one for each page
        // postpone remaining vectors to the next round
        for (auto &frontier_nhood : frontier_nhoods) {
          char    *sector_buf = frontier_nhood.second;
          unsigned pid = id2page_[frontier_nhood.first];
          memcpy(last_pages.data() + last_io_ids.size() * SECTOR_LEN, sector_buf, SECTOR_LEN);
          last_io_ids.emplace_back(frontier_nhood.first);

          for (unsigned j = 0; j < gp_layout_[pid].size(); ++j) {
            unsigned id = gp_layout_[pid][j];
            if (id == frontier_nhood.first) {
              char *node_buf = sector_buf + j * max_node_len;
              compute_extact_dists_and_push(node_buf, id);
              compute_and_push_nbrs(node_buf, nk);
            }
          }
        }
        for (size_t i = 0; i < last_io_ids.size(); ++i) {
          const unsigned last_io_id = last_io_ids[i];
          char    *sector_buf = last_pages.data() + i * SECTOR_LEN;
          const unsigned pid = id2page_[last_io_id];
          const unsigned p_size = gp_layout_[pid].size();
          // minus one for the vector that is computed previously
          unsigned vis_size = use_ratio * (p_size - 1);
          std::vector<std::pair<float, const char*>> vis_cand;
          vis_cand.reserve(p_size);

          // compute exact distances of the vectors within the page
          for (unsigned j = 0; j < p_size; ++j) {
            const unsigned id = gp_layout_[pid][j];
            if (id == last_io_id) continue;
            const char* node_buf = sector_buf + j * max_node_len;
            float dist = compute_extact_dists_and_push(node_buf, id);
            vis_cand.emplace_back(dist, node_buf);
          }
          if (vis_size && vis_size != p_size) {
            std::sort(vis_cand.begin(), vis_cand.end());
          }

          // compute PQ distances for neighbours of the vectors in the page
          for (unsigned j = 0; j < vis_size; ++j) {
            compute_and_push_nbrs(vis_cand[j].second, nk);
          }
        }
        last_io_ids.clear();

        if (this_coro_stats != nullptr) {
          this_coro_stats->cpu_us += (double) cpu_timer.elapsed();
        }

        // update best inserted position
        if (nk <= k)
          k = nk;  // k is the best position in retset updated in this round.
        else
          ++k;
      }

      // re-sort by distance
      std::sort(full_retset.begin(), full_retset.end(),
                [](const Neighbor &left, const Neighbor &right) {
                  return left.distance < right.distance;
                });
      // copy k_search values
      _u64 t = 0;
      for (_u64 i = 0; i < full_retset.size() && t < k_search; i++) {
        if (i > 0 && full_retset[i].id == full_retset[i - 1].id) {
          continue;
        }
        this_coro_result_ids_64[t] = full_retset[i].id;
        if (this_coro_result_distances != nullptr) {
          this_coro_result_distances[t] = full_retset[i].distance;
          if (metric == diskann::Metric::INNER_PRODUCT) {
            // flip the sign to convert min to max
            this_coro_result_distances[t] = (-this_coro_result_distances[t]);
            // rescale to revert back to original norms (cancelling the effect
            // of base and query pre-processing)
            if (max_base_norm != 0)
              this_coro_result_distances[t] *= (max_base_norm * query_norm);
          }
        }
        t++;
      }
      if (t < k_search) {
        diskann::cerr << "The number of unique ids is less than topk"
                      << std::endl;
        exit(1);
      }
      if (this_coro_stats != nullptr) {
        this_coro_stats->total_us = (double) query_timer.elapsed();
        this_coro_stats->executing_in_coro_us += (double) coro_timer.elapsed();

        this_coro_stats->mean_io_push_queue_time = get_mean_vec(io_push_queue_time_vec);
        this_coro_stats->mean_io_submit_time = get_mean_vec(io_submit_time_vec);
        this_coro_stats->mean_io_complete_time = get_mean_vec(io_complete_time_vec);
        this_coro_stats->mean_io_resume_time = get_mean_vec(io_resume_time_vec);
        this_coro_stats->mean_io_time = get_mean_vec(io_single_time_vec);
      }
    }  // end for
    co_return;
  }
  template<typename T>
  cppcoro::task<void> PQFlashIndex<T>::pure_io_query_coro(
      const T *query, const size_t _query_num, const _u64 beam_width,
      const _u32 io_limit, QueryStats *stats, int thread_id, int coro_id,
      BQANN::Countdown &countdown, ThreadStats *thread_stat) {
    QueryScratch<T> scratch = this->coro_data.pop();
    while (scratch.sector_scratch == nullptr) {
      this->coro_data.wait_for_push_notify();
      scratch = this->coro_data.pop();
    }
    Timer thread_timer;
    thread_timer.reset();

    for (size_t q_id = thread_id * max_ncoroutines + coro_id;;
         q_id = q_id + max_nthreads * max_ncoroutines) {
      if (q_id >= _query_num) {
        this->coro_data.push(scratch);
        this->coro_data.push_notify_all();
        if (verbose_) {
          std::cout << "Coro Exit." << thread_id << "," << coro_id << std::endl;
        }
        thread_stat->executing_in_coro_us += (double) thread_timer.elapsed();
        countdown.Decrement();
        co_return;
      }
      // std::cout << q_id<<std::endl;

      QueryStats *this_coro_stats = stats + q_id;
      auto        query_scratch = &(scratch);
      query_scratch->reset();

      char *sector_scratch = query_scratch->sector_scratch;
      _u64 &sector_scratch_idx = query_scratch->sector_idx;

      Timer query_timer, io_timer, cpu_timer, coro_timer;
      query_timer.reset();
      coro_timer.reset();

      unsigned num_ios = 0;

      std::vector<AlignedRead> frontier_read_reqs;
      frontier_read_reqs.reserve(2 * beam_width);

      while (num_ios < 30000) {
        frontier_read_reqs.clear();
        sector_scratch_idx = 0;

        for (size_t k = 0; k < beam_width; k++) {
          // int   block_id = generate_random(gp_layout_.size());
          int   block_id = 1000*thread_id + k*100;
          char *tmp_buf = sector_scratch + sector_scratch_idx * SECTOR_LEN;
          sector_scratch_idx++;
          frontier_read_reqs.emplace_back((block_id + 1) * SECTOR_LEN,
                                          SECTOR_LEN, tmp_buf, block_id,
                                          thread_id, coro_id);
          if (this_coro_stats != nullptr) {
            this_coro_stats->n_4k++;
            this_coro_stats->n_ios++;
          }
        }
        io_timer.reset();
        thread_stat->executing_in_coro_us += (double) thread_timer.elapsed();
        if (this_coro_stats != nullptr) {
          this_coro_stats->executing_in_coro_us +=
              (double) coro_timer.elapsed();
        }
        // int io_return_num = co_await diskann::IORegisterAwaiter<T>(
        //     (PQFlashIndex<T> *) (this), frontier_read_reqs, thread_id, coro_id,
        //     thread_stat);
        int io_return_num = co_await diskann::NewIORegisterAwaiter<T>((PQFlashIndex<T>*)(this), frontier_read_reqs,thread_id,coro_id,thread_stat);
        
        io_return_num++;
        num_ios += beam_width;

        coro_timer.reset();
        thread_timer.reset();
      }
      if (this_coro_stats != nullptr) {
        this_coro_stats->total_us = (double) query_timer.elapsed();
        this_coro_stats->executing_in_coro_us +=
              (double) coro_timer.elapsed();
      }
      
      thread_stat->executing_in_coro_us += (double) coro_timer.elapsed();
    }
    co_return;
  }

  template<typename T>
  cppcoro::task<void> PQFlashIndex<T>::scheduler_coro(int thread_id,BQANN::Countdown &countdown, ThreadStats* thread_stat)
  {
    Timer all_timer,cpu_timer,wait_timer;
    all_timer.reset();
    cpu_timer.reset();
    wait_timer.reset();
    int last_idx = 0;
    int expected = 1;
    while(true){
      if(countdown.IsZero()){
        if(verbose_)
        std::cout<<"scheduler_coro exit. thread:"<<thread_id<<std::endl;
        thread_stat->scheduler_cpu_us+=cpu_timer.elapsed();
        thread_stat->scheduler_total_us+=all_timer.elapsed();
        co_return;
      }
      // thread_stat->scheduler_cpu_us+=cpu_timer.elapsed();
      // int coro_id;

      // wait_timer.reset();
      // while(thread_complete_io_queue[thread_id]->empty()){
      //   thread_complete_io_queue[thread_id]->wait_for_push_notify();
      // }
      // thread_stat->scheduler_wait_us+=wait_timer.elapsed();
      // cpu_timer.reset();
      // coro_id = thread_complete_io_queue[thread_id]->pop();

      // thread_stat->scheduler_cpu_us+=cpu_timer.elapsed();
      // thread_stat->scheduler_total_us+=all_timer.elapsed();
      // handles_map[thread_id][coro_id].resume();   
      // cpu_timer.reset();
      // all_timer.reset();

      expected = libaio_cnt[thread_id][last_idx];
      if(this->atomic_mark[thread_id*max_ncoroutines+last_idx].compare_exchange_strong(expected,0)){
        // std::cout<<"try to awake, thr:"<<thread_id<<", coro:"<<last_idx<<std::endl;
        int coro_id = last_idx;
        last_idx = (last_idx+1)%max_ncoroutines;
        thread_stat->scheduler_total_us+=all_timer.elapsed();
        thread_stat->scheduler_cpu_us+=cpu_timer.elapsed();
        // std::cout<<"Before resume."<<std::endl;
        // float io_complete_time = query_io_per_coro[thread_id][coro_id].get_elapsed_time();
        handles_map[thread_id][coro_id].resume();
        // std::cout<<"After resume."<<std::endl;
        all_timer.reset();
        cpu_timer.reset();
      }else{
        last_idx = (last_idx+1)%max_ncoroutines;
      }

      
      // std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    co_return;  
  }
  template<typename T>
  void PQFlashIndex<T>::async_search(const T *query1, const _u64 k_search,
                                     const _u32 mem_L, const _u64 l_search,
                                     _u64 *indices, float *distances,
                                     const _u64 beam_width, const _u32 io_limit,
                                     const bool  use_reorder_data,
                                     const float use_ratio, QueryStats *stats) {
    // Get thread data
    ThreadData<T> data = this->thread_data.pop();
    while (data.scratch.sector_scratch == nullptr) {
      this->thread_data.wait_for_push_notify();
      data = this->thread_data.pop();
    }

    if (beam_width > MAX_N_SECTOR_READS)
      throw ANNException("Beamwidth can not be higher than MAX_N_SECTOR_READS",
                         -1, __FUNCSIG__, __FILE__, __LINE__);

    // copy query to thread specific aligned and allocated memory (for distance
    // calculations we need aligned data)
    float        query_norm = 0;
    const T     *query = data.scratch.aligned_query_T;
    const float *query_float = data.scratch.aligned_query_float;

    uint32_t query_dim = metric == diskann::Metric::INNER_PRODUCT
                             ? this->data_dim - 1
                             : this->data_dim;

    for (uint32_t i = 0; i < query_dim; i++) {
      data.scratch.aligned_query_float[i] = query1[i];
      data.scratch.aligned_query_T[i] = query1[i];
      query_norm += query1[i] * query1[i];
    }

    // if inner product, we also normalize the query and set the last coordinate
    // to 0 (this is the extra coordindate used to convert MIPS to L2 search)
    if (metric == diskann::Metric::INNER_PRODUCT) {
      query_norm = std::sqrt(query_norm);
      data.scratch.aligned_query_T[this->data_dim - 1] = 0;
      data.scratch.aligned_query_float[this->data_dim - 1] = 0;
      for (uint32_t i = 0; i < this->data_dim - 1; i++) {
        data.scratch.aligned_query_T[i] /= query_norm;
        data.scratch.aligned_query_float[i] /= query_norm;
      }
    }

		// boost::asio::io_context ioctx; 
    // IOContext &ctx = data.ctx;
    auto       query_scratch = &(data.scratch);

    // reset query
    query_scratch->reset();

    // pointers to buffers for data
    T *data_buf = query_scratch->coord_scratch;
    _mm_prefetch((char *) data_buf, _MM_HINT_T1);

    // sector scratch
    char *sector_scratch = query_scratch->sector_scratch;
    _u64 &sector_scratch_idx = query_scratch->sector_idx;

    // query <-> PQ chunk centers distances
    float *pq_dists = query_scratch->aligned_pqtable_dist_scratch;
    pq_table.populate_chunk_distances(query_float, pq_dists);

    // query <-> neighbor list
    float *dist_scratch = query_scratch->aligned_dist_scratch;
    _u8   *pq_coord_scratch = query_scratch->aligned_pq_coord_scratch;

    Timer query_timer, io_timer, cpu_timer;
    query_timer.reset();
    std::vector<Neighbor>     retset(l_search + 1);
    tsl::robin_set<_u64>     &visited = *(query_scratch->visited);
    tsl::robin_set<unsigned> &page_visited = *(query_scratch->page_visited);
    unsigned                  cur_list_size = 0;

    std::vector<Neighbor> full_retset;
    full_retset.reserve(4096);
    _u32                        best_medoid = 0;
    float                       best_dist = (std::numeric_limits<float>::max)();
    std::vector<SimpleNeighbor> medoid_dists;
    for (_u64 cur_m = 0; cur_m < num_medoids; cur_m++) {
      float cur_expanded_dist = dist_cmp_float->compare(
          query_float, centroid_data + aligned_dim * cur_m,
          (unsigned) aligned_dim);
      if (cur_expanded_dist < best_dist) {
        best_medoid = medoids[cur_m];
        best_dist = cur_expanded_dist;
      }
    }

    // lambda to batch compute query<-> node distances in PQ space
    auto compute_pq_dists = [this, pq_coord_scratch, pq_dists](
                                const unsigned *ids, const _u64 n_ids,
                                float *dists_out) {
      pq_flash_index_utils::aggregate_coords(ids, n_ids, this->data,
                                             this->n_chunks, pq_coord_scratch);
      pq_flash_index_utils::pq_dist_lookup(pq_coord_scratch, n_ids,
                                           this->n_chunks, pq_dists, dists_out);
    };

    // 将id的点强制push进入full_retset
    auto compute_extact_dists_and_push = [&](const char    *node_buf,
                                             const unsigned id) -> float {
      T *node_fp_coords_copy = data_buf;
      memcpy(node_fp_coords_copy, node_buf, disk_bytes_per_point);
      float cur_expanded_dist =
          dist_cmp->compare(query, node_fp_coords_copy, (unsigned) aligned_dim);
      full_retset.push_back(Neighbor(id, cur_expanded_dist, true));
      return cur_expanded_dist;
    };

    auto compute_and_push_nbrs = [&](const char *node_buf, unsigned &nk) {
      unsigned *node_nbrs = OFFSET_TO_NODE_NHOOD(node_buf);
      unsigned  nnbrs = *(node_nbrs++);
      unsigned  nbors_cand_size = 0;
      for (unsigned m = 0; m < nnbrs; ++m) {
        if (visited.find(node_nbrs[m]) == visited.end()) {
          node_nbrs[nbors_cand_size++] = node_nbrs[m];
          visited.insert(node_nbrs[m]);
        }
      }
      if (nbors_cand_size) {
        compute_pq_dists(node_nbrs, nbors_cand_size, dist_scratch);
        for (unsigned m = 0; m < nbors_cand_size; ++m) {
          const int   nbor_id = node_nbrs[m];
          const float nbor_dist = dist_scratch[m];
          if (stats != nullptr) {
            stats->n_cmps++;
          }
          if (nbor_dist >= retset[cur_list_size - 1].distance &&
              (cur_list_size == l_search))
            continue;
          Neighbor nn(nbor_id, nbor_dist, true);
          // Return position in sorted list where nn inserted
          auto r = InsertIntoPool(retset.data(), cur_list_size, nn);
          if (cur_list_size < l_search)
            ++cur_list_size;
          // nk logs the best position in the retset that was updated due to
          // neighbors of n.
          if (r < nk)
            nk = r;
        }
      }
    };

    // 这个是原本就有的结构
    // 计算 node_ids[] 若干个节点的pq距离并加入到retset中
    auto compute_and_add_to_retset = [&](const unsigned *node_ids,
                                         const _u64      n_ids) {
      compute_pq_dists(node_ids, n_ids, dist_scratch);
      for (_u64 i = 0; i < n_ids; ++i) {
        retset[cur_list_size].id = node_ids[i];
        retset[cur_list_size].distance = dist_scratch[i];
        retset[cur_list_size++].flag = true;
        visited.insert(node_ids[i]);
      }
    };

    if (mem_L) {
      std::vector<unsigned> mem_tags(mem_L);
      std::vector<float>    mem_dists(mem_L);
      std::vector<T *>      res = std::vector<T *>();
      mem_index_->search_with_tags(query, mem_L, mem_L, mem_tags.data(),
                                   mem_dists.data(), nullptr, res);
      compute_and_add_to_retset(
          mem_tags.data(), std::min((unsigned) mem_L, (unsigned) l_search));
    } else {
      compute_and_add_to_retset(&best_medoid, 1);
    }

    std::sort(retset.begin(), retset.begin() + cur_list_size);

    unsigned num_ios = 0;
    unsigned k = 0;

    // cleared every iteration
    std::vector<unsigned> frontier;
    frontier.reserve(2 * beam_width);
    std::vector<std::pair<unsigned, char *>> frontier_nhoods;
    frontier_nhoods.reserve(2 * beam_width);
    std::vector<std::pair<unsigned, char *>> prefetch_frontier_nhoods;
    prefetch_frontier_nhoods.reserve(2 * beam_width * affinity_size_);

    std::vector<AlignedRead> frontier_read_reqs;
    frontier_read_reqs.reserve(2 * beam_width);
    std::vector<std::pair<unsigned, std::pair<unsigned, unsigned *>>>
        cached_nhoods;
    cached_nhoods.reserve(2 * beam_width);

    std::vector<std::pair<unsigned, std::pair<unsigned, unsigned *>>>
        aff_cached_nhoods;
    aff_cached_nhoods.reserve(2 * beam_width);

    std::vector<unsigned> last_io_ids;
    last_io_ids.reserve(2 * beam_width);
    std::vector<char> last_pages(SECTOR_LEN * beam_width * 2);
    // int               n_ops = 0;
    
    BQANN::IOUring     ring(64);
    const BQANN::File index_file(this->disk_index_file.c_str(), BQANN::File::kRead, true);

    while (k < cur_list_size && num_ios < io_limit) {
      if (this->verbose_) {
        std::cout << cur_list_size << "," << k
                  << ", fullret_size: " << full_retset.size() << std::endl;
        std::cout << retset[k].print() << std::endl;
      }
      unsigned nk = cur_list_size;
      // clear iteration state
      frontier.clear();
      frontier_nhoods.clear();
      frontier_read_reqs.clear();
      prefetch_frontier_nhoods.clear();
      cached_nhoods.clear();
      sector_scratch_idx = 0;
      // find new beam
      _u32 marker = k;
      _u32 num_seen = 0;

      // Log the id of block be visited.
      std::vector<BlockVisited> block_visited_in_this_iter;
      // distribute cache and disk-read nodes
      while (marker < cur_list_size && frontier.size() < beam_width &&
             num_seen < beam_width) {
        const unsigned pid = id2page_[retset[marker].id];
        if (page_visited.find(pid) == page_visited.end() &&
            retset[marker].flag) {
          num_seen++;
          auto iter = nhood_cache.find(retset[marker].id);
          if (iter != nhood_cache.end()) {
            cached_nhoods.push_back(
                std::make_pair(retset[marker].id, iter->second));
            if (stats != nullptr) {
              stats->n_cache_hits++;
            }
          } else {
            frontier.push_back(retset[marker].id);
            page_visited.insert(pid);
          }
          retset[marker].flag = false;
        }
        marker++;
      }

      // read nhoods of frontier ids
      if (!frontier.empty()) {
        // std::vector<_u64> prefetch_block_ids;
        if (stats != nullptr)
          stats->n_hops++;
        for (_u64 i = 0; i < frontier.size(); i++) {
          auto                    id = frontier[i];
          _u64                    block_id = static_cast<_u64>(id2page_[id]);
          std::pair<_u32, char *> fnhood;
          fnhood.first = id;
          fnhood.second = sector_scratch + sector_scratch_idx * SECTOR_LEN;
          sector_scratch_idx++;
          frontier_nhoods.push_back(fnhood);
          frontier_read_reqs.emplace_back(
              (static_cast<_u64>(id2page_[id] + 1)) * SECTOR_LEN, SECTOR_LEN,
              fnhood.second);
          // if (use_affinity_) {
          //   for (size_t idx = 0; idx < affinity_size_; idx++) {
          //     prefetch_block_ids.push_back(
          //         affinity_prefetch_dict_[block_id][idx]);
          //   }
          // }
          if (stats != nullptr) {
            stats->n_4k++;
            stats->n_ios++;
            block_visited_in_this_iter.push_back(BlockVisited(
                block_id, std::chrono::high_resolution_clock::now()));
          }
          num_ios++;
        }
        // if (use_affinity_) {
        //   for (_u64 i = 0; i < prefetch_block_ids.size(); i++) {
        //     unsigned pid = prefetch_block_ids[i];
        //     char    *sector_buf_ptr =
        //         sector_scratch + sector_scratch_idx * SECTOR_LEN;
        //     unsigned id = gp_layout_[pid][0];  // 只把每个块的第一个点放入
        //     std::pair<_u32, char *> fnhood;
        //     fnhood.first = id;
        //     fnhood.second = sector_buf_ptr;
        //     prefetch_frontier_nhoods.push_back(fnhood);
        //     frontier_read_reqs.emplace_back((pid + 1) * SECTOR_LEN, SECTOR_LEN,
        //                                     sector_buf_ptr);
        //     sector_scratch_idx++;
        //     if (stats != nullptr) {
        //       stats->n_4k++;
        //       stats->n_ios++;

        //     }
        //     num_ios++;
        //   }
        // }

        io_timer.reset();
        size_t coro_size = 1;
        // size_t coro_size = frontier_read_reqs.size();
        if (this->verbose_) {
          std::cout << "Begin" << std::endl;
          std::cout << coro_size << std::endl;
        }
        // std::cout<<cur_list_size<<","<<k<<std::endl;
        // std::cout<<retset[k].print()<<std::endl;
        
        BQANN::Countdown   countdown(coro_size);
        

        std::vector<cppcoro::task<void>> tasks;
        tasks.reserve(coro_size+1);
        
        // for (auto &req : frontier_read_reqs) {
        //   // co_await
        //   // file.async_read_some_at(req.offset,req.buff,as::use_awaitable);
        //   // co_await index_file.AsyncReadBlock(ring,)
        //   // printFirst128Chars(req);
        //   tasks.emplace_back(AsyncProcessPages(req.offset, (char*)(req.buf), req.len,
        //                                        index_file, ring, countdown));
        // }
        for(size_t k = 0;k<coro_size;k++){
            tasks.emplace_back(AsyncBatchRead(frontier_read_reqs,index_file,ring,countdown,k));
        }
        tasks.emplace_back(DrainRing(ring, countdown,coro_size));
        // std::cout<<tasks.size()<<std::endl;
        cppcoro::sync_wait(cppcoro::when_all_ready(std::move(tasks)));
        if (this->verbose_) {
          std::cout << io_timer.elapsed() << std::endl;
          std::cout << "End" << std::endl;
        }

        if (stats != nullptr) {
          stats->io_us += (double) io_timer.elapsed();
          for (auto item : block_visited_in_this_iter) {
            stats->block_visited_queue.push_back(BlockVisited(
                item.block_id, std::chrono::high_resolution_clock::now()));
          }
          block_visited_in_this_iter.clear();
        }
        if (this->count_visited_nodes) {
#pragma omp critical
          {
            auto &cnt = this->node_visit_counter[retset[marker].id].second;
            ++cnt;
          }
        }
      }
      cpu_timer.reset();

      // process cached nhoods
      for (auto &cached_nhood : cached_nhoods) {
        auto      id = cached_nhood.first;
        auto      global_cache_iter = coord_cache.find(cached_nhood.first);
        T        *node_fp_coords_copy = global_cache_iter->second;
        unsigned  nnr = cached_nhood.second.first;
        unsigned *cnhood = cached_nhood.second.second;
        char      node_buf[max_node_len];
        memcpy(node_buf, node_fp_coords_copy, disk_bytes_per_point);
        memcpy((node_buf + disk_bytes_per_point), &nnr, sizeof(unsigned));
        memcpy((node_buf + disk_bytes_per_point + sizeof(unsigned)), cnhood,
               sizeof(unsigned) * nnr);
        compute_extact_dists_and_push(node_buf, id);
        compute_and_push_nbrs(node_buf, nk);
      }
      if (stats != nullptr) {
        stats->cpu_us += (double) cpu_timer.elapsed();
      }

      cpu_timer.reset();
      // compute only the desired vectors in the pages - one for each page
      // postpone remaining vectors to the next round
      for (auto &frontier_nhood : frontier_nhoods) {
        char *sector_buf = frontier_nhood.second;
        unsigned pid = id2page_[frontier_nhood.first];
        const unsigned p_size = gp_layout_[pid].size();

        if(true){
          unsigned vis_size = use_ratio * (p_size);
          std::vector<std::pair<float, const char*>> vis_cand;
          vis_cand.reserve(p_size);
          // compute exact distances of the vectors within the page
          for (unsigned j = 0; j < p_size; ++j) {
            const unsigned id = gp_layout_[pid][j];
            const char* node_buf = sector_buf + j * max_node_len;
            float dist = compute_extact_dists_and_push(node_buf, id);
            vis_cand.emplace_back(dist, node_buf);
          }
          if (vis_size && vis_size != p_size) {
            std::sort(vis_cand.begin(), vis_cand.end());
          }
          // compute PQ distances for neighbours of the vectors in the page
          for (unsigned j = 0; j < vis_size; ++j) {
            compute_and_push_nbrs(vis_cand[j].second, nk);
          }
        }else{
          // compute exact distances of the vectors within the page
          for (unsigned j = 0; j < p_size; ++j) {
            const unsigned id = gp_layout_[pid][j];
            if(id == frontier_nhood.first){
              const char* node_buf = sector_buf + j * max_node_len;
              compute_extact_dists_and_push(node_buf, id);
              compute_and_push_nbrs(node_buf, nk);
            }
            
          }
        }
        
      }
      last_io_ids.clear();
      if (use_affinity_) {
        // unsigned * tmp_nhood_cache = query_scratch->affinity_nhood_cache_buf;
        // T * tmp_coord_cache = query_scratch->affinity_coord_cache_buf;
        _u64 node_idx = 0;
        for (auto &nhood : prefetch_frontier_nhoods) {
          char    *node_buf = nullptr;
          char    *sector_buf = nhood.second;
          unsigned pid = id2page_[nhood.first];
          memcpy(last_pages.data() + last_io_ids.size() * SECTOR_LEN,
                 sector_buf, SECTOR_LEN);
          last_io_ids.emplace_back(nhood.first);
          for (unsigned j = 0; j < gp_layout_[pid].size(); ++j) {
            unsigned id = gp_layout_[pid][j];
            if (id == nhood.first) {
              node_buf = sector_buf + j * max_node_len;
              // compute_extact_dists_and_push(node_buf, id);
              compute_and_push_nbrs(node_buf, nk);
            }
          }
        }
        if (stats != nullptr) {
          stats->n_affinity_cache += node_idx;
        }
      }

      if (stats != nullptr) {
        stats->cpu_us += (double) cpu_timer.elapsed();
      }

      // update best inserted position
      if (nk <= k)
        k = nk;  // k is the best position in retset updated in this round.
      else
        ++k;
    }

    // re-sort by distance
    std::sort(full_retset.begin(), full_retset.end(),
              [](const Neighbor &left, const Neighbor &right) {
                return left.distance < right.distance;
              });

    // copy k_search values
    _u64 t = 0;
    for (_u64 i = 0; i < full_retset.size() && t < k_search; i++) {
      if (i > 0 && full_retset[i].id == full_retset[i - 1].id) {
        continue;
      }
      indices[t] = full_retset[i].id;
      if (distances != nullptr) {
        distances[t] = full_retset[i].distance;
        if (metric == diskann::Metric::INNER_PRODUCT) {
          // flip the sign to convert min to max
          distances[t] = (-distances[t]);
          // rescale to revert back to original norms (cancelling the effect of
          // base and query pre-processing)
          if (max_base_norm != 0)
            distances[t] *= (max_base_norm * query_norm);
        }
      }
      t++;
    }

    if (t < k_search) {
      diskann::cerr << "The number of unique ids is less than topk"
                    << std::endl;
      exit(1);
    }

    this->thread_data.push(data);
    this->thread_data.push_notify_all();

    if (stats != nullptr) {
      stats->total_us = (double) query_timer.elapsed();
    }
  }
  template<typename T>
  void PQFlashIndex<T>::pure_io_search(
    const T *query, const size_t query_num,const _u64 beam_width,
    const _u32  io_limit, QueryStats *stats){
      // TODO 设置各个线程的初始内存分配

    auto thread_stats = new diskann::ThreadStats[this->max_nthreads+MAX_IO_RING_NUM];
    std::cout<<"Pure IO"<<std::endl;
    // std::unique_lock<std::mutex> lk(mtx);
    this->executing_thread_num = this->max_nthreads;
    // lk.unlock();
    std::vector<std::thread> all_threads;
    size_t n_io_thread_num = 1;
    size_t issue_io_thread_num = 8;

    // add worker
    for (_u64 i = 0; i < this->max_nthreads; i++) {
      std::string thread_name = "WORKER" + std::to_string(i);
      if (verbose_)
        std::cout << thread_name << std::endl;
      std::thread t(&PQFlashIndex<T>::pure_io_worker_thread, this, query, query_num,
                    beam_width,
                    io_limit, stats, i,
                    thread_stats + i);

      pthread_t pthread_handle =
          *reinterpret_cast<pthread_t *>(t.native_handle());

      pthread_setname_np(pthread_handle, thread_name.c_str());
      all_threads.push_back(std::move(t));
    }
    // add issue io thread
    for (_u64 i = this->max_nthreads ; i < this->max_nthreads + issue_io_thread_num; i++) {
      std::string thread_name = "ISSUE-IO" + std::to_string(i);
      if (verbose_)
        std::cout << thread_name << std::endl;
      std::thread t(&PQFlashIndex<T>::issue_io_thread, this,i,thread_stats + i, i-this->max_nthreads,issue_io_thread_num);

      pthread_t pthread_handle =
          *reinterpret_cast<pthread_t *>(t.native_handle());

      pthread_setname_np(pthread_handle, thread_name.c_str());
      all_threads.push_back(std::move(t));
    }
    // add io
    for (_u64 i = this->max_nthreads+issue_io_thread_num ; i < this->max_nthreads + issue_io_thread_num+ n_io_thread_num; i++) {
      std::string thread_name = "REAP-IO" + std::to_string(i);
      if (verbose_)
        std::cout << thread_name << std::endl;
      std::thread t(&PQFlashIndex<T>::reap_io_thread, this,i,thread_stats + i);

      pthread_t pthread_handle =
          *reinterpret_cast<pthread_t *>(t.native_handle());

      pthread_setname_np(pthread_handle, thread_name.c_str());
      all_threads.push_back(std::move(t));
    }
    for (auto& t : all_threads) {
        t.join();
    }
    for (_u64 i = 0; i < this->max_nthreads; i++) {
      ThreadStats *thread_stat = thread_stats + i;
      std::cout << "[WORKER Thread #" << i << "] ";
      thread_stat_print(thread_stat, true);
    }
    for (_u64 i = this->max_nthreads;
         i < this->max_nthreads + issue_io_thread_num; i++) {
      ThreadStats *tmp = thread_stats + i;
      std::cout << "[ISSUE IO Thread #" << i << "] "
                << ", cpu time:" << tmp->cpu_us / tmp->total_us
                << ", io time:" << tmp->io_us / tmp->total_us
                << ", total time:" << tmp->total_us
                << ", iops: " << tmp->n_ios / (tmp->total_us / 1000 * 1000)
                << std::endl;
    }
    for (_u64 i = this->max_nthreads + issue_io_thread_num;
         i < this->max_nthreads + issue_io_thread_num + n_io_thread_num; i++) {
      ThreadStats *tmp = thread_stats + i;
      std::cout << "[Reap IO Thread #" << i << "] "
                << ", cpu time:" << tmp->cpu_us / tmp->total_us
                << ", io time:" << tmp->io_us / tmp->total_us
                << ", total time:" << tmp->total_us
                << ", iops: " << tmp->n_ios / (tmp->total_us / 1000 * 1000)
                << std::endl;
    }
  }

  template<typename T>
  void PQFlashIndex<T>::pure_libaio_search(
    const T *query, const size_t query_num,const _u64 beam_width,
    const _u32  io_limit, QueryStats *stats){
      // TODO 设置各个线程的初始内存分配

    auto thread_stats = new diskann::ThreadStats[this->max_nthreads];
    std::cout<<"Pure IO using libaio"<<std::endl;
    std::vector<std::thread> all_threads;
    

    // add worker
    for (_u64 i = 0; i < this->max_nthreads; i++) {
      std::string thread_name = "WORKER" + std::to_string(i);
      if (verbose_)
        std::cout << thread_name << std::endl;
      std::thread t(&PQFlashIndex<T>::pure_io_libaio_worker_thread, this, query, query_num,
                    beam_width,
                    io_limit, stats, i,
                    thread_stats + i);

      pthread_t pthread_handle =
          *reinterpret_cast<pthread_t *>(t.native_handle());

      pthread_setname_np(pthread_handle, thread_name.c_str());
      all_threads.push_back(std::move(t));
    }
    for (auto& t : all_threads) {
        t.join();
    }
    for (_u64 i = 0; i < this->max_nthreads; i++) {
      ThreadStats *thread_stat = thread_stats + i;
    std::cout<<  "[WORKER Thread #" << i << "] ";
    std::cout <<"cpu:"<< (thread_stat->cpu_us) / thread_stat->total_us
              << ", io: " << thread_stat->io_us / thread_stat->total_us
              << std::endl;
    }
  }

  template<typename T>
  void PQFlashIndex<T>::bqann_search(const T *query, const size_t query_num,
                                     const _u64 k_search, const _u32 mem_L,
                                     const _u64 l_search, _u64 *indices,
                                     float *distances, const _u64 beam_width,
                                     const _u32  io_limit,
                                     const bool  use_reorder_data,
                                     const float use_ratio, QueryStats *stats) {

    // TODO 设置各个线程的初始内存分配

    auto thread_stats = new diskann::ThreadStats[this->max_nthreads+MAX_IO_RING_NUM];

    // std::unique_lock<std::mutex> lk(mtx);
    this->executing_thread_num = this->max_nthreads;
    // lk.unlock();
    std::vector<std::thread> all_threads;
    // size_t n_io_thread_num = 1;
    size_t issue_io_thread_num = this->io_nthreads;
    size_t reap_io_thread_num = 0;


    LOG(INFO)<<"BQANN search start. worker: "<< this->max_nthreads << " io: "<< issue_io_thread_num;

    // add worker
    for (_u64 i = 0; i < this->max_nthreads; i++) {
      std::string thread_name = "WORKER" + std::to_string(i);
      if (verbose_)
        std::cout << thread_name << std::endl;
      std::thread t(&PQFlashIndex<T>::worker_thread, this, query, query_num,
                    k_search, mem_L, l_search, indices, distances, beam_width,
                    io_limit, use_reorder_data, use_ratio, stats, i,
                    thread_stats + i);

      pthread_t pthread_handle =
          *reinterpret_cast<pthread_t *>(t.native_handle());

      pthread_setname_np(pthread_handle, thread_name.c_str());
      all_threads.push_back(std::move(t));
    }
    // add io
    for (_u64 i = this->max_nthreads ; i < this->max_nthreads + issue_io_thread_num; i++) {
      std::string thread_name = "LIBAIO-ISSUE-IO" + std::to_string(i);
      if (verbose_)
        std::cout << thread_name << std::endl;
      std::thread t(&PQFlashIndex<T>::spdk_issue_io_thread, this,i,thread_stats + i, i-this->max_nthreads,issue_io_thread_num);

      pthread_t pthread_handle =
          *reinterpret_cast<pthread_t *>(t.native_handle());

      pthread_setname_np(pthread_handle, thread_name.c_str());
      all_threads.push_back(std::move(t));
    }

    // add io
    for (_u64 i = this->max_nthreads+issue_io_thread_num ; i < this->max_nthreads + issue_io_thread_num+ reap_io_thread_num; i++) {
      std::string thread_name = "LIBAIO-REAP-IO" + std::to_string(i);
      if (verbose_)
        std::cout << thread_name << std::endl;
      // std::thread t(&PQFlashIndex<T>::reap_io_thread, this,i,thread_stats + i);
      std::thread t(&PQFlashIndex<T>::libaio_reap_io_thread, this,i,thread_stats + i, i-this->max_nthreads-issue_io_thread_num,reap_io_thread_num);
      pthread_t pthread_handle =
          *reinterpret_cast<pthread_t *>(t.native_handle());

      pthread_setname_np(pthread_handle, thread_name.c_str());
      all_threads.push_back(std::move(t));
    }
    for (auto& t : all_threads) {
        t.join();
    }
    for (_u64 i = 0; i < this->max_nthreads; i++) {
      ThreadStats *thread_stat = thread_stats + i;
      std::cout << "[WORKER Thread #" << i << "] ";
      thread_stat_print(thread_stat, true);
    }
    for (_u64 i = this->max_nthreads;
         i < this->max_nthreads + issue_io_thread_num; i++) {
      ThreadStats *tmp = thread_stats + i;
      std::cout << "[ISSUE IO Thread #" << i << "] "
                << ", cpu time:" << tmp->cpu_us / tmp->total_us
                << ", io time:" << tmp->io_us / tmp->total_us
                << ", submit time: "<< tmp->io_submit_us / tmp->total_us
                << ", reap time: "<< tmp->io_reap_us / tmp->total_us
                << ", distinct: "<<tmp->compute_us / tmp->total_us
                << ", iops: " << tmp->n_ios / (tmp->total_us / 1000 * 1000)
                << std::endl;
    }
    for (_u64 i = this->max_nthreads + issue_io_thread_num;
         i < this->max_nthreads + issue_io_thread_num + reap_io_thread_num; i++) {
      ThreadStats *tmp = thread_stats + i;
      std::cout << "[Reap IO Thread #" << i << "] "
                << ", cpu time:" << tmp->cpu_us / tmp->total_us
                << ", io time:" << tmp->io_us / tmp->total_us
                << ", total time:" << tmp->total_us
                << ", iops: " << tmp->n_ios / (tmp->total_us / 1000 * 1000)
                << std::endl;
    }
  }

  template<typename T>
  void PQFlashIndex<T>::worker_thread(
      const T *query, const size_t query_num, const _u64 k_search,
      const _u32 mem_L, const _u64 l_search, _u64 *indices, float *distances,
      const _u64 beam_width, const _u32 io_limit, const bool use_reorder_data,
      const float use_ratio, QueryStats *stats, int thread_id, ThreadStats* thread_stat) {
    // 绑定线程核心
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(thread_id, &mask);
    pthread_t current_thread = pthread_self();
    // 将当前线程绑定到指定的核心
    if (pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &mask) != 0) {
      std::cerr << "Error binding thread to core " << thread_id << std::endl;
      exit(1);
    }

    Timer all_timer;
    all_timer.reset();
    if (verbose_)
      std::cout << "[Worker Thread]Enter thread." << std::endl;
    size_t coro_size = max_ncoroutines;

    BQANN::Countdown countdown(coro_size);

    std::vector<cppcoro::task<void>> tasks;
    tasks.reserve(coro_size + 1);
    for (size_t k = 0; k < coro_size; k++) {
      // std::cout << "query_coro[" <<k<<"]"<< std::endl;
      tasks.emplace_back(
           query_coro(query,query_num,k_search,mem_L,l_search,indices, distances,beam_width,io_limit, use_reorder_data,use_ratio,stats,thread_id,k, countdown,thread_stat));
    }
    tasks.emplace_back(scheduler_coro(thread_id,countdown,thread_stat));
    cppcoro::sync_wait(cppcoro::when_all_ready(std::move(tasks)));


    std::unique_lock<std::mutex> lk(mtx);
    this->executing_thread_num--;
    lk.unlock();
    thread_stat->total_us+=all_timer.elapsed();

    // if(verbose_)
    std::cout << "[Worker Thread]Exit."<<thread_id<< std::endl;
    return;
  }
template<typename T>
  void PQFlashIndex<T>::pure_io_worker_thread(
      const T *query, const size_t query_num,
      const _u64 beam_width, const _u32 io_limit, QueryStats *stats, int thread_id, ThreadStats* thread_stat) {
    // 绑定线程核心
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(thread_id, &mask);
    if (sched_setaffinity(0, sizeof(mask), &mask) == -1) {
      std::cout << "Could not set CPU affinity" << std::endl;
    }

    Timer all_timer, io_timer, cpu_timer;
    all_timer.reset();
    if (verbose_)
      std::cout << "[Worker Thread]Enter thread." << std::endl;
    size_t coro_size = max_ncoroutines;

    BQANN::Countdown countdown(coro_size);

    std::vector<cppcoro::task<void>> tasks;
    tasks.reserve(coro_size + 1);
    for (size_t k = 0; k < coro_size; k++) {
      // std::cout << "query_coro[" <<k<<"]"<< std::endl;
      tasks.emplace_back(
           pure_io_query_coro(query,query_num,beam_width,io_limit,stats,thread_id,k, countdown,thread_stat));
    }
    tasks.emplace_back(scheduler_coro(thread_id,countdown,thread_stat));
    cppcoro::sync_wait(cppcoro::when_all_ready(std::move(tasks)));


    std::unique_lock<std::mutex> lk(mtx);
    this->executing_thread_num--;
    lk.unlock();
    thread_stat->total_us+=all_timer.elapsed();

    // if(verbose_)
    std::cout << "[Worker Thread]Exit."<<thread_id<< std::endl;
    return;
  }
template<typename T>
  void PQFlashIndex<T>::pure_io_libaio_worker_thread(
      const T *query, const size_t query_num,
      const _u64 beam_width, const _u32 io_limit, QueryStats *stats, int thread_id, ThreadStats* thread_stat) {
    // 绑定线程核心
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(thread_id, &mask);
    if (sched_setaffinity(0, sizeof(mask), &mask) == -1) {
      std::cout << "Could not set CPU affinity" << std::endl;
    }

    ThreadData<T> data = this->thread_data.pop();
    while (data.scratch.sector_scratch == nullptr) {
      this->thread_data.wait_for_push_notify();
      data = this->thread_data.pop();
    }

    // for thread-granularity
    Timer thread_timer;
    thread_timer.reset();

    IOContext &ctx = data.ctx;
    auto       query_scratch = &(data.scratch);

    // sector scratch
    char *sector_scratch = query_scratch->sector_scratch;
    _u64 &sector_scratch_idx = query_scratch->sector_idx;

    
    for (size_t q_id = thread_id;; q_id = q_id + max_nthreads) {
      if (q_id >= query_num) {
        break;
      }
      Timer query_timer, io_timer, cpu_timer;
      query_timer.reset();
      cpu_timer.reset();

      QueryStats *this_query_stats = stats + q_id;
      query_scratch->reset();

      unsigned num_ios = 0;

      std::vector<AlignedRead> frontier_read_reqs;
      frontier_read_reqs.reserve(2 * beam_width);
      while (num_ios < 50000) {
        frontier_read_reqs.clear();
        sector_scratch_idx = 0;

        for (size_t k = 0; k < beam_width; k++) {
          // int   block_id = generate_random(gp_layout_.size());
          int   block_id = 1000*thread_id + k*100;
          char *tmp_buf = sector_scratch + sector_scratch_idx * SECTOR_LEN;
          sector_scratch_idx++;
          frontier_read_reqs.emplace_back((block_id + 1) * SECTOR_LEN,
                                          SECTOR_LEN, tmp_buf, block_id);
          if (this_query_stats != nullptr) {
            this_query_stats->n_4k++;
            this_query_stats->n_ios++;
          }
        }

        if (this_query_stats != nullptr) {
          this_query_stats->cpu_us += (double) cpu_timer.elapsed();
        }

        io_timer.reset();
        int n_ops = reader->submit_reqs(frontier_read_reqs, ctx);
        reader->get_events(ctx, n_ops);
        // reader->read(frontier_read_reqs,ctx);
        if (this_query_stats != nullptr) {
          this_query_stats->io_us += (double) io_timer.elapsed();
        }
        num_ios += beam_width;

        cpu_timer.reset();
      }
      this_query_stats->total_us += (double)query_timer.elapsed();
      thread_stat->cpu_us += this_query_stats->cpu_us;
      thread_stat->io_us += this_query_stats->io_us;
    }

    this->thread_data.push(data);
    this->thread_data.push_notify_all();
    thread_stat->total_us+=thread_timer.elapsed();
    // if(verbose_)
    std::cout << "[Worker Thread]Exit."<<thread_id<< std::endl;
    return;
  }

  template<typename T>
  void PQFlashIndex<T>::io_thread(int io_thread_id,ThreadStats* thread_stat) {
    // 绑定线程核心 到最大工作线程数加1的位置
    // std::this_thread::sleep_for(std::chrono::milliseconds(10));
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(io_thread_id, &mask);
    if (sched_setaffinity(0, sizeof(mask), &mask) == -1) {
      std::cout << "Could not set CPU affinity" << std::endl;
    }
    Timer all_timer, cpu_timer, io_timer;
    all_timer.reset();
    cpu_timer.reset();
    io_timer.reset();
    if (verbose_)
      std::cout << "[IO Thread]Enter thread." << std::endl;
    constexpr size_t kBatchSize = 1024;

    // int tmp_mod2 = io_thread_id %2;
    // int ring_id = io_thread_id-max_nthreads;
    while(true){
        _u64 executing_thread_num_now;
        // std::unique_lock<std::mutex> lk(mtx);
        executing_thread_num_now = this->executing_thread_num;
        // lk.unlock();
        if( (executing_thread_num_now == 0)){
          break;
        }

        // TODO 单个query轮询所有的ring
        for(size_t ring_idx = 0; ring_idx<max_nthreads;ring_idx++){
          std::array<io_uring_cqe *, kBatchSize>              cqes;
          // if(ring_idx %2 != tmp_mod2){
          //   continue;
          // }
          // std::cout << "[IO Thread]wait for io."<<std::endl;
          // collect up to kBatchSize handles
          // this->ring_mutex_lock();
          // unsigned num_returned =
          //     io_uring_peek_batch_cqe(this->get_iouring(ring_id,0), cqes.data(), kBatchSize);
          io_timer.reset();
          unsigned num_returned =
              io_uring_peek_batch_cqe(this->get_iouring(ring_idx,0), cqes.data(), kBatchSize);
          thread_stat->io_us += io_timer.elapsed();
          thread_stat->n_ios += num_returned;
          // std::cout << "[IO Thread]Get io#:"<<num_returned<<std::endl;
          if(num_returned== kBatchSize){
            std::cout << "[IO Thread] IO full power."<< std::endl;
          }
          cpu_timer.reset();
          for (unsigned i = 0; i < num_returned; i++) {
            auto *coro_io_issue_aligned_read_tmp =
                reinterpret_cast<AlignedRead *>(io_uring_cqe_get_data(cqes[i]));
            // awaiter->SetResult(cqes[i]->res);
            io_uring_cqe_seen(this->get_iouring(ring_idx,0), cqes[i]);
            if(cqes[i]->res != 4096){
              std::cout<<"Return num is not 4096. "<<cqes[i]->res<<"<<<"<<std::endl;
            }
            int thread_id = coro_io_issue_aligned_read_tmp->thread_id;
            int coro_id = coro_io_issue_aligned_read_tmp->coro_id;
            // int idx = CORO_FINAL_NO(thread_id,coro_id);
            // std::cout << "[IO Thread]Thread id:"<<thread_id<<", coro id:"<<coro_id<< std::endl;
            // handles_map[thread_id][coro_id] = coro_io_issue_data_tmp->handle;
            // coro_io_queue_mutex.lock();
            this->n_io_completed[thread_id][coro_id]++;
            // if(this->io_state[thread_id][coro_id] == IORequestState::WaitingForIO&&this->n_io_completed[thread_id][coro_id] == this->n_io_executing[thread_id][coro_id]){
            if(this->n_io_completed[thread_id][coro_id] == this->n_io_executing[thread_id][coro_id]){
              this->n_io_executing[thread_id][coro_id] = 0;
              this->n_io_completed[thread_id][coro_id] = 0;
              // this->io_state[thread_id][coro_id] = IORequestState::IOCompleted;
              // thread_complete_io_queue[thread_id]->push(coro_id);
              atomic_mark[thread_id*max_ncoroutines+coro_id]++;
            }
            // coro_io_queue_mutex.unlock();
            // std::cout << "[IO Thread]Execute completed."<<this->n_io_completed[CORO_FINAL_NO(thread_id,coro_id)]<<","<<this->n_io_executing[CORO_FINAL_NO(thread_id,coro_id)]<< std::endl;
          }
          thread_stat->cpu_us += cpu_timer.elapsed();
          // this->ring_mutex_unlock();
          // std::this_thread::sleep_for(std::chrono::milliseconds(1));
          // std::this_thread::sleep_for(std::chrono::nanoseconds(100));
        }
        
    }
    thread_stat->total_us += all_timer.elapsed();
    if (verbose_)
      std::cout << "[IO Thread]Exit." << std::endl;
    return;
  }

  template<typename T>
  void PQFlashIndex<T>::reap_io_thread(int          io_thread_id,
                                       ThreadStats *thread_stat) {
    // 绑定线程核心 到最大工作线程数加1的位置
    // std::this_thread::sleep_for(std::chrono::milliseconds(10));
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(io_thread_id, &mask);
    pthread_t current_thread = pthread_self();
    // 将当前线程绑定到指定的核心
    if (pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &mask) != 0) {
      std::cerr << "Error binding thread to core " << io_thread_id << std::endl;
      exit(1);
    }
    Timer all_timer, cpu_timer, io_timer;
    all_timer.reset();
    cpu_timer.reset();
    if (DEBUG_LOG) {
      std::cout << "[Reap IO Thread]Enter thread." << std::endl;
    }
    constexpr size_t kBatchSize = 1024;

    while (true) {
      _u64 executing_thread_num_now;
      // std::unique_lock<std::mutex> lk(mtx);
      executing_thread_num_now = this->executing_thread_num;
      // lk.unlock();
      if ((executing_thread_num_now == 0)) {
        break;
      }
      for (size_t thread_idx = 0; thread_idx < max_nthreads; thread_idx++) {
        io_uring *ring_ptr = this->get_iouring(thread_idx,0);
        std::array<io_uring_cqe *, kBatchSize> cqes;
        io_timer.reset();
        unsigned num_returned =
            io_uring_peek_batch_cqe(ring_ptr, cqes.data(), kBatchSize);
        thread_stat->io_us += io_timer.elapsed();
        thread_stat->n_ios += num_returned;
        if (DEBUG_LOG) {
          if (num_returned > 0)
            std::cout << "[Reap IO Thread]Get io#:" << num_returned
                      << std::endl;
          if (num_returned == kBatchSize) {
            std::cout << "[Reap IO Thread] IO full power." << std::endl;
          }
        }
        cpu_timer.reset();
        for (unsigned i = 0; i < num_returned; i++) {
          auto *coro_io_issue_aligned_read_tmp =
              reinterpret_cast<AlignedRead *>(io_uring_cqe_get_data(cqes[i]));
          // coro_io_issue_aligned_read_tmp->print();
          // awaiter->SetResult(cqes[i]->res);
          if (DEBUG_LOG) {
            std::cout << "[Reap IO Thread]";
            std::cout << coro_io_issue_aligned_read_tmp << ",";
            coro_io_issue_aligned_read_tmp->print();
          }
          io_uring_cqe_seen(ring_ptr, cqes[i]);
          if (cqes[i]->res != 4096) {
            std::cout << "Return num is not 4096." <<cqes[i]->res<<"<<<"<<std::endl;
          }

          // double time = std::chrono::duration_cast<std::chrono::microseconds>(
          //                   std::chrono::high_resolution_clock::now() -
          //                   coro_io_issue_aligned_read_tmp->begin_ts)
          //                   .count();
          // std::cout<<time<<std::endl;
          int thread_id = coro_io_issue_aligned_read_tmp->thread_id;
          int coro_id = coro_io_issue_aligned_read_tmp->coro_id;
          // std::cout << "[Reap IO Thread]thread: " << thread_id << ", coro: "
          // << coro_id << std::endl;
          delete coro_io_issue_aligned_read_tmp;
          this->n_io_completed[thread_id][coro_id]++;

          if (this->n_io_completed[thread_id][coro_id] ==
                  this->n_io_executing[thread_id][coro_id] &&
              this->n_io_executing[thread_id][coro_id] > 0) {
            std::unique_lock<std::mutex> lk(coro_io_queue_mutex);
            this->n_io_executing[thread_id][coro_id] = 0;
            lk.unlock();
            this->n_io_completed[thread_id][coro_id] = 0;
            if (DEBUG_LOG) {
              std::cout << "[Reap IO Thread]Try to awake thr:" << thread_id
                        << ",coro:" << coro_id << std::endl;
            }
            atomic_mark[thread_id * max_ncoroutines + coro_id]++;
          }
        }
      }
      thread_stat->cpu_us += cpu_timer.elapsed();
    }
    thread_stat->total_us += all_timer.elapsed();
    if (verbose_)
      std::cout << "[Reap IO Thread]Exit." << std::endl;
    return;
  }

  template<typename T>
  void PQFlashIndex<T>::libaio_reap_io_thread(int          io_thread_id,
                                              ThreadStats *thread_stat,int this_thread_idx, int io_thread_num) {
    // 绑定线程核心 到最大工作线程数加1的位置
    // std::this_thread::sleep_for(std::chrono::milliseconds(10));
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(io_thread_id, &mask);
    pthread_t current_thread = pthread_self();
    // 将当前线程绑定到指定的核心
    if (pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &mask) != 0) {
      std::cerr << "Error binding thread to core " << io_thread_id << std::endl;
      exit(1);
    }
    Timer all_timer, cpu_timer, io_timer;
    all_timer.reset();
    cpu_timer.reset();
    if (DEBUG_LOG) {
      std::cout << "[Libaio Reap IO Thread]Enter thread." << std::endl;
    }

    while (true) {
      _u64 executing_thread_num_now;
      // std::unique_lock<std::mutex> lk(mtx);
      executing_thread_num_now = this->executing_thread_num;
      // lk.unlock();
      if ((executing_thread_num_now == 0)) {
        break;
      }
      for (size_t thread_idx = this_thread_idx; thread_idx < max_nthreads; thread_idx+=io_thread_num) {
        for (size_t coro_idx = 0; coro_idx < max_ncoroutines; coro_idx++) {
          if (atomic_mark[thread_idx*max_ncoroutines+coro_idx] == 0) {
            io_timer.reset();
            // std::cout<<ctx_vec[thread_idx][coro_idx]<<", libaio_cnt[thread_idx][coro_idx]:"<<libaio_cnt[thread_idx][coro_idx]<<std::endl;
            reader->get_events(ctx_vec[thread_idx][coro_idx],
                               libaio_cnt[thread_idx][coro_idx]);
            // std::cout<<"get_events succes"<<std::endl;
            // std::cout<<"atomic_mark[thread_idx*max_ncoroutines+coro_idx]:"<<atomic_mark[thread_idx*max_ncoroutines+coro_idx]<<std::endl;
            thread_stat->io_us += io_timer.elapsed();
            atomic_mark[thread_idx*max_ncoroutines+coro_idx]=1;
            // std::cout<<"atomic_mark[thread_idx*max_ncoroutines+coro_idx]:"<<atomic_mark[thread_idx*max_ncoroutines+coro_idx]<<std::endl;
          }
        }
      }
    }

    thread_stat->total_us += all_timer.elapsed();
    // if (verbose_)
      std::cout << "[Libaio Reap IO Thread]Exit." << std::endl;
    return;
  }

  template<typename T>
  void PQFlashIndex<T>::issue_io_thread(int io_thread_id,ThreadStats* thread_stat,int this_thread_idx, int io_thread_num) {
    // 绑定线程核心 到最大工作线程数加1的位置
    // std::this_thread::sleep_for(std::chrono::milliseconds(10));
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(io_thread_id, &mask);
    pthread_t current_thread = pthread_self();
    // 将当前线程绑定到指定的核心
    if (pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &mask) != 0) {
      std::cerr << "Error binding thread to core " << io_thread_id << std::endl;
      exit(1);
    }

    Timer all_timer, cpu_timer, io_timer;
    all_timer.reset();
    cpu_timer.reset();
    io_timer.reset();
    uint64_t num_issued = 0;
    
    std::cout << "[Issue IO Thread]Enter thread." << std::endl;
    // constexpr size_t kBatchSize = 1024;

    while(true){
        _u64 executing_thread_num_now;
        // std::unique_lock<std::mutex> lk(mtx);
        executing_thread_num_now = this->executing_thread_num;
        // lk.unlock();
        if( (executing_thread_num_now == 0) && this->batch_read_queue.empty()){
          thread_stat->cpu_us += cpu_timer.elapsed();
          break;
        }

        int result = 0;
        int kBatchSize = 128;
        ConcurrentQueue<AlignedRead>* q_ptr;
        // moodycamel::ConcurrentQueue<AlignedRead*> *q_ptr;

        for (size_t k = this_thread_idx; k < max_nthreads; k= k+io_thread_num) {
          q_ptr = batch_read_queue_thread[k];
          // q_ptr = q[k];
          // if (q_ptr->size_approx()>0) {
          if (!q_ptr->empty()) {
            cpu_timer.reset();
            std::vector<AlignedRead> iter_read =
                q_ptr->batch_pop(kBatchSize, result);
            // std::vector<AlignedRead*> tmp_vec;
            // tmp_vec.reserve(kBatchSize);
            // AlignedRead* tmp0 = tmp_vec[0];
            // std::cout<<"ptr0:"<<tmp0<<std::endl;
            // int result = q_ptr->try_dequeue_bulk(tmp_vec.begin(),kBatchSize);
            // std::cout<<"ptr1:"<<tmp_vec[0]<<std::endl;
            thread_stat->cpu_us += cpu_timer.elapsed();
            if (result < kBatchSize) {
              // std::cout << "IO not full.";
            }
            num_issued += result;
            if (true) {
            std::cout << "[Issue IO Thread]batch poped: " << result << " total: " << num_issued
                      << std::endl;
            }
            int cnt = 0;
            io_uring* ring_ptr = this->get_iouring(k,0);

            for (int idx = 0; idx < result; idx++) {
              // AlignedRead *tmp_ptr = &iter_read[idx];
              AlignedRead *tmp_ptr = new AlignedRead(iter_read[idx]);
              // AlignedRead *tmp_ptr = tmp_vec[idx];
              // std::cout<<"ptr:"<<tmp_ptr<<std::endl;
              tmp_ptr->begin_ts = std::chrono::high_resolution_clock::now();
              if (DEBUG_LOG) {
                std::cout << "[Issue IO Thread]";
                std::cout << tmp_ptr << ",";
                tmp_ptr->print();
              }
              io_uring_sqe *sqe = io_uring_get_sqe(ring_ptr);
              while (sqe == nullptr) {
                sqe = io_uring_get_sqe(ring_ptr);
              }

              io_uring_prep_read(sqe, this->get_index_fd(), tmp_ptr->buf,
                                 tmp_ptr->len, tmp_ptr->offset);
              io_uring_sqe_set_data(sqe, tmp_ptr);
              cnt++;
            }
            
            io_timer.reset();
            int res = io_uring_submit(ring_ptr);
            thread_stat->io_us += io_timer.elapsed();
            if (res != cnt) {
              std::cout << "io_uring_submit submitted less: " << res
                        << std::endl;
            }
          }
        }
    }
    thread_stat->total_us += all_timer.elapsed();
    if (verbose_)
      std::cout << "[Issue IO Thread]Exit." << std::endl;
    return;
  }
  template<typename T>
  void PQFlashIndex<T>::spdk_issue_io_thread(int io_thread_id,ThreadStats* thread_stat,int this_thread_idx, int io_thread_num) {
    // 绑定线程核心 到最大工作线程数加1的位置
    // std::this_thread::sleep_for(std::chrono::milliseconds(10));
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(io_thread_id, &mask);
    pthread_t current_thread = pthread_self();
    // 将当前线程绑定到指定的核心
    if (pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &mask) != 0) {
      std::cerr << "Error binding thread to core " << io_thread_id << std::endl;
      exit(1);
    }

    Timer all_timer, cpu_timer, io_timer,wait_timer;
    all_timer.reset();
    cpu_timer.reset();
    io_timer.reset();
    
    std::cout << "[SPDK Issue IO Thread]Enter thread." << std::endl;
    int uniqueReadNum = 0, allReadNum = 0;


    while(true){
        if( (this->executing_thread_num == 0) ){
          
          break;
        }
        std::vector<AlignedRead> collections;
        // collections.reserve(512);
        std::vector<std::pair<int,int>> collect_coro_id;

        int pop_cnt = 0;
        
        while(this->bqann_io_queues[this_thread_idx].size_approx() > 0 && pop_cnt < 512){
          cpu_timer.reset();
          std::pair<int,int> tmp_pair;
          this->bqann_io_queues[this_thread_idx].try_dequeue(tmp_pair);
          int thread_id = tmp_pair.first;
          int coro_id = tmp_pair.second;
          for(auto item: query_io_per_coro[thread_id][coro_id].aligned_read_vec){
            collections.emplace_back(item);
          }
          this->query_io_per_coro[thread_id][coro_id].submit_to_spdk();
          collect_coro_id.emplace_back(thread_id,coro_id);
          pop_cnt++;
          thread_stat->cpu_us += cpu_timer.elapsed();
        }
        
        // for (size_t thread_idx = this_thread_idx; thread_idx < max_nthreads; thread_idx+=io_thread_num) {
        //   for (size_t coro_idx = 0; coro_idx < max_ncoroutines; coro_idx++) {
        //     if (query_io_per_coro[thread_idx][coro_idx].valid) {
        //       io_timer.reset();
        //       wait_timer.reset();
        //       query_io_per_coro[thread_idx][coro_idx].valid = false;
        //       for(auto item: query_io_per_coro[thread_idx][coro_idx].aligned_read_vec){
        //         collections.emplace_back(item);
        //         // spdk_reader->SubmitRead4K(item,&cb,&atomic_mark[thread_idx * max_ncoroutines + coro_idx],this_thread_idx);
        //       }
        //       // std::cout<<atomic_mark[thread_idx * max_ncoroutines + coro_idx]<<std::endl;
        //       // collect_coro_id.emplace_back(thread_idx,coro_idx);
        //       thread_stat->io_submit_us += wait_timer.elapsed();
        //       thread_stat->io_us += io_timer.elapsed();
        //     }
        //   }
        // }
        wait_timer.reset();
        int    distinctNum = 0, allNum = 0;
        double replicatedRate = diskann::calculateBlockIdFrequency(
            collections, distinctNum, allNum);
        if (replicatedRate > 0) {
          LOG(INFO) << "Replicated Rate:" << replicatedRate;
        }
        uniqueReadNum += distinctNum;
        allReadNum += allNum;
        thread_stat->compute_us += wait_timer.elapsed();

        io_timer.reset();
        if (collections.size() > 0) {
          // std::cout << "collections.size() = " << collections.size()<<std::endl;
          wait_timer.reset();
          for(auto item : collections){
            // spdk_reader->SubmitRead4K(item,&cb,&atomic_mark[item.thread_id * max_ncoroutines + item.coro_id],this_thread_idx);
            spdk_reader->SubmitRead4K(
                item, &cb_in_queryIO_way,
                &query_io_per_coro[item.thread_id][item.coro_id],
                this_thread_idx);
            thread_stat->n_ios++;
          }
          thread_stat->io_submit_us += wait_timer.elapsed();

          // spdk_reader->BatchSyncRead4K(collections, this_thread_idx);
          // for (auto item : collect_coro_id) {
          //   int thread_idx = item.first;
          //   int coro_idx = item.second;
          //   atomic_mark[thread_idx * max_ncoroutines + coro_idx] =
          //       1;  // fetch atomic to resume worker coro.
          // }
          
        }
        wait_timer.reset();
        while (true) {
          int32_t return_val =
              spdk_reader->myPollCompleteQueue(this_thread_idx);
          if (return_val < 0) {
            LOG(ERROR) << "Poll return negated.";
          } else if (return_val == 0) {
            break;
          }
        }
        thread_stat->io_reap_us += wait_timer.elapsed();
        thread_stat->io_us += io_timer.elapsed();
    }
    thread_stat->total_us += all_timer.elapsed();
    LOG(INFO) << "[SPDK Issue IO Thread] dupRate: "<<(double)uniqueReadNum/allReadNum<<" distinct ios: "<<uniqueReadNum<<", total ios: "<<allReadNum;
    std::cout << "[SPDK Issue IO Thread]Exit." << std::endl;
    return;
  }

  
  
  
  template class PQFlashIndex<_u8>;
  template class PQFlashIndex<_s8>;
  template class PQFlashIndex<float>;
}  // namespace diskann