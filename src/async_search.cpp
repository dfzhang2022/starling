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

#include "cppcoro/sync_wait.hpp"
#include "cppcoro/task.hpp"
#include "cppcoro/when_all_ready.hpp"

// #include "cppcoro/io_service.hpp"
// #include "cppcoro/read_only_file.hpp"

// #include <boost/asio.hpp>

// #include <boost/lambda/lambda.hpp>
// // #include <boost/asio/random_access_file.hpp>
// #include <boost/asio/awaitable.hpp>
// #include <boost/asio/use_awaitable.hpp>
// #include <boost/asio/co_spawn.hpp>
// #include <boost/asio/detached.hpp>
// #include <boost/asio/posix/stream_descriptor.hpp>

// namespace as =  boost::asio;

// using asio_error = boost::system::error_code;

namespace diskann {
  static cppcoro::task<void> AsyncProcessPages(uint64_t   offset,
                                               char *buffer, size_t len,
                                               const BQANN::File &data_file,
                                               BQANN::IOUring    &ring,
                                               BQANN::Countdown  &countdown) {
    // std::cout << "AAAAAAAA" << std::endl;
    co_await data_file.AsyncReadBlock(ring, buffer, offset, len);
    // co_await IOUringAwaiter(ring, buffer, len, offset, data_file.getFd());
    std::cout << "DDDDDs" << std::endl;
    countdown.Decrement();
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
  static cppcoro::task<void> AsyncBatchRead(std::vector<AlignedRead> &aligned_read_vec,
                                          const BQANN::File &data_file,
                                          BQANN::IOUring &ring,
                                          BQANN::Countdown &countdown, int coro_id)
{
    // std::cout<<"In AsyncBatchRead(): aligned_read_vec_.size()"<<aligned_read_vec.size()<<std::endl;
    // for(int i = 0;i<1;i++){
        co_await data_file.AsyncBatchReadBlock(ring, aligned_read_vec, coro_id);
        // std::cout<<"Done "<<i<<" batch read in coro id:"<<coro_id<<std::endl;
    // }
    // std::cout<<"Done all batch read in coro id:"<<coro_id<<std::endl;
    countdown.Decrement();
}
  // cppcoro::task<std::uint64_t> count_lines(cppcoro::io_service &ioService,
  //                                          fs::path             path) {
  //   auto file = cppcoro::read_only_file::open(ioService, path);

  //   constexpr size_t bufferSize = 4096;
  //   auto             buffer = std::make_unique<std::uint8_t[]>(bufferSize);

  //   std::uint64_t newlineCount = 0;

  //   for (std::uint64_t offset = 0, fileSize = file.size(); offset < fileSize;) {
  //     const auto bytesToRead = static_cast<size_t>(
  //         std::min<std::uint64_t>(bufferSize, fileSize - offset));

  //     const auto bytesRead =
  //         co_await file.read(offset, buffer.get(), bytesToRead);

  //     newlineCount += std::count(buffer.get(), buffer.get() + bytesRead, '\n');

  //     offset += bytesRead;
  //   }

  //   co_return newlineCount;
  // }

  // cppcoro::task<> run(cppcoro::io_service &ioService) {
  //   cppcoro::io_work_scope ioScope(ioService);

  //   auto lineCount = co_await count_lines(ioService, fs::path{"foo.txt"});

  //   std::cout << "foo.txt has " << lineCount << " lines." << std::endl;
  //   ;
  // }

  // cppcoro::task<> process_events(cppcoro::io_service &ioService) {
  //   // 处理事件至 io_service 被停止时
  //   // 比如：当最后一个 io_work_scope 退出作用域时
  //   ioService.process_events();
  //   co_return;
  // }

  // // 异步读取 SSD 块的实现
  // boost::asio::awaitable<void> read_block(boost::asio::io_context &io_context,
  //                                         int fd, uint64_t offset, char *buffer,
  //                                         size_t len) {
  //   struct aiocb cb;  // Linux AIO Control Block

  //   // 初始化 AIO 控制块
  //   memset(&cb, 0, sizeof(cb));
  //   cb.aio_fildes = fd;      // 文件描述符
  //   cb.aio_buf = buffer;     // 缓冲区
  //   cb.aio_nbytes = len;     // 读取字节数
  //   cb.aio_offset = offset;  // 文件偏移量

  //   // 通过 boost::asio 调用异步 I/O 操作
  //   co_await boost::asio::use_awaitable([&cb, &io_context](auto &&handler) {
  //     // 使用 POSIX AIO 提交异步读取请求
  //     if (aio_read(&cb) == -1) {
  //       std::cerr << "aio_read failed: " << errno << std::endl;
  //       handler(boost::asio::error::operation_aborted);  // 失败时回调
  //       return;
  //     }

  //     // 等待读取完成
  //     while (aio_error(&cb) == EINPROGRESS) {
  //       // 等待 AIO 完成（这部分是异步的，协程会等待）
  //       boost::asio::steady_timer(io_context)
  //           .expires_after(std::chrono::milliseconds(10))
  //           .async_wait(std::move(handler));
  //       return;
  //     }

  //     // 确保读取没有出错
  //     int ret = aio_error(&cb);
  //     if (ret != 0) {
  //       std::cerr << "AIO read failed: " << strerror(ret) << std::endl;
  //       handler(boost::asio::error::operation_aborted);
  //       return;
  //     }

  //     handler(boost::asio::error::eof);  // 成功时回调
  //   });
  // }
  void boost_example() {
    // typedef std::istream_iterator<int> in;
    // as::io_context                     ioctx;
    // std::unique_ptr<std::vector<char>> my_buffer = ;

    // std::for_each(in(std::cin), in(), std::cout << (_1 * 3) << " ");
    // as::random_access_file file(ioctx, "/path/to/file",
    //                             boost::asio::random_access_file::read_only);

    // file.async_read_some_at(1234, my_buffer,
    //                         [](asio_error::error_code e, size_t n) {
    //                           // ...
    //                         });
  }
//   template<typename T>
//   void coro_search(const T *query1, const _u64 k_search, const _u32 mem_L,
//                    const _u64 l_search, _u64 *indices, float *distances,
//                    const _u64 beam_width, const _u32 io_limit,
//                    const bool use_reorder_data, const float use_ratio,
//                    QueryStats *stats, const std::string index_fname) {
    
//     QueryScratch<T> scratch;
//     _u64 coord_alloc_size = ROUND_UP(sizeof(T) * MAX_N_CMPS * this->aligned_dim, 256);
//     diskann::alloc_aligned((void **) &scratch.coord_scratch,
//                           coord_alloc_size, 256);
//     diskann::alloc_aligned((void **) &scratch.sector_scratch,
//                             (_u64) MAX_N_SECTOR_READS * (_u64) SECTOR_LEN,
//                             SECTOR_LEN);
//     diskann::alloc_aligned(
//         (void **) &scratch.aligned_pq_coord_scratch,
//         (_u64) MAX_GRAPH_DEGREE * (_u64) MAX_PQ_CHUNKS * sizeof(_u8), 256);
//     diskann::alloc_aligned((void **) &scratch.aligned_pqtable_dist_scratch,
//                             256 * (_u64) MAX_PQ_CHUNKS * sizeof(float), 256);
//     diskann::alloc_aligned((void **) &scratch.aligned_dist_scratch,
//                             (_u64) MAX_GRAPH_DEGREE * sizeof(float), 256);
//     diskann::alloc_aligned((void **) &scratch.aligned_query_T,
//                             this->aligned_dim * sizeof(T), 8 * sizeof(T));
//     diskann::alloc_aligned((void **) &scratch.aligned_query_float,
//                             this->aligned_dim * sizeof(float),
//                             8 * sizeof(float));
//     scratch.visited = new tsl::robin_set<_u64>(4096);
//     scratch.page_visited = new tsl::robin_set<unsigned>(4096);

//     memset(scratch.coord_scratch, 0, coord_alloc_size);
//     memset(scratch.aligned_query_T, 0, this->aligned_dim * sizeof(T));
//     memset(scratch.aligned_query_float, 0,
//             this->aligned_dim * sizeof(float));


//     // this->disk_index_file
//     // reader->open(index_fname);


//     float        query_norm = 0;
//     const T *    query = scratch.aligned_query_T;
//     const float *query_float = scratch.aligned_query_float;


//     uint32_t query_dim = metric == diskann::Metric::INNER_PRODUCT ? this-> data_dim - 1: this-> data_dim;

//     for (uint32_t i = 0; i < query_dim; i++) {
//       scratch.aligned_query_float[i] = query1[i];
//       scratch.aligned_query_T[i] = query1[i];
//       query_norm += query1[i] * query1[i];
//     }

//     // if inner product, we also normalize the query and set the last coordinate
//     // to 0 (this is the extra coordindate used to convert MIPS to L2 search)
//     if (metric == diskann::Metric::INNER_PRODUCT) {
//       query_norm = std::sqrt(query_norm);
//       scratch.aligned_query_T[this->data_dim - 1] = 0;
//       scratch.aligned_query_float[this->data_dim - 1] = 0;
//       for (uint32_t i = 0; i < this->data_dim - 1; i++) {
//         scratch.aligned_query_T[i] /= query_norm;
//         scratch.aligned_query_float[i] /= query_norm;
//       }
//     }

//     as::io_context                     ioctx;
//     auto       query_scratch = &(scratch);

//     as::random_access_file file(ioctx, index_fname.str(),
//                             boost::asio::random_access_file::read_only);
    


//     const BQANN::File index_file{index_fname.c_str(), File::kRead, true};
//     auto file_size = index_file.ReadSize();


//     // reset query
//     query_scratch->reset();

//     // pointers to buffers for data
//     T *   data_buf = query_scratch->coord_scratch;
//     _mm_prefetch((char *) data_buf, _MM_HINT_T1);

//     // sector scratch
//     char *sector_scratch = query_scratch->sector_scratch;
//     _u64 &sector_scratch_idx = query_scratch->sector_idx;

//     // query <-> PQ chunk centers distances
//     float *pq_dists = query_scratch->aligned_pqtable_dist_scratch;
//     pq_table.populate_chunk_distances(query_float, pq_dists);

//     // query <-> neighbor list
//     float *dist_scratch = query_scratch->aligned_dist_scratch;
//     _u8 *  pq_coord_scratch = query_scratch->aligned_pq_coord_scratch;

//     Timer                 query_timer, io_timer, cpu_timer;
//     query_timer.reset();
//     std::vector<Neighbor> retset(l_search + 1);
//     tsl::robin_set<_u64> &visited = *(query_scratch->visited);
//     tsl::robin_set<unsigned> &page_visited = *(query_scratch->page_visited);
//     unsigned cur_list_size = 0;

//     std::vector<Neighbor> full_retset;
//     full_retset.reserve(4096);
//     _u32                        best_medoid = 0;
//     float                       best_dist = (std::numeric_limits<float>::max)();
//     std::vector<SimpleNeighbor> medoid_dists;
//     for (_u64 cur_m = 0; cur_m < num_medoids; cur_m++) {
//       float cur_expanded_dist = dist_cmp_float->compare(
//           query_float, centroid_data + aligned_dim * cur_m,
//           (unsigned) aligned_dim);
//       if (cur_expanded_dist < best_dist) {
//         best_medoid = medoids[cur_m];
//         best_dist = cur_expanded_dist;
//       }
//     }

//     // lambda to batch compute query<-> node distances in PQ space
//     auto compute_pq_dists = [this, pq_coord_scratch, pq_dists](const unsigned *ids,
//                                                             const _u64 n_ids,
//                                                             float *dists_out) {
//       pq_flash_index_utils::aggregate_coords(ids, n_ids, this->data, this->n_chunks,
//                          pq_coord_scratch);
//       pq_flash_index_utils::pq_dist_lookup(pq_coord_scratch, n_ids, this->n_chunks, pq_dists,
//                        dists_out);
//     };

//     // 将id的点强制push进入full_retset
//     auto compute_extact_dists_and_push = [&](const char* node_buf, const unsigned id) -> float {
//       T *node_fp_coords_copy = data_buf;
//       memcpy(node_fp_coords_copy, node_buf, disk_bytes_per_point);
//       float cur_expanded_dist = dist_cmp->compare(query, node_fp_coords_copy,
//                                             (unsigned) aligned_dim);
//       full_retset.push_back(Neighbor(id, cur_expanded_dist, true));
//       return cur_expanded_dist;
//     };

//     auto compute_and_push_nbrs = [&](const char *node_buf, unsigned& nk) {
//       unsigned *node_nbrs = OFFSET_TO_NODE_NHOOD(node_buf);
//       unsigned nnbrs = *(node_nbrs++);
//       unsigned nbors_cand_size = 0;
//       for (unsigned m = 0; m < nnbrs; ++m) {
//         if (visited.find(node_nbrs[m]) == visited.end()) {
//           node_nbrs[nbors_cand_size++] = node_nbrs[m];
//           visited.insert(node_nbrs[m]);
//         }
//       }
//       if (nbors_cand_size) {
//         compute_pq_dists(node_nbrs, nbors_cand_size, dist_scratch);
//         for (unsigned m = 0; m < nbors_cand_size; ++m) {
//           const int nbor_id = node_nbrs[m];
//           const float nbor_dist = dist_scratch[m];
//           if (stats != nullptr) {
//             stats->n_cmps++;
//           }
//           if (nbor_dist >= retset[cur_list_size - 1].distance &&
//               (cur_list_size == l_search))
//             continue;
//           Neighbor nn(nbor_id, nbor_dist, true);
//           // Return position in sorted list where nn inserted
//           auto     r = InsertIntoPool(retset.data(), cur_list_size, nn);
//           if (cur_list_size < l_search) ++cur_list_size;
//           // nk logs the best position in the retset that was updated due to neighbors of n.
//           if (r < nk) nk = r;
//         }
//       }
//     };
//     // 计算 node_ids[] 若干个节点的pq距离并加入到retset中
//     auto compute_and_add_to_retset = [&](const unsigned *node_ids, const _u64 n_ids) {
//       compute_pq_dists(node_ids, n_ids, dist_scratch);
//       for (_u64 i = 0; i < n_ids; ++i) {
//         retset[cur_list_size].id = node_ids[i];
//         retset[cur_list_size].distance = dist_scratch[i];
//         retset[cur_list_size++].flag = true;
//         visited.insert(node_ids[i]);
//       }
//     };

//     if (mem_L) {
//       std::vector<unsigned> mem_tags(mem_L);
//       std::vector<float> mem_dists(mem_L);
//       std::vector<T*> res = std::vector<T*>();
//       mem_index_->search_with_tags(query, mem_L, mem_L, mem_tags.data(), mem_dists.data(), nullptr, res);
//       compute_and_add_to_retset(mem_tags.data(), std::min((unsigned)mem_L,(unsigned)l_search));
//     } else {
//       compute_and_add_to_retset(&best_medoid, 1);
//     }


//     std::sort(retset.begin(), retset.begin() + cur_list_size);

//     unsigned num_ios = 0;
//     unsigned k = 0;

//     // cleared every iteration
//     std::vector<unsigned> frontier;
//     frontier.reserve(2 * beam_width);
//     std::vector<std::pair<unsigned, char *>> frontier_nhoods;
//     frontier_nhoods.reserve(2 * beam_width);
//     std::vector<std::pair<unsigned, char *>> prefetch_frontier_nhoods;
//     prefetch_frontier_nhoods.reserve(2 * beam_width * affinity_size_);

//     std::vector<AlignedRead> frontier_read_reqs;
//     frontier_read_reqs.reserve(2 * beam_width);
//     std::vector<std::pair<unsigned, std::pair<unsigned, unsigned *>>>
//         cached_nhoods;
//     cached_nhoods.reserve(2 * beam_width);

//     while (k < cur_list_size && num_ios < io_limit) {
//       unsigned nk = cur_list_size;
//       // clear iteration state
//       frontier.clear();
//       frontier_nhoods.clear();
//       frontier_read_reqs.clear();
//       prefetch_frontier_nhoods.clear();
//       cached_nhoods.clear();
//       sector_scratch_idx = 0;
//       // find new beam
//       _u32 marker = k;
//       _u32 num_seen = 0;

//       // Log the id of block be visited.
//       std::vector<BlockVisited> block_visited_in_this_iter;
//       // distribute cache and disk-read nodes
//       while (marker < cur_list_size && frontier.size() < beam_width &&
//              num_seen < beam_width) {
//         const unsigned pid = id2page_[retset[marker].id];
//         if (page_visited.find(pid) == page_visited.end() && retset[marker].flag) {
//           num_seen++;
//           auto iter = nhood_cache.find(retset[marker].id);
//           if (iter != nhood_cache.end()) {
//             cached_nhoods.push_back(
//                 std::make_pair(retset[marker].id, iter->second));
//             if (stats != nullptr) {
//               stats->n_cache_hits++;
//             }
//           }
//           else {
//             frontier.push_back(retset[marker].id);
//             page_visited.insert(pid);
//           }
//           retset[marker].flag = false;
//         }
//         marker++;
//       }

//       // read nhoods of frontier ids
//       if (!frontier.empty()) {
//         if (stats != nullptr)
//           stats->n_hops++;
//         for (_u64 i = 0; i < frontier.size(); i++) {
//           auto                    id = frontier[i];
//           _u64 block_id              = static_cast<_u64>(id2page_[id]);
//           std::pair<_u32, char *> fnhood;
//           fnhood.first = id;
//           fnhood.second = sector_scratch + sector_scratch_idx * SECTOR_LEN;
//           sector_scratch_idx++;
//           frontier_nhoods.push_back(fnhood);
//           frontier_read_reqs.emplace_back(
//               (static_cast<_u64>(id2page_[id]+1)) * SECTOR_LEN, SECTOR_LEN,
//               fnhood.second);
//           if (stats != nullptr) {
//             stats->n_4k++;
//             stats->n_ios++;
//             block_visited_in_this_iter.push_back( BlockVisited(block_id, std::chrono::high_resolution_clock::now()));
//           }
//           num_ios++;
//         }
//         io_timer.reset();

//         // Countdown countdown(num_ring_entries);

//         constexpr uint64_t kNumConcurrentTasks = 64ull;
//         BQANN::Countdown   countdown(kNumConcurrentTasks);
//         BQANN::IOUring     ring(kNumConcurrentTasks);

//         std::vector<cppcoro::task<void>> tasks;
//         tasks.reserve(kNumConcurrentTasks);

//         for (auto &req : frontier_read_reqs) {
//           // TODO 补充完整协程的read
//           // co_await
//           // file.async_read_some_at(req.offset,req.buff,as::use_awaitable);
//           // co_await index_file.AsyncReadBlock(ring,)
//           tasks.emplace_back(AsyncProcessPages(ring, countdown));
//         }
//         tasks.emplace_back(DrainRing(req.offset, req.buff, req.len, index_file,
//                                      ring, countdown));
//         cppcoro::sync_wait(cppcoro::when_all_ready(std::move(tasks)));
//         if (this->count_visited_nodes) {
// #pragma omp critical
//           {
//             auto &cnt = this->node_visit_counter[retset[marker].id].second;
//             ++cnt;
//           }
//         }
//         if (stats != nullptr) {
//           stats->io_us += (double) io_timer.elapsed();
//           for (auto item : block_visited_in_this_iter) {
//             stats->block_visited_queue.push_back(BlockVisited(
//                 item.block_id, std::chrono::high_resolution_clock::now()));
//           }
//           block_visited_in_this_iter.clear();
//         }
//       }
//       cpu_timer.reset();

//       // process cached nhoods
//       for (auto &cached_nhood : cached_nhoods) {
//         auto id = cached_nhood.first;
//         auto  global_cache_iter = coord_cache.find(cached_nhood.first);
//         T *   node_fp_coords_copy = global_cache_iter->second;
//         unsigned nnr = cached_nhood.second.first;
//         unsigned* cnhood = cached_nhood.second.second;
//         char node_buf[max_node_len];
//         memcpy(node_buf, node_fp_coords_copy, disk_bytes_per_point);
//         memcpy((node_buf + disk_bytes_per_point), &nnr, sizeof(unsigned));
//         memcpy((node_buf + disk_bytes_per_point + sizeof(unsigned)), cnhood, sizeof(unsigned)*nnr);
//         compute_extact_dists_and_push(node_buf, id);
//         compute_and_push_nbrs(node_buf, nk);
//       }
//       if (stats != nullptr) {
//         stats->cpu_us += (double) cpu_timer.elapsed();
//       }

//       cpu_timer.reset();
//       // compute only the desired vectors in the pages - one for each page
//       // postpone remaining vectors to the next round
//       for (auto &frontier_nhood : frontier_nhoods) {
//         char    *sector_buf = frontier_nhood.second;
//         unsigned pid = id2page_[frontier_nhood.first];

//         for (unsigned j = 0; j < gp_layout_[pid].size(); ++j) {
//           unsigned id = gp_layout_[pid][j];
//           char    *node_buf = sector_buf + j * max_node_len;
//           compute_extact_dists_and_push(node_buf, id);
//           compute_and_push_nbrs(node_buf, nk);
//         }
//       }
//       if (stats != nullptr) {
//         stats->cpu_us += (double) cpu_timer.elapsed();
//       }

//       // update best inserted position
//       if (nk <= k)
//         k = nk;  // k is the best position in retset updated in this round.
//       else
//         ++k;
//     }


//     // re-sort by distance
//     std::sort(full_retset.begin(), full_retset.end(),
//               [](const Neighbor &left, const Neighbor &right) {
//                 return left.distance < right.distance;
//               });

//     // copy k_search values
//     _u64 t = 0;
//     for (_u64 i = 0; i < full_retset.size() && t < k_search; i++) {
//       if(i > 0 && full_retset[i].id == full_retset[i-1].id){
//         continue;
//       }
//       indices[t] = full_retset[i].id;
//       if (distances != nullptr) {
//         distances[t] = full_retset[i].distance;
//         if (metric == diskann::Metric::INNER_PRODUCT) {
//           // flip the sign to convert min to max
//           distances[t] = (-distances[t]);
//           // rescale to revert back to original norms (cancelling the effect of
//           // base and query pre-processing)
//           if (max_base_norm != 0)
//             distances[t] *= (max_base_norm * query_norm);
//         }
//       }
//       t++;
//     }

//     if (t < k_search) {
//       diskann::cerr << "The number of unique ids is less than topk" << std::endl;
//       exit(1);
//     }

//     this->thread_data.push(data);
//     this->thread_data.push_notify_all();

//     if (stats != nullptr) {
//       stats->total_us = (double) query_timer.elapsed();
//     }



//     co_return;
//   }

  // template<typename T>
  // void search(const std::vector<T *> query1, const _u64 k_search,
  //             const _u32 mem_L, const _u64 l_search,
  //             const std::vector<uint32_t *> indices,
  //             const std::vector<float *> distances, const _u64 beam_width,
  //             const _u32 io_limit, const bool use_reorder_data,
  //             const float use_ratio, QueryStats *stats, const size_t query_num,
  //             const unsigned recall_at) {

  //               if(query_num > 10)
  //               {
  //                 diskann::cerr<<"Coroutine search now is in testing, can
  //                 only handle 10 queries."<<diskann::endl;
  //               }
  //   as::io_context                     ioctx;
  //   // std::unique_ptr<std::vector<char>> my_buffer;
  //   // auto coro_search = [&]
  //   std::vector<float>
  // }
  // template<typename T>
  //  cppcoro::task<void> PQFlashIndex<T>::QueryCoro(const T *query1, const _u64 k_search,
  //                                    const _u32 mem_L, const _u64 l_search,
  //                                    _u64 *indices, float *distances,
  //                                    const _u64 beam_width, const _u32 io_limit,
  //                                    const bool  use_reorder_data,
  //                                    const float use_ratio, QueryStats *stats) {
    
  // }
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
    IOContext &ctx = data.ctx;
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
        for(int k = 0;k<coro_size;k++){
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
  template class PQFlashIndex<_u8>;
  template class PQFlashIndex<_s8>;
  template class PQFlashIndex<float>;

}  // namespace diskann