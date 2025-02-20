#include "cppcoro/sync_wait.hpp"
#include "cppcoro/task.hpp"
#include "cppcoro/when_all_ready.hpp"

// #include "concurrentqueue.h"

#include "liburing.h"

#include "utils.h"
#include "aux_utils.h"
#include "timer.h"
#include "linux_aligned_file_reader.h"
#include "percentile_stats.h"
#include "spdk_wrapper.h"

#include <iostream>
#include <chrono>
#include <stdio.h>
#include <fcntl.h>
#include <string.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <atomic>
#include <thread>
#include <array>
#include <atomic>

#include <glog/logging.h>

std::vector<bool> check(10,false);
std::vector<cppcoro::coroutine_handle<>> handle_map;

size_t final_coro_size = 2;

#define READ_SECTOR_LEN (size_t) 4096

#define QD	1024

#define IO_NUM 10000000

#define BEAMWIDTH 512

#define SECTOR_LEN (_u64) 512

#define LBA_SIZE 512

#define ISSUE_IO_NUM 4
#define REAP_IO_NUM 1

#define TEST_TIME 5



int all_fd = 0;
std::atomic<int> executing_thread = 0;
std::array<std::atomic<int>, ISSUE_IO_NUM> flag;
std::array<std::atomic<int>, ISSUE_IO_NUM> libaio_cnt;

std::vector<int> thread_cnter(ISSUE_IO_NUM,0);


struct libaio_stat{
  float io_time;
  float io_submit_time;
  float io_get_events_time;

  unsigned n_hops = 0;
};


class NewIORegisterAwaiter {
 public:
  NewIORegisterAwaiter(int coro_id) noexcept
      {
    coro_id_ = coro_id;
    std::cout <<"[coro] "<<coro_id_<<" awaiter_init"<<std::endl;
  }

  bool await_ready() const noexcept {
    std::cout <<"[coro] "<<coro_id_<<" awaiter_ready"<<std::endl;
    return false;
  }

  void await_suspend(cppcoro::coroutine_handle<> handle){
    this->handle_=handle;
    handle_map[coro_id_] = handle;
    check[coro_id_] = true;
    std::cout <<"[coro] "<<coro_id_<<" awaiter_suspend"<<std::endl;
    int flag = true;
    for (size_t i = 0; i < final_coro_size; i++) {
      flag = flag && check[i];
    }
    if(flag){
      handle_map[(coro_id_+1)%final_coro_size].resume();
    }
  }

  int await_resume() const noexcept {
    std::cout <<"[coro] "<<coro_id_<<" awaiter_resume"<<std::endl;
    // check[coro_id_] = false;
    return result_;
  }

  // void SetResult(int result) noexcept {
  //   result_ = result;
  // }

  // cppcoro::coroutine_handle<> GetHandle() const noexcept {
  //   return handle_;
  // }

 private:
  cppcoro::coroutine_handle<> handle_;
  int coro_id_;
  int result_;
};


cppcoro::task<void> count_lines(int k)
{
  int cnt = 3;
  while(cnt>0){

  
  std::cout<<k<<" in coro"<<std::endl;
  auto awaiter = new NewIORegisterAwaiter(k);
  std::cout<<"Before co_await"<<std::endl;
   co_await *awaiter;
  // if(k<1)
  // res = co_await *awaiter;
  std::cout<<k<<" back to coro"<<std::endl;
  cnt--;
  }
  std::cout<<k<<" end"<<std::endl;
  co_return;
}
cppcoro::task<void> sche(int k)
{
  for (int idx = 0; idx < k; idx++) {
    std::cout <<"[sche] " << idx << " before resume" << std::endl;
    handle_map[idx].resume();
    std::cout <<"[sche] "<< idx << " after resume" << std::endl;
  }

  co_return;
}
cppcoro::task<int> single(int k)
{

  std::cout<<"a"<<std::endl;
  co_return k;
}

cppcoro::task<void> main_task(int k){
  size_t coro_size = k;
  std::vector<cppcoro::task<void>> tasks;
  handle_map.reserve(coro_size);
  tasks.reserve(coro_size);
  for (size_t k = 0; k < coro_size; k++) {
    tasks.emplace_back(count_lines((k)));
    // tasks.emplace_back(single((k)));
  }
  // tasks.emplace_back(sche(coro_size));
  cppcoro::sync_wait(cppcoro::when_all_ready(std::move(tasks)));
  // while(true){
  // int m = 0;
  //   for (size_t k = 0; k < coro_size; k++) {
  //     m = co_await tasks[k];
  //     std::cout<<m<<std::endl;
  //   }
  //   for (size_t k = 0; k < coro_size; k++) {
  //     m = co_await tasks[k];
  //     std::cout<<m<<std::endl;
  //   }
  // // }
}

// void test_moodyqueue(){
//   moodycamel::ConcurrentQueue<AlignedRead*>* q;
//   q = new moodycamel::ConcurrentQueue<AlignedRead*>();
//   std::vector<AlignedRead*> tmp_vec;
//   for(size_t k = 0;k<5;k++){
//     AlignedRead* t_ptr =  new AlignedRead(k,1,nullptr);
//     tmp_vec.emplace_back(t_ptr);
//   }
//   for(size_t k = 0;k<5;k++){
//     q->enqueue(tmp_vec[k]);
//   }
//   std::cout<<q->size_approx()<<std::endl;

//   std::vector<AlignedRead*> another;
//   another.reserve(10);

//   int result = q->try_dequeue_bulk(another.begin(),7);

//   std::cout<<result<<std::endl;

//   std::cout<<q->size_approx()<<std::endl;

//   for(size_t k = 0;k<5;k++){
//     delete tmp_vec[k];
//   }
//   tmp_vec.clear();
//   delete q;
// }


int io_uring_test(){
  struct io_uring ring;
	int i, fd, ret, pending, done;
	struct io_uring_sqe *sqe;
	struct io_uring_cqe *cqe;
	struct iovec *iovecs;
	struct stat sb;
	ssize_t fsize;
	off_t offset;
	void *buf;

  std::string file_name = "/data/io_test/raid0-test.0.0";


	ret = io_uring_queue_init(QD, &ring, 0);
	if (ret < 0) {
		fprintf(stderr, "queue_init: %s\n", strerror(-ret));
		return 1;
	}

	fd = open(file_name.c_str(), O_RDONLY | O_DIRECT);
	if (fd < 0) {
		perror("open");
		return 1;
	}

	if (fstat(fd, &sb) < 0) {
		perror("fstat");
		return 1;
	}

	fsize = 0;
	iovecs = (iovec*)calloc(QD, sizeof(struct iovec));
	for (i = 0; i < QD; i++) {
		if (posix_memalign(&buf, 4096, 4096))
			return 1;
		iovecs[i].iov_base = buf;
		iovecs[i].iov_len = 4096;
		fsize += 4096;
	}

	offset = 0;
	i = 0;
	do {
		sqe = io_uring_get_sqe(&ring);
		if (!sqe)
			break;
		io_uring_prep_readv(sqe, fd, &iovecs[i], 1, offset);
		offset += iovecs[i].iov_len;
		i++;
		if (offset >= sb.st_size)
			break;
	} while (1);

	ret = io_uring_submit(&ring);
	if (ret < 0) {
		fprintf(stderr, "io_uring_submit: %s\n", strerror(-ret));
		return 1;
	} else if (ret != i) {
		fprintf(stderr, "io_uring_submit submitted less %d\n", ret);
		return 1;
	}

	done = 0;
	pending = ret;
	fsize = 0;
	for (i = 0; i < pending; i++) {
		ret = io_uring_wait_cqe(&ring, &cqe);
		if (ret < 0) {
			fprintf(stderr, "io_uring_wait_cqe: %s\n", strerror(-ret));
			return 1;
		}

		done++;
		ret = 0;
		if (cqe->res != 4096 && cqe->res + fsize != sb.st_size) {
			fprintf(stderr, "ret=%d, wanted 4096\n", cqe->res);
			ret = 1;
		}
		fsize += cqe->res;
		io_uring_cqe_seen(&ring, cqe);
		if (ret)
			break;
	}

	printf("Submitted=%d, completed=%d, bytes=%lu\n", pending, done,
						(unsigned long) fsize);
	close(fd);
	io_uring_queue_exit(&ring);
	for (i = 0; i < QD; i++)
		free(iovecs[i].iov_base);
	free(iovecs);
	return 0;
}

void issue_thread_v2(int io_thread_id, io_uring* ring) {
  cpu_set_t mask;
  CPU_ZERO(&mask);
  CPU_SET(io_thread_id, &mask);
  pthread_t current_thread = pthread_self();
  // 将当前线程绑定到指定的核心
  if (pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &mask) != 0) {
    std::cerr << "Error binding thread to core " << io_thread_id << std::endl;
    exit(1);
  }
  flag[io_thread_id] = 0;

  size_t         num_ios = 0;
  diskann::Timer all_timer, io_timer, wait_timer;
  double         io_time = 0, all_time = 0, wait_time = 0;

  char* sector_scratch = nullptr;

  diskann::alloc_aligned((void**) &sector_scratch,
                         (_u64) QD * (_u64) SECTOR_LEN, SECTOR_LEN);

  size_t sector_scratch_idx = 0;

  while (num_ios < IO_NUM) {
    sector_scratch_idx = 0;
    size_t in_queue_num = flag[io_thread_id];
    int tmp_cnt = 0;
    for (size_t read_idx = 0; read_idx < BEAMWIDTH-in_queue_num; read_idx++) {
      int   block_id = 100 * io_thread_id + read_idx * 100;
      char* tmp_buf = sector_scratch + sector_scratch_idx * SECTOR_LEN;
      sector_scratch_idx++;
      AlignedRead* tmp_ptr = new AlignedRead((block_id + 1) * SECTOR_LEN,
                                             SECTOR_LEN, tmp_buf, io_thread_id);

      io_uring_sqe* sqe = io_uring_get_sqe(ring);
      if (!sqe)
        std::cout << "[issue thread] failed get sqe." << std::endl;
      io_uring_prep_read(sqe, all_fd, tmp_ptr->buf, tmp_ptr->len,
                         tmp_ptr->offset);
      io_uring_sqe_set_data(sqe, tmp_ptr);
      num_ios++;
      tmp_cnt++;
    }
    
    // std::cout<<flag[io_thread_id]<<std::endl;
    io_timer.reset();
    flag[io_thread_id] += tmp_cnt;
    int res = io_uring_submit(ring);
    if (res != tmp_cnt) {
      std::cout << "io_uring_submit submitted less: " << res << std::endl;
    }
    io_time += io_timer.elapsed();
    wait_timer.reset();
    int k = flag[io_thread_id];
    while (k == BEAMWIDTH) {
      k = flag[io_thread_id];
    }
    // std::cout<<flag[io_thread_id]<<std::endl;
    wait_time += wait_timer.elapsed();
  }
  all_time = all_timer.elapsed();
  std::cout << "[Issue io] io proportion:" << io_time / all_time << std::endl;
  std::cout << "[Issue io] wait proportion:" << wait_time / all_time
            << std::endl;
  std::cout << "[Issue io] io ps:" << num_ios / all_time << std::endl;
  std::cout << "[Issue io] Exit." << io_thread_id << std::endl;
  executing_thread--;
}
void issue_thread(int io_thread_id, io_uring* ring) {
  cpu_set_t mask;
  CPU_ZERO(&mask);
  CPU_SET(io_thread_id, &mask);
  pthread_t current_thread = pthread_self();
  // 将当前线程绑定到指定的核心
  if (pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &mask) != 0) {
    std::cerr << "Error binding thread to core " << io_thread_id << std::endl;
    exit(1);
  }
  

  size_t         num_ios = 0;
  diskann::Timer all_timer, io_timer, wait_timer;
  double         io_time = 0, all_time = 0, wait_time = 0;

  char* sector_scratch = nullptr;

  diskann::alloc_aligned((void**) &sector_scratch,
                         (_u64) QD * (_u64) SECTOR_LEN, SECTOR_LEN);

  size_t sector_scratch_idx = 0;

  while (num_ios < IO_NUM) {
    sector_scratch_idx = 0;
    // int in_queue_num = flag[io_thread_id];
    int tmp_cnt = 0;
    for (size_t read_idx = 0; read_idx < BEAMWIDTH; read_idx++) {
      int   block_id = 100 * io_thread_id + read_idx * 100;
      char* tmp_buf = sector_scratch + sector_scratch_idx * SECTOR_LEN;
      sector_scratch_idx++;
      AlignedRead* tmp_ptr = new AlignedRead((block_id + 1) * SECTOR_LEN,
                                             SECTOR_LEN, tmp_buf, io_thread_id);

      io_uring_sqe* sqe = io_uring_get_sqe(ring);
      if (!sqe)
        std::cout << "[issue thread] failed get sqe." << std::endl;
      io_uring_prep_read(sqe, all_fd, tmp_ptr->buf, tmp_ptr->len,
                         tmp_ptr->offset);
      io_uring_sqe_set_data(sqe, tmp_ptr);
      num_ios++;
      tmp_cnt++;
    }
    
    // std::cout<<flag[io_thread_id]<<std::endl;
    io_timer.reset();
    flag[io_thread_id] = BEAMWIDTH;
    int res = io_uring_submit(ring);
    if (res != tmp_cnt) {
      std::cout << "io_uring_submit submitted less: " << res << std::endl;
    }
    io_time += io_timer.elapsed();
    wait_timer.reset();
    int k = flag[io_thread_id];
    while (k > 0) {
      k = flag[io_thread_id];
    }
    // std::cout<<flag[io_thread_id]<<std::endl;
    wait_time += wait_timer.elapsed();
  }
  all_time = all_timer.elapsed();
  std::cout << "[Issue io] io proportion:" << io_time / all_time << std::endl;
  std::cout << "[Issue io] wait proportion:" << wait_time / all_time
            << std::endl;
  std::cout << "[Issue io] io ps:" << num_ios / all_time << std::endl;
  std::cout << "[Issue io] Exit." << io_thread_id << std::endl;
  executing_thread--;
}

void reap_thread(int io_thread_id, std::vector<io_uring*> ring_vec,diskann::ThreadStats* stat){
  cpu_set_t mask;
  CPU_ZERO(&mask);
  CPU_SET(io_thread_id, &mask);
  pthread_t current_thread = pthread_self();
  // 将当前线程绑定到指定的核心
  if (pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &mask) != 0) {
      std::cerr << "Error binding thread to core " << io_thread_id << std::endl;
      exit(1);
  }
  // if (sched_setaffinity(0, sizeof(mask), &mask) == -1) {
  //   std::cout << "Could not set CPU affinity" << std::endl;
  // }

  constexpr size_t kBatchSize = 4096;
  

  size_t n_io = 0;
  diskann::Timer all_timer, cpu_timer, io_timer;
  double io_time = 0, all_time = 0;
  while(true){
      int tmp  = executing_thread;
      if(tmp == 0){
        break;
      }
      for(size_t k = io_thread_id-ISSUE_IO_NUM;k<ISSUE_IO_NUM;k = k+REAP_IO_NUM){
          io_uring* ring = ring_vec[k];
          std::array<io_uring_cqe *, kBatchSize> cqes;
          io_timer.reset();
          unsigned num_returned =
            io_uring_peek_batch_cqe(ring, cqes.data(), kBatchSize);
          io_time += io_timer.elapsed();
          // if (num_returned > 0)
          //   std::cout << num_returned <<" ";
          
          n_io+=num_returned;

          for (unsigned i = 0; i < num_returned; i++) {
            auto* coro_io_issue_aligned_read_tmp =
                reinterpret_cast<AlignedRead*>(io_uring_cqe_get_data(cqes[i]));
            io_uring_cqe_seen(ring, cqes[i]);
            if (cqes[i]->res != 4096) {
              std::cout << "Return num is not 4096." << cqes[i]->res << "<<<"
                        << std::endl;
            }
            thread_cnter[coro_io_issue_aligned_read_tmp->block_id]++;
            flag[coro_io_issue_aligned_read_tmp->block_id]--;
            delete coro_io_issue_aligned_read_tmp;

            // double time =
            // std::chrono::duration_cast<std::chrono::microseconds>(
            //                   std::chrono::high_resolution_clock::now() -
            //                   coro_io_issue_aligned_read_tmp->begin_ts)
            //                   .count();
            // std::cout<<time<<std::endl;
            // int thread_id = coro_io_issue_aligned_read_tmp->thread_id;
            // int coro_id = coro_io_issue_aligned_read_tmp->coro_id;
            // std::cout << "[Reap IO Thread]thread: " << thread_id << ", coro:
            // "
            // << coro_id << std::endl;
            // delete coro_io_issue_aligned_read_tmp;
            // this->n_io_completed[thread_id][coro_id]++;

            // if (this->n_io_completed[thread_id][coro_id] ==
            //         this->n_io_executing[thread_id][coro_id] &&
            //     this->n_io_executing[thread_id][coro_id] > 0) {
            //   std::unique_lock<std::mutex> lk(coro_io_queue_mutex);
            //   this->n_io_executing[thread_id][coro_id] = 0;
            //   lk.unlock();
            //   this->n_io_completed[thread_id][coro_id] = 0;
            //   if (DEBUG_LOG) {
            //     std::cout << "[Reap IO Thread]Try to awake thr:" << thread_id
            //               << ",coro:" << coro_id << std::endl;
            //   }
            //   atomic_mark[thread_id * max_ncoroutines + coro_id]++;
            // }
          }
      }

  }
  all_time += all_timer.elapsed();
  stat->n_ios = n_io;
  stat->total_us = all_time;

  // for(auto item : thread_cnter){
  //   std::cout<<"[Reap io]  reap io : "<<item<<std::endl;  
  // }
  std::cout<<"[Reap io]  io proportion:"<<io_time/all_time<<std::endl;
  std::cout<<"[Reap io]  reap io sum:"<<n_io<<std::endl;
  std::cout<<"[Reap io]  io ps:"<<n_io/all_time<<std::endl;
  std::cout<<"[Reap io] Exit."<<std::endl;
}

void libaio_reap_thread(int thread_id ,std::vector<io_context_t> ctx_vec, LinuxAlignedFileReader* reader,diskann::ThreadStats* stat){
  cpu_set_t mask;
  CPU_ZERO(&mask);
  CPU_SET(thread_id, &mask);
  pthread_t current_thread = pthread_self();
  // 将当前线程绑定到指定的核心
  if (pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &mask) != 0) {
    std::cerr << "Error binding thread to core " << thread_id << std::endl;
    exit(1);
  }

  diskann::Timer all_timer, io_timer, wait_timer;
  float         io_time = 0, all_time = 0, wait_time = 0;
  while(executing_thread >0)
  {
    for(size_t idx = 0;idx<ISSUE_IO_NUM;idx++){
      if(flag[idx]==0){
        wait_timer.reset();
        reader->get_events(ctx_vec[idx],libaio_cnt[idx]);
        stat->n_hops++;
        libaio_cnt[idx] = 0;
        flag[idx] = 1;
        wait_time += wait_timer.elapsed();
      }
    }
  }
  all_time += all_timer.elapsed();
  // std::cout << "[Libaio reap io] wait proportion:" << wait_time / all_time
  //           << std::endl;
  // std::cout << "[Libaio reap io] Exit." << thread_id << std::endl;
  std::cout << "[Libaio reap io] get_events_avg: " << wait_time/stat->n_hops << std::endl;
  return;
}

void libaio_thread(int thread_id ,io_context_t* ctx, LinuxAlignedFileReader* reader,diskann::ThreadStats* stat){
  cpu_set_t mask;
  CPU_ZERO(&mask);
  CPU_SET(thread_id, &mask);
  pthread_t current_thread = pthread_self();
  // 将当前线程绑定到指定的核心
  if (pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &mask) != 0) {
    std::cerr << "Error binding thread to core " << thread_id << std::endl;
    exit(1);
  }
  // std::cout << "[Issue io] IN"<< std::endl;
  size_t         num_ios = 0;
  diskann::Timer all_timer, io_timer, wait_timer;
  float         io_time = 0, all_time = 0, wait_time = 0;

  char* sector_scratch = nullptr;

  diskann::alloc_aligned((void**) &sector_scratch,
                         (_u64) QD * (_u64) SECTOR_LEN, SECTOR_LEN);
                         

  size_t sector_scratch_idx = 0;
  std::vector<AlignedRead> tmp_vec;
  std::vector<float> io_submit_costs;
  tmp_vec.reserve(BEAMWIDTH);

  while (num_ios < IO_NUM) {
    tmp_vec.clear();
    sector_scratch_idx = 0;

    for (size_t read_idx = 0; read_idx < BEAMWIDTH; read_idx++) {
      int   block_id = 100 * thread_id + read_idx * 100;
      char* tmp_buf = sector_scratch + sector_scratch_idx * SECTOR_LEN;
      sector_scratch_idx++;
      // AlignedRead* tmp_ptr = new AlignedRead((block_id + 1) * SECTOR_LEN,
      //                                        SECTOR_LEN, tmp_buf, thread_id);
      tmp_vec.emplace_back((block_id + 1) * SECTOR_LEN,
                                             SECTOR_LEN, tmp_buf, thread_id);
      num_ios++;
    }
    stat->n_hops++;
    io_timer.reset();
    int n_ops = reader->submit_reqs(tmp_vec, *ctx);
    flag[thread_id] = 0;
    libaio_cnt[thread_id] = n_ops;
    
    // reader->get_events(*ctx,n_ops);
    // reader->read(tmp_vec,*ctx);
    io_time += io_timer.elapsed();
    io_submit_costs.push_back(io_timer.elapsed());
    // for(auto item:tmp_vec){
    //   delete &item;
    // }
    wait_timer.reset();
    while(flag[thread_id] == 0)
    {
      continue;
    }
    wait_time += wait_timer.elapsed();
    
  }
  all_time = all_timer.elapsed();
  stat->io_us = io_time;
  stat->total_us = all_time;
  stat->n_ios = num_ios;
  float io_submit_cost_avg = 0;
  for(auto item :io_submit_costs){
    io_submit_cost_avg += item;
  }
  io_submit_cost_avg = io_submit_cost_avg/io_submit_costs.size();
  
  // std::cout << "[Issue io] io proportion:" << io_time / all_time << std::endl;
  // std::cout << "[Issue io] wait proportion:" << wait_time / all_time
  //           << std::endl;
  // std::cout << "[Issue io] io ps:" << num_ios / all_time << std::endl;
  // std::cout << "[Issue io] Exit." << thread_id << std::endl;
  // std::cout<<"[Issue io] io_submit avg cost:"<<io_time/stat->n_hops<<" BEAMWIDTH:"<<BEAMWIDTH<<std::endl;
  // std::cout<<"[Issue io] all_time:"<<all_time<<std::endl;
  diskann::aligned_free(sector_scratch);
  executing_thread--;
}
void cb(void *ctx, const struct spdk_nvme_cpl *cpl) {
  std::atomic<int> *p = (std::atomic<int> *)ctx;
  p->fetch_add(1);
}
void libaio_thread_spdk(int thread_id , ssdps::SpdkWrapper* reader,diskann::ThreadStats* stat){
  cpu_set_t mask;
  CPU_ZERO(&mask);
  CPU_SET(thread_id, &mask);
  pthread_t current_thread = pthread_self();
  // 将当前线程绑定到指定的核心
  if (pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &mask) != 0) {
    std::cerr << "Error binding thread to core " << thread_id << std::endl;
    exit(1);
  }
  LOG(INFO) << "[SPDK Issue io] "<<thread_id;
  size_t         num_ios = 0;
  diskann::Timer all_timer, io_timer,sub_timer, wait_timer;
  float         io_time = 0, all_time = 0, wait_time = 0,io_submit_time = 0, io_getevents_time = 0;

  // char* sector_scratch = nullptr;

  // diskann::alloc_aligned((void**) &sector_scratch,
  //                        (_u64) QD * (_u64) SECTOR_LEN, SECTOR_LEN);
  char *sector_scratch = (char *)spdk_zmalloc(QD*SECTOR_LEN, LBA_SIZE, NULL, SPDK_ENV_SOCKET_ID_ANY,
                          SPDK_MALLOC_DMA);
  size_t sector_scratch_idx = 0;
  std::vector<float> io_submit_costs,io_getevents_costs;

  while (num_ios < IO_NUM) {

    sector_scratch_idx = 0;
    std::atomic<int> counter{0};
    io_timer.reset();
    sub_timer.reset();
    for (int i = 0; i < BEAMWIDTH; i++) {
      int   block_id =  i;
      char* tmp_buf = sector_scratch + sector_scratch_idx * SECTOR_LEN;
      sector_scratch_idx++;
      
      block_id = block_id*(SECTOR_LEN/LBA_SIZE);
      reader->SubmitReadCommand((void*) tmp_buf, block_id, i,
                             cb, &counter, thread_id);
      num_ios++;
    }
    io_submit_time+= sub_timer.elapsed();
    wait_timer.reset();
    while (counter != BEAMWIDTH) reader->PollCompleteQueue(thread_id);
    stat->n_hops++;
    wait_time += wait_timer.elapsed();
    io_time += io_timer.elapsed();
    
  }
  all_time = all_timer.elapsed();
  stat->io_us = io_time;
  stat->total_us = all_time;
  stat->n_ios = num_ios;
  stat->io_submit_us = io_submit_time;
  stat->io_reap_us = wait_time;
  // float io_submit_cost_avg = 0;
  // for(auto item :io_submit_costs){
  //   io_submit_cost_avg += item;
  // }
  // io_submit_cost_avg = io_submit_cost_avg/io_submit_costs.size();

}

void libaio_thread_sync(int thread_id ,io_context_t* ctx, LinuxAlignedFileReader* reader,diskann::ThreadStats* stat){
  cpu_set_t mask;
  CPU_ZERO(&mask);
  CPU_SET(thread_id, &mask);
  pthread_t current_thread = pthread_self();
  // 将当前线程绑定到指定的核心
  if (pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &mask) != 0) {
    std::cerr << "Error binding thread to core " << thread_id << std::endl;
    exit(1);
  }
  // std::cout << "[Issue io] IN"<< std::endl;
  size_t         num_ios = 0;
  diskann::Timer all_timer, io_timer,sub_timer, wait_timer;
  float         io_time = 0, all_time = 0, wait_time = 0,io_submit_time = 0, io_getevents_time = 0;

  char* sector_scratch = nullptr;

  diskann::alloc_aligned((void**) &sector_scratch,
                         (_u64) QD * (_u64) SECTOR_LEN, SECTOR_LEN);
                         

  size_t sector_scratch_idx = 0;
  std::vector<AlignedRead> tmp_vec;
  std::vector<float> io_submit_costs,io_getevents_costs;
  tmp_vec.reserve(BEAMWIDTH);

  while (num_ios < IO_NUM) {
    tmp_vec.clear();
    sector_scratch_idx = 0;

    for (size_t read_idx = 0; read_idx < BEAMWIDTH; read_idx++) {
      int   block_id = 100 * thread_id + read_idx * 100;
      char* tmp_buf = sector_scratch + sector_scratch_idx * SECTOR_LEN;
      sector_scratch_idx++;
      // AlignedRead* tmp_ptr = new AlignedRead((block_id + 1) * SECTOR_LEN,
      //                                        SECTOR_LEN, tmp_buf, thread_id);
      tmp_vec.emplace_back((block_id + 1) * SECTOR_LEN,
                                             SECTOR_LEN, tmp_buf, thread_id);
      num_ios++;
    }
    stat->n_hops++;
    io_timer.reset();
    sub_timer.reset();
    int n_ops = reader->submit_reqs(tmp_vec, *ctx);
    io_submit_time+=sub_timer.elapsed();
    
    sub_timer.reset();
    reader->get_events(*ctx,n_ops);
    io_getevents_time+=sub_timer.elapsed();
    // reader->read(tmp_vec,*ctx);
    io_time += io_timer.elapsed();
    wait_timer.reset();
    wait_time += wait_timer.elapsed();
    
  }
  all_time = all_timer.elapsed();
  stat->io_us = io_time;
  stat->total_us = all_time;
  stat->n_ios = num_ios;
  stat->io_submit_us = io_submit_time;
  stat->io_reap_us = io_getevents_time;
  float io_submit_cost_avg = 0;
  for(auto item :io_submit_costs){
    io_submit_cost_avg += item;
  }
  io_submit_cost_avg = io_submit_cost_avg/io_submit_costs.size();
  
  // std::cout << "[Issue io] io proportion:" << io_time / all_time << std::endl;
  // std::cout << "[Issue io] wait proportion:" << wait_time / all_time
  //           << std::endl;
  // std::cout << "[Issue io] io ps:" << num_ios / all_time << std::endl;
  // std::cout << "[Issue io] Exit." << thread_id << std::endl;
  // std::cout<<"[Issue io] io_submit avg cost:"<<io_submit_time/stat->n_hops<<" BEAMWIDTH:"<<BEAMWIDTH<<std::endl;
  // std::cout<<"[Issue io] get_events avg cost:"<<io_getevents_time/stat->n_hops<<" BEAMWIDTH:"<<BEAMWIDTH<<std::endl;
  // std::cout<<"[Issue io] io avg cost:"<<io_time/stat->n_hops<<" BEAMWIDTH:"<<BEAMWIDTH<<std::endl;
  // std::cout<<"[Issue io] all_time:"<<all_time<<std::endl;
  diskann::aligned_free(sector_scratch);
  executing_thread--;
}
float multithread_libaio(){
  std::string file_name = "/data/dataset/indices/bigann_100m_M200_R64_L100_B5/_disk.index";
  LinuxAlignedFileReader* reader = new LinuxAlignedFileReader();
  reader->open(file_name);
  std::vector<io_context_t> ctx_vec;
  for (size_t i = 0;i<ISSUE_IO_NUM;i++) {
    io_context_t a = 0;
    int ret = io_setup(QD, &a);
    if (ret != 0) {
      assert(errno != EAGAIN);
      assert(errno != ENOMEM);
      std::cerr << "io_setup() failed; returned " << ret << ", errno=" << errno
                << ":" << ::strerror(errno) << std::endl;
      return 0;
    } else {
      // diskann::cout<<i << " allocating ctx: " << a<< std::endl;
    }
    ctx_vec.emplace_back(a);
  }
  auto thread_stats = new diskann::ThreadStats[ISSUE_IO_NUM];


  std::vector<std::thread> all_threads;
  executing_thread = ISSUE_IO_NUM;
  for (_u64 i = 0; i < ISSUE_IO_NUM; i++) {
    std::string thread_name = "ISSUE-IO" + std::to_string(i);
    std::thread t(libaio_thread,i, &ctx_vec[i],reader,thread_stats+i);
    pthread_t pthread_handle =
          *reinterpret_cast<pthread_t *>(t.native_handle());
    pthread_setname_np(pthread_handle, thread_name.c_str());
    all_threads.push_back(std::move(t));
  }
  for (_u64 i = ISSUE_IO_NUM; i < ISSUE_IO_NUM+REAP_IO_NUM; i++) {
    std::string thread_name = "LibaioREAP-IO" + std::to_string(i);
    std::thread t(libaio_reap_thread,i, ctx_vec,reader,thread_stats+i);
    pthread_t pthread_handle =
          *reinterpret_cast<pthread_t *>(t.native_handle());
    pthread_setname_np(pthread_handle, thread_name.c_str());
    all_threads.push_back(std::move(t));
  }
  for (auto& t : all_threads) {
        t.join();
  }
  double all_time = 0, num_io = 0,iops = 0,io_time = 0,io_submit_time = 0, io_reap_time = 0;
  unsigned all_n_hops = 0;
  for (_u64 i = 0; i < ISSUE_IO_NUM; i++) {
    all_n_hops += thread_stats[i].n_hops;
    io_time += thread_stats[i].io_us;
    io_submit_time += thread_stats[i].io_submit_us;
    io_reap_time += thread_stats[i].io_reap_us;
    num_io=thread_stats[i].n_ios;
    all_time=thread_stats[i].total_us / (1000*1000);
    iops += num_io/all_time;
  }
  float res = io_time/all_n_hops;
  std::cout<<"[LIBAIO] iops: "<<(int)iops<<std::endl;
  std::cout<<"[LIBAIO] avg io_submit latency: "<<io_submit_time/all_n_hops<<std::endl;
  std::cout<<"[LIBAIO] avg io_reap latency: "<<io_reap_time/all_n_hops<<std::endl;
  std::cout<<"[LIBAIO] avg io latency: "<<io_time/all_n_hops<<std::endl;

  reader->close();
  for (size_t i = 0;i<ISSUE_IO_NUM;i++) {
    io_destroy(ctx_vec[i]);
  }
  return res;

}


float multithread_libaio_sync(float& io_latency, float & io_submit_latency, float & io_getevents_latency){
  std::string file_name = "/data/dataset/indices/bigann_100m_M200_R64_L100_B5/_disk.index";
  LinuxAlignedFileReader* reader = new LinuxAlignedFileReader();
  reader->open(file_name);
  std::vector<io_context_t> ctx_vec;
  for (size_t i = 0;i<ISSUE_IO_NUM;i++) {
    io_context_t a = 0;
    int ret = io_setup(QD, &a);
    if (ret != 0) {
      assert(errno != EAGAIN);
      assert(errno != ENOMEM);
      std::cerr << "io_setup() failed; returned " << ret << ", errno=" << errno
                << ":" << ::strerror(errno) << std::endl;
      return 0;
    } else {
      // diskann::cout<<i << " allocating ctx: " << a<< std::endl;
    }
    ctx_vec.emplace_back(a);
  }
  auto thread_stats = new diskann::ThreadStats[ISSUE_IO_NUM];


  std::vector<std::thread> all_threads;
  executing_thread = ISSUE_IO_NUM;
  for (_u64 i = 0; i < ISSUE_IO_NUM; i++) {
    std::string thread_name = "ISSUE-IO" + std::to_string(i);
    std::thread t(libaio_thread_sync,i, &ctx_vec[i],reader,thread_stats+i);
    pthread_t pthread_handle =
          *reinterpret_cast<pthread_t *>(t.native_handle());
    pthread_setname_np(pthread_handle, thread_name.c_str());
    all_threads.push_back(std::move(t));
  }

  for (auto& t : all_threads) {
        t.join();
  }
  double all_time = 0, num_io = 0,iops = 0,io_time = 0,io_submit_time = 0, io_reap_time = 0;
  unsigned all_n_hops = 0;
  for (_u64 i = 0; i < ISSUE_IO_NUM; i++) {
    all_n_hops += thread_stats[i].n_hops;
    io_time += thread_stats[i].io_us;
    io_submit_time += thread_stats[i].io_submit_us;
    io_reap_time += thread_stats[i].io_reap_us;
    num_io=thread_stats[i].n_ios;
    all_time=thread_stats[i].total_us / (1000*1000);
    iops += num_io/all_time;
  }
  float res = io_time/all_n_hops;
  std::cout<<"[LIBAIO] iops: "<<(int)iops<<std::endl;
  std::cout<<"[LIBAIO] avg io_submit latency: "<<io_submit_time/all_n_hops<<std::endl;
  std::cout<<"[LIBAIO] avg io_reap latency: "<<io_reap_time/all_n_hops<<std::endl;
  std::cout<<"[LIBAIO] avg io latency: "<<io_time/all_n_hops<<std::endl;

  io_latency = io_time/all_n_hops;
  io_submit_latency = io_submit_time/all_n_hops;
  io_getevents_latency = io_reap_time/all_n_hops;

  reader->close();
  for (size_t i = 0;i<ISSUE_IO_NUM;i++) {
    io_destroy(ctx_vec[i]);
  }
  return res;

}


float multithread_libaio_spdk(float& io_latency, float & io_submit_latency, float & io_getevents_latency){
  
  auto thread_stats = new diskann::ThreadStats[ISSUE_IO_NUM];
  std::shared_ptr<ssdps::SpdkWrapper> spdk_reader = ssdps::SpdkWrapper::create(4);
  spdk_reader->Init();

  std::vector<std::thread> all_threads;
  executing_thread = ISSUE_IO_NUM;
  for (_u64 i = 0; i < ISSUE_IO_NUM; i++) {
    std::string thread_name = "ISSUE-IO" + std::to_string(i);
    std::thread t(libaio_thread_spdk,i,spdk_reader.get() ,thread_stats+i);
    pthread_t pthread_handle =
          *reinterpret_cast<pthread_t *>(t.native_handle());
    pthread_setname_np(pthread_handle, thread_name.c_str());
    all_threads.push_back(std::move(t));
  }

  for (auto& t : all_threads) {
        t.join();
  }
  double all_time = 0, num_io = 0,iops = 0,io_time = 0,io_submit_time = 0, io_reap_time = 0, io_wait_time = 0;
  unsigned all_n_hops = 0;
  for (_u64 i = 0; i < ISSUE_IO_NUM; i++) {
    all_n_hops += thread_stats[i].n_hops;
    io_time += thread_stats[i].io_us;
    io_submit_time += thread_stats[i].io_submit_us;
    io_reap_time += thread_stats[i].io_reap_us;
    num_io=thread_stats[i].n_ios;
    all_time=thread_stats[i].total_us / (1000*1000);
    iops += num_io/all_time;
  }
  float res = io_time/all_n_hops;
  LOG(INFO)<<"[LIBAIO] iops: "<<(int)iops<<std::endl;
  LOG(INFO)<<"[LIBAIO] avg io_submit latency: "<<io_submit_time/all_n_hops<<std::endl;
  LOG(INFO)<<"[LIBAIO] avg io_reap latency: "<<io_reap_time/all_n_hops<<std::endl;
  LOG(INFO)<<"[LIBAIO] avg io latency: "<<io_time/all_n_hops<<std::endl;

  io_latency = io_time/all_n_hops;
  io_submit_latency = io_submit_time/all_n_hops;
  io_getevents_latency = io_reap_time/all_n_hops;

  return res;

}

void multithread_io_uring(){
  std::string file_name = "/data/dataset/indices/bigann_100m_M200_R64_L100_B5/_disk.index";
  all_fd = open(file_name.c_str(), O_RDONLY | O_DIRECT);

  std::vector<io_uring*> ring_vec;
  int cnt =0;
  while (cnt<ISSUE_IO_NUM){
    ring_vec.emplace_back(new io_uring);
    // int ret = io_uring_queue_init(QD, ring_vec[cnt], IORING_SETUP_SQPOLL);
    int ret = io_uring_queue_init(QD, ring_vec[cnt], 0);
    // int ret = io_uring_queue_init(QD, ring_vec[cnt], IORING_SETUP_KERNEL_POOL);
    if (ret < 0) {
      LOG(INFO)<<"io_uring init failed at "<<cnt<<std::endl;
      
    }
    cnt++;
  }
  auto thread_stats = new diskann::ThreadStats[REAP_IO_NUM];

  std::vector<std::thread> all_threads;
  executing_thread = ISSUE_IO_NUM;
  for (_u64 i = 0; i < ISSUE_IO_NUM; i++) {
    std::string thread_name = "ISSUE-IO" + std::to_string(i);
    std::thread t(issue_thread,i, ring_vec[i]);
    pthread_t pthread_handle =
          *reinterpret_cast<pthread_t *>(t.native_handle());
    pthread_setname_np(pthread_handle, thread_name.c_str());
    all_threads.push_back(std::move(t));
  }
  for (_u64 i = ISSUE_IO_NUM; i < ISSUE_IO_NUM+REAP_IO_NUM; i++) {
    std::string thread_name = "REAP-IO" + std::to_string(i);
    std::thread t(reap_thread,i, ring_vec,thread_stats+(i-ISSUE_IO_NUM));
    pthread_t pthread_handle =
          *reinterpret_cast<pthread_t *>(t.native_handle());
    pthread_setname_np(pthread_handle, thread_name.c_str());
    all_threads.push_back(std::move(t));
  }
  
  for (auto& t : all_threads) {
        t.join();
  }

  double all_time = 0, num_io = 0,iops = 0;
  for (_u64 i = 0; i < REAP_IO_NUM; i++) {
    num_io=thread_stats[i].n_ios;
    all_time=thread_stats[i].total_us / (1000*1000);
    iops += num_io/all_time;
  }
  LOG(INFO)<<"[IO_URING] iops: "<<(int)iops<<std::endl;

  close(all_fd);
	cnt =0;
  while (cnt<ISSUE_IO_NUM){
    io_uring_queue_exit(ring_vec[cnt]);
    cnt++;
  }
  ring_vec.clear();
}

void writeIndexToSPDK(std::string indexname, ssdps::SpdkWrapper* reader){
  auto meta_pair = diskann::get_disk_index_meta(indexname);
  _u64 actual_index_size = get_file_size(indexname);
  _u64 expected_file_size, expected_npts;
  _u64                               _nd;
  _u64                               max_node_len;

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
  // if (expected_npts != _nd) {
  //   diskann::cout << "expect _nd: " << _nd
  //                 << " actual _nd: " << expected_npts << std::endl;
  //   exit(-1);
  // }
  max_node_len = meta_pair.second[3];
  unsigned nnodes_per_sector = meta_pair.second[4];
  // if (SECTOR_LEN / max_node_len != C) {
  //   diskann::cout << "nnodes per sector: " << SECTOR_LEN / max_node_len << " C: " << C
  //                 << std::endl;
  //   exit(-1);
  // }
  _u64 file_size = READ_SECTOR_LEN + READ_SECTOR_LEN * ((expected_npts + nnodes_per_sector - 1) / nnodes_per_sector);
  LOG(INFO)<<"file_size:  "<<file_size<<std::endl;
  std::unique_ptr<char[]> mem_index =
      std::make_unique<char[]>(file_size);
  std::ifstream diskann_reader(indexname);
  diskann_reader.read(mem_index.get(),file_size);

  unsigned batch_size = 1024;
  unsigned sector_size = ((expected_npts + nnodes_per_sector - 1) / nnodes_per_sector) + 1;
  LOG(INFO)<<"sector_size:  "<<sector_size<<std::endl;

  unsigned wrt_idx = 0;

  char *buf_2 = (char *)spdk_zmalloc(READ_SECTOR_LEN* batch_size, READ_SECTOR_LEN, NULL,
    SPDK_ENV_SOCKET_ID_ANY, SPDK_MALLOC_DMA);
  
  while(wrt_idx<sector_size){
    unsigned wrt_size_iter = std::min(sector_size - wrt_idx , batch_size);
    memcpy(buf_2,mem_index.get()+wrt_idx*READ_SECTOR_LEN,wrt_size_iter*READ_SECTOR_LEN);
    int res = memcmp(buf_2,mem_index.get()+wrt_idx*READ_SECTOR_LEN,wrt_size_iter*READ_SECTOR_LEN);
    if(res!=0){
      LOG(INFO)<<res<<" "<<wrt_idx<<std::endl;
    }
    reader->SyncWrite(buf_2,wrt_size_iter*READ_SECTOR_LEN,wrt_idx,0);
    wrt_idx += wrt_size_iter;
  }
  reader->SyncRead(buf_2,batch_size*READ_SECTOR_LEN,0,0);
  int res = memcmp(buf_2,mem_index.get()+0*READ_SECTOR_LEN,batch_size*READ_SECTOR_LEN);
  LOG(INFO)<<res<<std::endl;
  LOG(INFO)<<wrt_idx<<std::endl;

  spdk_free(buf_2);


}

int main(int argc, char* argv[]) {

  google::InitGoogleLogging(argv[0]);
  // FLAGS_log_dir = "./logs";
  FLAGS_logtostderr = false;  // 不输出到标准错误流
  FLAGS_alsologtostderr = true;
  FLAGS_minloglevel = 0;  // 0: INFO, 1: WARNING, 2: ERROR, 3: FATAL

  // int                              coro_size = 4;
  // std::vector<cppcoro::task<void>> tasks;
  // handle_map.reserve(coro_size);
  // tasks.reserve(coro_size+1);
  // for (size_t k = 0; k < coro_size; k++) {
  //   tasks.emplace_back(count_lines((k)));
  // }
  // tasks.emplace_back(sche(coro_size));
  // cppcoro::sync_wait(cppcoro::when_all_ready(std::move(tasks)));

  // cppcoro::sync_wait(main_task(final_coro_size));
  
  // test_moodyqueue();
  // io_uring_test();

  // multithread_io_uring();
  // float io_time = 0, submit_time = 0, reap_time = 0;
  // for(int k = 0;k< TEST_TIME;k++){
  //   // res+=multithread_libaio();
  //   std::cout<<"Time "<<k<<std::endl;
  //   float a = 0,b = 0,c = 0;
  //   multithread_libaio_sync(a,b,c);
  //   io_time+=a;
  //   submit_time += b;
  //   reap_time+=c;
  // }
  
  // std::cout<<"Final io avg: "<<io_time/TEST_TIME<< " for "<<BEAMWIDTH<<std::endl;
  // std::cout<<"Final submit avg: "<<submit_time/TEST_TIME<< " for "<<BEAMWIDTH<<std::endl;
  // std::cout<<"Final reap avg: "<<reap_time/TEST_TIME<< " for "<<BEAMWIDTH<<std::endl;
  // std::cout<<io_time/TEST_TIME<<" "<<submit_time/TEST_TIME<<" "<<reap_time/TEST_TIME<<std::endl;

  // std::shared_ptr<ssdps::SpdkWrapper> spdk_reader = ssdps::SpdkWrapper::create(1);
  // spdk_reader->Init();
  // char *sector_scratch = (char *)spdk_zmalloc(LBA_SIZE*QD, LBA_SIZE, NULL, SPDK_ENV_SOCKET_ID_ANY,
  //   SPDK_MALLOC_DMA);
  // spdk_reader->SyncRead(sector_scratch,LBA_SIZE,0,0);

  float a = 0,b = 0,c = 0;
  multithread_libaio_spdk(a,b,c);
  // writeIndexToSPDK("/data/dataset/indices/bigann_100m_M200_R64_L100_B5/_disk.index",spdk_reader.get());

  return 0;
}