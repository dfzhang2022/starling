#ifndef STORAGE_IO_URING_H_
#define STORAGE_IO_URING_H_

#include <array>
#include <cstddef>
#include <system_error>
#include <map>

#include "cppcoro/coroutine.hpp"
#include "cppcoro/task.hpp"
#include "liburing.h"
#include "aligned_file_reader.h"

#define IS_ALIGNED(X, Y) ((uint64_t)(X) % (uint64_t)(Y) == 0)
#define IS_512_ALIGNED(X) IS_ALIGNED(X, 512)

namespace BQANN {

  // struct AlignedRead {
  //   uint64_t offset;  // where to read from
  //   uint64_t len;     // how much to read
  //   void    *buf;     // where to read into

  //   AlignedRead() : offset(0), len(0), buf(nullptr) {
  //   }

  //   AlignedRead(uint64_t offset, uint64_t len, void *buf)
  //       : offset(offset), len(len), buf(buf) {
  //     assert(IS_512_ALIGNED(offset));
  //     assert(IS_512_ALIGNED(len));
  //     assert(IS_512_ALIGNED(buf));
  //     // assert(malloc_usable_size(buf) >= len);
  //   }
  // };

  struct CoroBatchReadData {
    std::coroutine_handle<> handle;
    int                     batch_num{-1};
    int                     coro_idx{-1};
  };

class IOUring;

class IOUringAwaiter {
 public:
  IOUringAwaiter(IOUring &ring, void *buffer, size_t num_bytes, off_t offset,
                 int fd) noexcept
      : ring_(ring),
        buffer_(buffer),
        num_bytes_(num_bytes),
        offset_(offset),
        fd_(fd) {}

  bool await_ready() const noexcept { return false; }

  void await_suspend(cppcoro::coroutine_handle<> handle);

  __s32 await_resume() const noexcept { return result_; }

  void SetResult(__s32 result) noexcept { result_ = result; }

  cppcoro::coroutine_handle<> GetHandle() const noexcept { return handle_; }

 private:
  cppcoro::coroutine_handle<> handle_;
  IOUring &ring_;
  void *buffer_;
  const size_t num_bytes_;
  const off_t offset_;
  const int fd_;
  __s32 result_;
};
class BatchIOUringAwaiter {
 public:
  BatchIOUringAwaiter(IOUring &ring, std::vector<AlignedRead>& aligned_read_vec,
                 int fd, int coro_id) noexcept
      : ring_(ring),
        aligned_read_vec_(aligned_read_vec),
        fd_(fd) {
          this->coro_batch_read_data_.coro_idx = coro_id;
          this->coro_batch_read_data_.batch_num = aligned_read_vec.size();
        }

  bool await_ready() const noexcept { return false; }

  void await_suspend(cppcoro::coroutine_handle<> handle);

  __s32 await_resume() const noexcept { return result_; }

  void SetResult(__s32 result) noexcept { result_ = result; }

  cppcoro::coroutine_handle<> GetHandle() const noexcept { return handle_; }

 private:
  cppcoro::coroutine_handle<> handle_;
  IOUring &ring_;
  std::vector<AlignedRead> aligned_read_vec_;
  const int fd_;
  __s32 result_;
  CoroBatchReadData coro_batch_read_data_;
};

class IOUring {
 public:
  explicit IOUring(unsigned num_entries) : num_waiting_(0) {
    auto result = io_uring_queue_init(num_entries, &ring_, 0);
    if (result != 0) {
      throw std::system_error{-result, std::generic_category()};
    }
  }

  ~IOUring() { io_uring_queue_exit(&ring_); }

  template <size_t kBatchSize = 16>
  void ProcessBatch() noexcept {
    std::array<io_uring_cqe *, kBatchSize> cqes;
    std::array<cppcoro::coroutine_handle<>, kBatchSize> handles;

    // collect up to kBatchSize handles
    unsigned num_returned =
        io_uring_peek_batch_cqe(&ring_, cqes.data(), kBatchSize);
    for (unsigned i = 0; i != num_returned; ++i) {
      auto *awaiter =
          reinterpret_cast<IOUringAwaiter *>(io_uring_cqe_get_data(cqes[i]));
      awaiter->SetResult(cqes[i]->res);
      io_uring_cqe_seen(&ring_, cqes[i]);
      handles[i] = awaiter->GetHandle();
    }
    num_waiting_ -= num_returned;

    // resume all collected handles
    for (unsigned i = 0; i != num_returned; ++i) {
      handles[i].resume();
    }
  }

  template<size_t kBatchSize = 16>
  void ProcessMyBatch(std::vector<int> &coro_io_counter) noexcept {
    std::array<io_uring_cqe *, kBatchSize>              cqes;
    std::array<cppcoro::coroutine_handle<>, kBatchSize> handles;
    std::vector<int> coro_io_num(coro_io_counter.size(), 0);
    std::map<int, cppcoro::coroutine_handle<>> coro_idx_to_handles;

    // collect up to kBatchSize handles
    unsigned num_returned =
        io_uring_peek_batch_cqe(&ring_, cqes.data(), kBatchSize);
    for (unsigned i = 0; i != num_returned; ++i) {
      auto *coroBatchReadData_tmp =
          reinterpret_cast<CoroBatchReadData *>(io_uring_cqe_get_data(cqes[i]));
      // awaiter->SetResult(cqes[i]->res);
      io_uring_cqe_seen(&ring_, cqes[i]);
      coro_io_counter[coroBatchReadData_tmp->coro_idx]++;
      coro_io_num[coroBatchReadData_tmp->coro_idx] =
          coroBatchReadData_tmp->batch_num;
      handles[i] = coroBatchReadData_tmp->handle;
      if (coro_idx_to_handles.find(coroBatchReadData_tmp->coro_idx) ==
          coro_idx_to_handles.end()) {
        coro_idx_to_handles[coroBatchReadData_tmp->coro_idx] =
            coroBatchReadData_tmp->handle;
      }
    }
    num_waiting_ -= num_returned;

    // resume all collected handles
    for (unsigned i = 0; i != coro_io_counter.size(); ++i) {
      if (coro_io_num[i] > 0 && coro_io_num[i] == coro_io_counter[i]) {
        coro_io_counter[i] = 0;
        coro_idx_to_handles[i].resume();
      }
    }
  }

  bool Empty() const noexcept { return num_waiting_ == 0; }

 private:
  friend class IOUringAwaiter;
  friend class BatchIOUringAwaiter;

  io_uring ring_;
  unsigned num_waiting_;
};

class SubmissionQueueFullError : public std::exception {
  [[nodiscard]] const char *what() const noexcept override {
    return "Submission queue is full";
  }
};

inline void IOUringAwaiter::await_suspend(cppcoro::coroutine_handle<> handle) {
  handle_ = handle;

  io_uring_sqe *sqe = io_uring_get_sqe(&ring_.ring_);
  if (sqe == nullptr) {
    throw SubmissionQueueFullError{};
  }

  io_uring_prep_read(sqe, fd_, buffer_, num_bytes_, offset_);

  io_uring_sqe_set_data(sqe, this);
  io_uring_submit(&ring_.ring_);
  ++ring_.num_waiting_;
}


inline void BatchIOUringAwaiter::await_suspend(cppcoro::coroutine_handle<> handle) {
  // handle_ = handle;
  // std::cout<<"aaa"<<std::endl;
  this->coro_batch_read_data_.handle = handle;
  int cnt = 0;
  // std::cout<<"aligned_read_vec_.size()"<<aligned_read_vec_.size()<<std::endl;
  for(int i = 0;i<this->aligned_read_vec_.size();i++){
    AlignedRead* tmp_ptr = &aligned_read_vec_[i];
    io_uring_sqe *sqe = io_uring_get_sqe(&ring_.ring_);
    if (sqe == nullptr) {
      throw SubmissionQueueFullError{};
    }

    io_uring_prep_read(sqe, fd_, tmp_ptr->buf, tmp_ptr->len, tmp_ptr->offset);

    io_uring_sqe_set_data(sqe, &(this->coro_batch_read_data_));
    cnt++;
    // std::cout<<"bbb"<<std::endl;
  }
  
  io_uring_submit(&ring_.ring_);
  // std::cout<<"ccccc"<<std::endl;
  ring_.num_waiting_ += cnt;
  // std::cout<<"dddddd"<<std::endl;
}

class Countdown {
 public:
  explicit Countdown(std::uint64_t counter) noexcept : counter_(counter) {}

  void Decrement() noexcept { --counter_; }

  bool IsZero() const noexcept { return counter_ == 0; }

  void Set(std::uint64_t counter) noexcept { counter_ = counter; }

 private:
  std::uint64_t counter_;
};

inline cppcoro::task<void> DrainRing(IOUring &ring,
                                     const Countdown &countdown, int coro_size) {
  std::vector<int> coro_io_counter(coro_size,0);
  // coro_io_counter.reserve(coro_size);
  // std::cout<<"BBBBB"<<std::endl;
  while (!countdown.IsZero()) {
    // std::cout<<"CCCCCC"<<std::endl;
    // ring.ProcessBatch();
    ring.ProcessMyBatch(coro_io_counter);
  }
  co_return;
}

}  // namespace storage

#endif  // STORAGE_IO_URING_H_