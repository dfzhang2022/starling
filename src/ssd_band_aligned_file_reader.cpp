#include "ssd_band_aligned_file_reader.h"

#include <cassert>
#include <cstdio>
#include <iostream>
#include "tsl/robin_map.h"
#include "utils.h"
#define SSD_BAND_MAX_EVENTS 128

namespace {
  typedef struct io_event io_event_t;
  typedef struct iocb     iocb_t;
}

SSDBandAlignedFileReader::SSDBandAlignedFileReader() {
  this->file_desc_vec.clear();
  this->file_sz_vec.clear();
}

SSDBandAlignedFileReader::~SSDBandAlignedFileReader() {
  int ret;
  // check to make sure file_desc is closed

  for (auto &fd : this->file_desc_vec) {
    ret = ::fcntl(fd, F_GETFD);
    if (ret == -1) {
      if (errno != EBADF) {
        std::cerr << "close() not called" << std::endl;
        // close file desc
        ret = ::close(fd);
        // error checks
        if (ret == -1) {
          std::cerr << "close() failed; returned " << ret << ", errno=" << errno
                    << ":" << ::strerror(errno) << std::endl;
        }
      }
    }
  }
}

io_context_t &SSDBandAlignedFileReader::get_ctx() {
  std::unique_lock<std::mutex> lk(ctx_mut);
  // perform checks only in DEBUG mode
  if (ctx_map.find(std::this_thread::get_id()) == ctx_map.end()) {
    std::cerr << "bad thread access; returning -1 as io_context_t" << std::endl;
    return this->bad_ctx;
  } else {
    return ctx_map[std::this_thread::get_id()];
  }
}

void SSDBandAlignedFileReader::register_thread() {
  auto                         my_id = std::this_thread::get_id();
  std::unique_lock<std::mutex> lk(ctx_mut);
  if (ctx_map.find(my_id) != ctx_map.end()) {
    std::cerr << "multiple calls to register_thread from the same thread"
              << std::endl;
    return;
  }
  io_context_t ctx = 0;
  int          ret = io_setup(SSD_BAND_MAX_EVENTS, &ctx);
  if (ret != 0) {
    lk.unlock();
    assert(errno != EAGAIN);
    assert(errno != ENOMEM);
    LOG(ERROR) << "io_setup() failed; returned " << ret << ", errno=" << errno
               << ":" << ::strerror(errno);
  } else {
    diskann::cout << "allocating ctx: " << ctx << " to thread-id:" << my_id
                  << std::endl;
    ctx_map[my_id] = ctx;
  }
}

void SSDBandAlignedFileReader::deregister_thread() {
  auto                         my_id = std::this_thread::get_id();
  std::unique_lock<std::mutex> lk(ctx_mut);
  assert(ctx_map.find(my_id) != ctx_map.end());

  lk.unlock();
  io_context_t ctx = this->get_ctx();
  io_destroy(ctx);
  //  assert(ret == 0);
  lk.lock();
  ctx_map.erase(my_id);
  std::cerr << "returned ctx from thread-id:" << my_id << std::endl;
  lk.unlock();
}

void SSDBandAlignedFileReader::deregister_all_threads() {
  std::unique_lock<std::mutex> lk(ctx_mut);
  for (auto x = ctx_map.begin(); x != ctx_map.end(); x++) {
    io_context_t ctx = x.value();
    io_destroy(ctx);
  }
  ctx_map.clear();
}

void SSDBandAlignedFileReader::open(const std::string &fname) {
  LOG(ERROR) << "open() not supported for single SSD";
}

void SSDBandAlignedFileReader::open_multi_ssd(
    std::vector<std::string> &fname_vec) {
  int ret;
  int flags = O_DIRECT | O_RDONLY | O_LARGEFILE;
  for (auto &fname : fname_vec) {
    FileHandle fd = ::open(fname.c_str(), flags);
    if (fd == -1) {
      std::cerr << "open() failed; returned " << fd << ", errno=" << errno
                << ":" << ::strerror(errno) << std::endl;
      exit(-1);
    }
    this->file_desc_vec.push_back(fd);
    struct stat st;
    ret = ::stat(fname.c_str(), &st);
    if (ret == -1) {
      std::cerr << "stat() failed; returned " << ret << ", errno=" << errno
                << ":" << ::strerror(errno) << std::endl;
      exit(-1);
    }
    this->file_sz_vec.push_back(st.st_size);
  }

  fd_num = this->file_desc_vec.size();
}

void SSDBandAlignedFileReader::close() {
  int ret;
  for (auto &fd : this->file_desc_vec) {
    ret = ::fcntl(fd, F_GETFD);
    assert(ret != -1);
    ret = ::close(fd);
    assert(ret != -1);
    if (ret == -1) {
      std::cerr << "close() failed; returned " << ret << ", errno=" << errno
                << ":" << ::strerror(errno) << std::endl;
    }
  }
}
void SSDBandAlignedFileReader::read(std::vector<> &read_reqs, io_context_t &ctx,
                                    bool async) {
  return read(read_reqs, ctx, async.0);
}

void SSDBandAlignedFileReader::read(std::vector<> &read_reqs, io_context_t &ctx,
                                    bool async, int ssd_id) {
  if (async == true) {
    diskann::cout << "Async currently not supported in linux." << std::endl;
  }
  assert(this->file_desc_vec[ssd_id] != -1);

  int n_ops = submit_reqs(read_reqs, ctx, ssd_id);
  get_events(ctx, n_ops);
  return;
}

int SSDBandAlignedFileReader::submit_reqs(std::vector<> &read_reqs,
                                          io_context_t  &ctx) {
  CHECK_GT(this->file_desc_vec.size(), 0);
  return submit_reqs(read_reqs, ctx, 0);
}

int SSDBandAlignedFileReader::submit_reqs(std::vector<AlignedRead> &read_reqs,
                                          IOContext &ctx, int ssd_id) {
  assert(this->file_desc_vec[ssd_id] != -1);

  if (read_reqs.size() > SSD_BAND_MAX_EVENTS) {
    std::cerr << "The number of requests should not exceed " << MAX_EVENTS
              << std::endl;
    exit(-1);
  }
  int                      n_ops = read_reqs.size();
  std::vector<iocb_t *>    cbs(n_ops, nullptr);
  std::vector<io_event_t>  evts(n_ops);
  std::vector<struct iocb> cb(n_ops);
  for (int j = 0; j < n_ops; j++) {
    io_prep_pread(cb.data() + j, this->file_desc_vec[ssd_id], read_reqs[j].buf,
                  read_reqs[j].len, read_reqs[j].offset);
  }
  for (int i = 0; i < n_ops; i++) {
    cbs[i] = cb.data() + i;
  }

  int ret = io_submit(ctx, (int64_t) n_ops, cbs.data());
  if (ret != n_ops) {
    std::cerr << "io_submit() failed; returned " << ret
              << ", expected=" << n_ops << ", ernno=" << errno << "="
              << ::strerror(-ret);
    std::cout << "ctx: " << ctx << "\n";
    exit(-1);
  }
  return n_ops;
}
void get_events(IOContext &ctx, int n_ops) {
  std::vector<io_event_t> evts(n_ops);
  auto                    ret =
      io_getevents(ctx, (int64_t) n_ops, (int64_t) n_ops, evts.data(), nullptr);
  if (ret != (int64_t) n_ops) {
    std::cerr << "io_getevents() failed; returned " << ret << "." << std::endl;
    exit(-1);
  }
}
}
