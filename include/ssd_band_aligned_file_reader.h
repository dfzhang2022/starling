#pragma once

#include "aligned_file_reader.h"

class SSDBandAlignedFileReader : public AlignedFileReader {
 private:
  std::vector<uint64_t>   file_sz_vec;
  std::vector<FileHandle> file_desc_vec;

  size_t fd_num = 0;

  io_context_t bad_ctx = (io_context_t) -1;

 public:
  SSDBandAlignedFileReader();
  ~SSDBandAlignedFileReader();

  IOContext &get_ctx();

  void register_thread();
  void deregister_thread();
  void deregister_all_threads();

  // Open & close ops
  // Blocking calls
  void open(const std::string &fname);
  void open_multi_ssd(std::vector<std::string> &fname_vec);
  void close();

  // process batch of aligned requests in parallel
  // NOTE :: blocking call
  void read(std::vector<AlignedRead> &read_reqs, IOContext &ctx,
            bool async = false);
  void read(std::vector<AlignedRead> &read_reqs, IOContext &ctx,
            bool async = false, int ssd_id);

  int  submit_reqs(std::vector<AlignedRead> &read_reqs, IOContext &ctx);
  int  submit_reqs(std::vector<AlignedRead> &read_reqs, IOContext &ctx,
                   int ssd_id);
  void get_events(IOContext &ctx, int n_ops);
}