#pragma once
#include "spdk/stdinc.h"

#include "spdk/nvme.h"
#include "spdk/vmd.h"
#include "spdk/nvme_zns.h"
#include "spdk/env.h"
#include "spdk/string.h"
#include "spdk/log.h"

#include "aligned_file_reader.h"

#include <string>
#include <unordered_map>
#include <memory>

#include <glog/logging.h>

#define SPDK_SECTOR_LEN 4096
#define LBA_SIZE 4096

namespace ssdps {

class SpdkWrapper {
 public:
  static std::shared_ptr<SpdkWrapper> create(int queue_cnt);
  virtual void Init() = 0;

  virtual void SubmitReadCommand(void *pinned_dst, const int64_t bytes, const int64_t lba,
                                 spdk_nvme_cmd_cb func, void *ctx, int qp_id) = 0;

  virtual int SubmitWriteCommand(const void *pinned_src, const int64_t bytes,
                                 const int64_t lba, spdk_nvme_cmd_cb func, void *ctx, int qp_id) = 0;

  virtual void SyncRead(void *pinned_dst, const int64_t bytes, const int64_t lba, int qp_id) = 0;

  virtual void BatchSyncRead(std::vector<AlignedRead> &read_vec, int qp_id) = 0;

  virtual void SyncWrite(const void *pinned_src, const int64_t bytes,
                         const int64_t lba, int qp_id) = 0;

  virtual void SyncRead4K(void *pinned_dst, const int64_t bytes,
                        const int64_t lba_4k, int qp_id) = 0;
  virtual void SubmitRead4K(AlignedRead &read, spdk_nvme_cmd_cb func, void *ctx,
                            int qp_id) = 0;

  virtual void BatchSyncRead4K(std::vector<AlignedRead> &read_vec, int qp_id) = 0;

  virtual void SyncWrite4K(const void *pinned_src, const int64_t bytes,
                         const int64_t lba_4k, int qp_id) = 0;

  virtual void Sync2Read(void *pinned_dst, const int64_t lba, int qp_id) = 0;

  virtual void PollCompleteQueue(int qp_id) = 0;
  virtual int32_t myPollCompleteQueue(int qp_id) = 0;
  virtual int GetLBASize() const = 0;
  virtual uint64_t GetLBANumber() const = 0;
  virtual ~SpdkWrapper() {}
};

}  // namespace ssdps
