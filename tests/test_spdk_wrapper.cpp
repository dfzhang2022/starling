
#include <atomic>
#include <chrono>
#include <iostream>
#include <thread>
#include <chrono>

#include "spdk_wrapper.h"

#define IO_NUM 100000
#define BEAM_WIDTH 8

class Timer {
  typedef std::chrono::high_resolution_clock _clock;
  std::chrono::time_point<_clock>            check_point;

 public:
  Timer() : check_point(_clock::now()) {
  }

  void reset() {
    check_point = _clock::now();
  }

  long long elapsed() const {
    return std::chrono::duration_cast<std::chrono::microseconds>(
               _clock::now() - check_point)
        .count();
  }
};


// TEST(SpdkWrapper, test_fused_operation) {
//   auto ssd = ssdps::SpdkWrapper::create();
//   ssd->Init();
//   char *buf = (char *)spdk_zmalloc(0x1000, 0x1000, NULL, SPDK_ENV_SOCKET_ID_ANY,
//                                    SPDK_MALLOC_DMA);
//   char *buf_2 = (char *)spdk_zmalloc(0x1000, 0x1000, NULL,
//                                      SPDK_ENV_SOCKET_ID_ANY, SPDK_MALLOC_DMA);
//   snprintf(buf, 0x1000, "%s", "Hello world!\n");

//   ssd->SyncWrite((void *)buf, 512, 0);
//   ssd->SyncWrite((void *)buf, 512, 1);
//   // ssd->Sync2Read((void *)buf_2, 0);
//   std::this_thread::sleep_for(std::chrono::seconds(2));
//   CHECK(memcmp(buf, buf_2, 512) == 0);
//   CHECK(memcmp(buf, buf_2 + 512, 512) == 0);
// }

int main(int argc, char **argv) {
  auto ssd = ssdps::SpdkWrapper::create(1);
  ssd->Init();

  char *buf = (char *)spdk_zmalloc(0x1000 * BEAM_WIDTH, 0x1000, NULL, SPDK_ENV_SOCKET_ID_ANY,
                                   SPDK_MALLOC_DMA);
  char *buf_2 = (char *)spdk_zmalloc(0x1000 * BEAM_WIDTH, 0x1000, NULL,
                                     SPDK_ENV_SOCKET_ID_ANY, SPDK_MALLOC_DMA);
  snprintf(buf, 0x1000, "%s", "Hello world!\n");
  std::cout <<"iops: "<<std::endl;

  ssd->SyncWrite((void *)buf, 0x1000, 0, 0);

  Timer t;
  int k = 0;
  while(k<IO_NUM){
    ssd->SyncRead((void *)buf_2, 0x1000 * BEAM_WIDTH, 7*k, 0);
    k+=BEAM_WIDTH;
  }
  
  double diff = t.elapsed();

  float iops = IO_NUM/diff * 1000*1000;

  // assert(strcmp(buf, buf_2) == 0);

  std::cout <<"iops: "<< iops<<std::endl;
}