# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_PREFIX_PATH /home/dfzhang/folly_install/folly)
set(folly_DIR /home/dfzhang/folly_install/folly/lib/cmake/folly)
find_package(folly REQUIRED)
find_package(gflags REQUIRED)
find_package(glog REQUIRED)

# using for spdk
find_package(PkgConfig)
pkg_search_module(SPDK REQUIRED IMPORTED_TARGET GLOBAL spdk_nvme spdk_env_dpdk spdk_syslibs )
set(SPDK_POS "/home/dfzhang/spdk")

set(SPDK_LIB "-L/usr/local/lib -L${SPDK_POS}/dpdk/build/lib -lspdk_nvme -lspdk_keyring -lspdk_sock -lspdk_vmd -lspdk_sock_posix -lspdk_thread -lspdk_trace -lspdk_rpc -lspdk_jsonrpc -lspdk_json -lspdk_dma -lspdk_env_dpdk -lspdk_util -lspdk_log -lrte_eal -lrte_mempool -lrte_ring -lrte_mbuf -lrte_bus_pci -lrte_pci -lrte_mempool_ring -lrte_telemetry -lrte_kvargs -lrte_rcu -lrte_power -lrte_ethdev -lrte_vhost -lrte_net -lrte_dmadev -lrte_cryptodev -lrte_hash -lrte_log -lfuse3 -lssl")

set(DPDK_LIB "-L${SPDK_POS}/dpdk/build/lib \
    -lrte_eal -lrte_eal -lrte_mempool -lrte_ring -lrte_mbuf \ 
    -lrte_bus_pci -lrte_pci -lrte_mempool_ring -lrte_telemetry -lrte_kvargs \ 
    -lrte_power -lrte_ethdev -lrte_net -lrte_vhost -lrte_net -lrte_dmadev -lrte_cryptodev -lrte_hash -lrte_rcu \ "
)

separate_arguments(SPDK_LIB UNIX_COMMAND "${SPDK_LIB}")
separate_arguments(DPDK_LIB UNIX_COMMAND "${DPDK_LIB}")


if(MSVC)
    add_subdirectory(dll)
else()
    #file(GLOB CPP_SOURCES *.cpp)
    set(CPP_SOURCES ann_exception.cpp aux_utils.cpp distance.cpp index.cpp
        linux_aligned_file_reader.cpp math_utils.cpp natural_number_map.cpp
        natural_number_set.cpp memory_mapper.cpp partition_and_pq.cpp
        pq_flash_index.cpp logger.cpp utils.cpp page_search.cpp async_search.cpp visit_freq.cpp
        range_search.cpp spdk_wrapper.cc bqann_queue.cpp)
    add_library(${PROJECT_NAME} ${CPP_SOURCES})
    # add_library(${PROJECT_NAME}_s STATIC ${CPP_SOURCES})
    target_link_options(${PROJECT_NAME} INTERFACE -Wl,--whole-archive -Wl,--no-as-needed)
    target_link_options(${PROJECT_NAME} INTERFACE ${SPDK_LIB} ${DPDK_LIB} )
    target_link_options(${PROJECT_NAME} INTERFACE -Wl,--no-whole-archive)
    target_link_options(${PROJECT_NAME} INTERFACE -lnuma -ldl -L${SPDK_POS}/isa-l/.libs -lisal -pthread -lrt -luuid -lcrypto -lm -laio -ldouble-conversion -lunwind)
    target_link_libraries(${PROJECT_NAME} PRIVATE uring cppcoro)
    target_link_libraries(${PROJECT_NAME} PUBLIC Folly::folly gflags glog)
    # target_link_libraries(${PROJECT_NAME}_s PRIVATE uring cppcoro)

endif()
install()
