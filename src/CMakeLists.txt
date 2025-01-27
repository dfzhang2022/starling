# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

set(CMAKE_CXX_STANDARD 20)
# add_subdirectory(${CMAKE_SOURCE_DIR}/src/storage)
# include_directories(/home/dfzhang/starling/src/storage/src/storage)
if(MSVC)
    add_subdirectory(dll)
else()
    #file(GLOB CPP_SOURCES *.cpp)
    set(CPP_SOURCES ann_exception.cpp aux_utils.cpp distance.cpp index.cpp
        linux_aligned_file_reader.cpp math_utils.cpp natural_number_map.cpp
        natural_number_set.cpp memory_mapper.cpp partition_and_pq.cpp
        pq_flash_index.cpp logger.cpp utils.cpp page_search.cpp async_search.cpp visit_freq.cpp
        range_search.cpp)
    add_library(${PROJECT_NAME} ${CPP_SOURCES})
    add_library(${PROJECT_NAME}_s STATIC ${CPP_SOURCES})
    target_link_libraries(${PROJECT_NAME} PRIVATE uring cppcoro)
    target_link_libraries(${PROJECT_NAME}_s PRIVATE uring cppcoro)
endif()
install()
