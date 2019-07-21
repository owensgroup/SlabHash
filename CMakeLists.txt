cmake_minimum_required (VERSION 3.8 FATAL_ERROR)
project (SlabHash)

find_package(CUDA 8.0 REQUIRED)

option(CMAKE_VERBOSE_MAKEFILE ON)
option(DGTEST,  "DGTEST"  ON)

set(CUDA_NVCC_FLAGS -std=c++11)
set (CMAKE_CXX_STANDARD 11)

if (CUDA_VERBOSE_PTXAS)
  set(VERBOSE_PTXAS --ptxas-options=-v)
endif (CUDA_VERBOSE_PTXAS)

set(CMAKE_BUILD_TYPE "Release")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(GENCODE_SM30
  -gencode=arch=compute_30,code=sm_30 -gencode=arch=compute_30,code=compute_30)
set(GENCODE_SM35
  -gencode=arch=compute_35,code=sm_35 -gencode=arch=compute_35,code=compute_35)
set(GENCODE_SM37
  -gencode=arch=compute_37,code=sm_37 -gencode=arch=compute_37,code=compute_37)
set(GENCODE_SM50
  -gencode=arch=compute_50,code=sm_50 -gencode=arch=compute_50,code=compute_50)
set(GENCODE_SM60
  -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_60,code=compute_60)
set(GENCODE_SM61
  -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_61,code=compute_61)
set(GENCODE_SM70
  -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_70,code=compute_70)
set(GENCODE_SM71
  -gencode=arch=compute_71,code=sm_71 -gencode=arch=compute_71,code=compute_71)
set(GENCODE_SM75
-gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_75,code=compute_75)

option(SLABHASH_GENCODE_SM30 "GENCODE_SM30" OFF)
option(SLABHASH_GENCODE_SM35 "GENCODE_SM35" ON)
option(SLABHASH_GENCODE_SM37 "GENCODE_SM37" OFF)
option(SLABHASH_GENCODE_SM50 "GENCODE_SM50" OFF)
option(SLABHASH_GENCODE_SM60 "GENCODE_SM60" OFF)
option(SLABHASH_GENCODE_SM61 "GENCODE_SM61" OFF)
option(SLABHASH_GENCODE_SM70 "GENCODE_SM70" OFF)
option(SLABHASH_GENCODE_SM71 "GENCODE_SM71" OFF)
option(SLABHASH_GENCODE_SM75 "GENCODE_SM75" OFF)

if (SLABHASH_GENCODE_SM30)
  set(GENCODE ${GENCODE} ${GENCODE_SM30})
endif(SLABHASH_GENCODE_SM30)

if (SLABHASH_GENCODE_SM35)
  set(GENCODE ${GENCODE} ${GENCODE_SM35})
endif(SLABHASH_GENCODE_SM35)

if (SLABHASH_GENCODE_SM37)
  set(GENCODE ${GENCODE} ${GENCODE_SM37})
endif(SLABHASH_GENCODE_SM37)

if (SLABHASH_GENCODE_SM50)
  set(GENCODE ${GENCODE} ${GENCODE_SM50})
endif(SLABHASH_GENCODE_SM50)

if (SLABHASH_GENCODE_SM60)
  set(GENCODE ${GENCODE} ${GENCODE_SM60})
endif(SLABHASH_GENCODE_SM60)

if (SLABHASH_GENCODE_SM61)
  set(GENCODE ${GENCODE} ${GENCODE_SM61})
endif(SLABHASH_GENCODE_SM61)

if (SLABHASH_GENCODE_SM70)
  set(GENCODE ${GENCODE} ${GENCODE_SM70})
endif(SLABHASH_GENCODE_SM70)

if(SLABHASH_GENCODE_SM71)
  set(GENCODE ${GENCODE} ${GENCODE_SM71})
endif(SLABHASH_GENCODE_SM71)

if(SLABHASH_GENCODE_SM75)
  set(GENCODE ${GENCODE} ${GENCODE_SM75})
endif(SLABHASH_GENCODE_SM75)

include_directories(SlabAlloc/src)
include_directories(src src/concurrent)
include_directories(ThirdParty/rapidjson/include)
include_directories(ThirdParty/googletest/googletest)
include_directories(ThirdParty/googletest/googletest/include)
add_subdirectory(ThirdParty/googletest/googletest)
add_subdirectory(test)
add_subdirectory(bench)

if (DGTEST)
  enable_testing()
endif()