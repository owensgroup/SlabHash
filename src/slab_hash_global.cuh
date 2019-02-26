/*
 * Copyright 2019 Saman Ashkiani
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied. See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "slab_alloc.cuh"

#define CHECK_CUDA_ERROR(call)                                \
  do {                                                        \
    cudaError_t err = call;                                   \
    if (err != cudaSuccess) {                                 \
      printf("CUDA error at %s %d: %s\n", __FILE__, __LINE__, \
             cudaGetErrorString(err));                        \
      exit(EXIT_FAILURE);                                     \
    }                                                         \
  } while (0)

// internal parameters for slab hash device functions:
static constexpr uint32_t EMPTY_KEY = 0xFFFFFFFF;
static constexpr uint32_t EMPTY_VALUE = 0xFFFFFFFF;
static constexpr uint64_t EMPTY_PAIR_64 = 0xFFFFFFFFFFFFFFFFLL;
static constexpr uint32_t WARP_WIDTH = 32;
static constexpr uint32_t SEARCH_NOT_FOUND = 0xFFFFFFFF;

// only works with up to 32-bit key/values
template <typename KeyT, typename ValueT>
struct key_value_pair {
  KeyT key;
  ValueT value;
};

template <typename KeyT, typename ValueT>
struct __align__(32) concurrent_slab {
  static constexpr uint32_t NUM_ELEMENTS_PER_SLAB = 15u;
  key_value_pair<KeyT, ValueT> data[NUM_ELEMENTS_PER_SLAB];
  uint32_t ptr_index[2];
};

/*
 * Different types of slab hash:
 * 1. Concurrent map: it assumes that all operations can be performed
 * concurrently
 * 2. Phase-concurrent map: it assumes updates and searches are done in
 * different phases
 */
enum class SlabHashType { ConcurrentMap, PhaseConcurrentMap };

// template <typename KeyT,
//           typename ValueT,
//           uint32_t DEVICE_IDX,
//           SlabHashType SlabHashT>
// class GpuSlabHash;

template <typename KeyT, typename ValueT, SlabHashType SlabHashT>
struct GpuSlabHashContext;

// The custom allocator that is being used for this code:
// this might need to be a template paramater itself
namespace slab_alloc_par {
constexpr uint32_t log_num_mem_blocks = 8;
constexpr uint32_t num_super_blocks = 32;
constexpr uint32_t num_replicas = 1;
}  // namespace slab_alloc_par

using DynamicAllocatorT = SlabAllocLight<slab_alloc_par::log_num_mem_blocks,
                                         slab_alloc_par::num_super_blocks,
                                         slab_alloc_par::num_replicas>;

using AllocatorContextT =
    SlabAllocLightContext<slab_alloc_par::log_num_mem_blocks,
                          slab_alloc_par::num_super_blocks,
                          slab_alloc_par::num_replicas>;

using SlabAddressT = uint32_t;
using BucketAddressT = SlabAddressT;