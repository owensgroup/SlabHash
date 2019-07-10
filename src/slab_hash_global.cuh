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

#define CHECK_CUDA_ERROR(call)                                                          \
  do {                                                                                  \
    cudaError_t err = call;                                                             \
    if (err != cudaSuccess) {                                                           \
      printf("CUDA error at %s %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
      exit(EXIT_FAILURE);                                                               \
    }                                                                                   \
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

// this slab structure is meant to be used in either concurrent sets,
// or phase-concurrent maps.
// | key 0 | key 1 | key 2 | ... | key 30 | next_ptr |
template <typename KeyT>
struct __align__(32) key_only_slab {
  static constexpr uint32_t NUM_ELEMENTS_PER_SLAB = 31u;
  KeyT keys[NUM_ELEMENTS_PER_SLAB];
  uint32_t next_ptr_index[1];
};

template <typename KeyT, typename ValueT>
struct __align__(32) phase_concurrent_slab {
  static constexpr uint32_t NUM_ELEMENTS_PER_SLAB = 31u;
  // main slab (128 bytes), contain keys
  key_only_slab<KeyT> keys;

  // value storage:
  ValueT values[NUM_ELEMENTS_PER_SLAB];
};

/*
 * Different types of slab hash:
 * 1. Concurrent map: it assumes that all operations can be performed
 * concurrently
 * 2. phase-concurrent map: supports concurrent updates, and concurrent
 * searches, but not a mixture of both
 */
enum class SlabHashTypeT { ConcurrentMap, ConcurrentSet, PhaseConcurrentMap };

template <typename KeyT, typename ValueT>
class ConcurrentMapT {
 public:
  // fixed parameters for the data structure
  static constexpr uint32_t A_INDEX_POINTER = 0xFFFFFFFE;
  static constexpr uint32_t EMPTY_INDEX_POINTER = 0xFFFFFFFF;
  static constexpr uint32_t BASE_UNIT_SIZE = 32;
  static constexpr uint32_t REGULAR_NODE_ADDRESS_MASK = 0x30000000;
  static constexpr uint32_t REGULAR_NODE_DATA_MASK = 0x3FFFFFFF;
  static constexpr uint32_t REGULAR_NODE_KEY_MASK = 0x15555555;

  using SlabTypeT = concurrent_slab<KeyT, ValueT>;

  static std::string getTypeName() { return std::string("ConcurrentMap"); }
};

template <typename KeyT>
class ConcurrentSetT {
 public:
  // fixed parameters for the data structure
  static constexpr uint32_t A_INDEX_POINTER = 0xFFFFFFFE;
  static constexpr uint32_t EMPTY_INDEX_POINTER = 0xFFFFFFFF;
  static constexpr uint32_t BASE_UNIT_SIZE = 32;
  static constexpr uint32_t REGULAR_NODE_ADDRESS_MASK = 0x80000000;
  static constexpr uint32_t REGULAR_NODE_DATA_MASK = 0x7FFFFFFF;
  static constexpr uint32_t REGULAR_NODE_KEY_MASK = 0x7FFFFFFF;
  static constexpr uint32_t NEXT_PTR_LANE = 31u;

  using SlabTypeT = key_only_slab<KeyT>;

  static std::string getTypeName() { return std::string("ConcurrentSet"); }
};

template <typename KeyT, typename ValueT>
class PhaseConcurrentMapT {
 public:
  using SlabTypeT = phase_concurrent_slab<KeyT, ValueT>;
  static std::string getTypeName() { return std::string("PhaseConcurrentMap"); }
};

// the main class to be specialized for different types of hash tables
template <typename KeyT, typename ValueT, SlabHashTypeT SlabHashT>
class GpuSlabHash;

template <typename KeyT, typename ValueT, SlabHashTypeT SlabHashT>
class GpuSlabHashContext;

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

using AllocatorContextT = SlabAllocLightContext<slab_alloc_par::log_num_mem_blocks,
                                                slab_alloc_par::num_super_blocks,
                                                slab_alloc_par::num_replicas>;

using SlabAddressT = uint32_t;
using BucketAddressT = SlabAddressT;