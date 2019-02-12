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

#include <cassert>
#include <iostream>
#include <random>

#include "concurrent/device/build.cuh"
#include "slab_hash_global.cuh"

template <typename KeyT, typename ValueT, uint32_t DEVICE_IDX>
class GpuSlabHash<KeyT, ValueT, DEVICE_IDX, SlabHashType::ConcurrentMap> {
 private:
  // fixed known parameters:
  static constexpr uint32_t BLOCKSIZE_ = 128;
  static constexpr uint32_t WARP_WIDTH_ = 32;
  static constexpr uint32_t PRIME_DIVISOR_ = 4294967291u;

  // the used hash function:
  uint32_t num_buckets_;
  // size_t 		allocator_heap_size_;

  struct hash_function {
    uint32_t x;
    uint32_t y;
  } hf_;

  // bucket data structures:
  concurrent_slab<KeyT, ValueT>* d_table_;
  // dynamic memory allocator:
  // slab_alloc::context_alloc<1> ctx_alloc_;
 public:
  GpuSlabHash(const uint32_t num_buckets, const time_t seed = 0)
      : num_buckets_(num_buckets), d_table_(nullptr) {
    std::cout << " == slab hash concstructor called" << std::endl;

    // a single slab on a ConcurrentMap should be 128 bytes
    assert(sizeof(concurrent_slab<KeyT, ValueT>) ==
           (WARP_WIDTH_ * sizeof(uint32_t)));

    int32_t devCount = 0;
    CHECK_CUDA_ERROR(cudaGetDeviceCount(&devCount));
    assert(DEVICE_IDX < devCount);

    CHECK_CUDA_ERROR(cudaSetDevice(DEVICE_IDX));

    // allocating initial buckets:
    CHECK_CUDA_ERROR(
        cudaMalloc((void**)&d_table_,
                   sizeof(concurrent_slab<KeyT, ValueT>) * num_buckets_));

    CHECK_CUDA_ERROR(cudaMemset(
        d_table_, 0xFF, sizeof(concurrent_slab<KeyT, ValueT>) * num_buckets_));

    // creating a random number generator:
    std::mt19937 rng(seed ? seed : time(0));
    hf_.x = rng() % PRIME_DIVISOR_;
    if (hf_.x < 1)
      hf_.x = 1;
    hf_.y = rng() % PRIME_DIVISOR_;
  }

  ~GpuSlabHash() {
    CHECK_CUDA_ERROR(cudaSetDevice(DEVICE_IDX));
    CHECK_CUDA_ERROR(cudaFree(d_table_));
  }

  // returns some debug information about the slab hash
  std::string to_string();

  __device__ __host__ __forceinline__ uint32_t
  computeBucket(const KeyT& key) const {
    return (((hf_.x ^ key) + hf_.y) % PRIME_DIVISOR_) % num_buckets_;
  }

  __device__ __host__ __forceinline__ concurrent_slab<KeyT, ValueT>*
  getDeviceTablePointer() {
    return d_table_;
  }

  void bulk_build(KeyT* d_key, ValueT* d_value, uint32_t num_keys) {
    uint32_t num_blocks = (num_keys + BLOCKSIZE_ - 1) / BLOCKSIZE_;
    // calling the kernel for bulk build:
    // build_table_kernel<KeyT, ValueT><<<num_blocks, BLOCKSIZE_>>>(
    //     d_key, d_value, num_keys, d_table_, num_buckets_, ctx_alloc_, hf_);
    build_table_kernel<KeyT, ValueT>
        <<<num_blocks, BLOCKSIZE_>>>(d_key, d_value, num_keys, *this);
  }
};

template <typename KeyT, typename ValueT, uint32_t DEVICE_IDX>
std::string GpuSlabHash<KeyT, ValueT, DEVICE_IDX, SlabHashType::ConcurrentMap>::
    to_string() {
  std::string result;
  result += " ==== GpuSlabHash: \n";
  result += "\t Running on device \t\t " + std::to_string(DEVICE_IDX) + "\n";
  result += "\t SlabHashType:     \t\t ConcurrentMap\n";
  result += "\t Number of buckets:\t\t " + std::to_string(num_buckets_) + "\n";
  result +=
      "\t d_table_ address: \t\t " +
      std::to_string(reinterpret_cast<uint64_t>(static_cast<void*>(d_table_))) +
      "\n";
  result += "\t hash function = \t\t (" + std::to_string(hf_.x) + ", " +
            std::to_string(hf_.y) + ")\n";
  return result;
}