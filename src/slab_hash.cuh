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
#include <typeinfo>

// global declarations
#include "slab_hash_global.cuh"

// class declaration:
#include "concurrent/context_concurrent.cuh"
#include "phase_concurrent/context_phase_concurrent.cuh"

// warp implementations of member functions:
#include "concurrent/warp/delete.cuh"
#include "concurrent/warp/insert.cuh"
#include "concurrent/warp/search.cuh"

// helper kernels:
#include "concurrent/device/build.cuh"
#include "concurrent/device/delete_kernel.cuh"
#include "concurrent/device/misc_kernels.cuh"
#include "concurrent/device/search_kernel.cuh"

/*
 * This class owns the allocated memory for the hash table
 */
template <typename KeyT,
          typename ValueT,
          uint32_t DEVICE_IDX,
          SlabHashType SlabHashT>
class GpuSlabHash {
 private:
  // fixed known parameters:
  static constexpr uint32_t BLOCKSIZE_ = 128;
  static constexpr uint32_t WARP_WIDTH_ = 32;
  static constexpr uint32_t PRIME_DIVISOR_ = 4294967291u;

  struct hash_function {
    uint32_t x;
    uint32_t y;
  } hf_;

  // total number of buckets (slabs) for this hash table
  uint32_t num_buckets_;

  // a raw pointer to the initial allocated memory for all buckets
  int8_t* d_table_;
  size_t slab_unit_size_;  // size of each slab unit in bytes (might differ
                           // based on the type)

  // slab hash context, contains everything that a GPU application needs to be
  // able to use this data structure
  GpuSlabHashContext<KeyT, ValueT, SlabHashT> gpu_context_;

  // const pointer to an allocator that all instances of slab hash are going to
  // use. The allocator itself is not owned by this class
  DynamicAllocatorT* dynamic_allocator_;

 public:
  GpuSlabHash(const uint32_t num_buckets,
              DynamicAllocatorT* dynamic_allocator,
              const time_t seed = 0)
      : num_buckets_(num_buckets),
        d_table_(nullptr),
        slab_unit_size_(0),
        dynamic_allocator_(dynamic_allocator) {
    assert(dynamic_allocator &&
           "No proper dynamic allocator attached to the slab hash.");
    int32_t devCount = 0;
    CHECK_CUDA_ERROR(cudaGetDeviceCount(&devCount));
    assert(DEVICE_IDX < devCount);

    CHECK_CUDA_ERROR(cudaSetDevice(DEVICE_IDX));

    slab_unit_size_ =
        GpuSlabHashContext<KeyT, ValueT, SlabHashT>::getSlabUnitSize();

    // allocating initial buckets:
    CHECK_CUDA_ERROR(
        cudaMalloc((void**)&d_table_, slab_unit_size_ * num_buckets_));

    CHECK_CUDA_ERROR(
        cudaMemset(d_table_, 0xFF, slab_unit_size_ * num_buckets_));

    // creating a random number generator:
    std::mt19937 rng(seed ? seed : time(0));
    hf_.x = rng() % PRIME_DIVISOR_;
    if (hf_.x < 1)
      hf_.x = 1;
    hf_.y = rng() % PRIME_DIVISOR_;

    // initializing the gpu_context_:
    gpu_context_.initParameters(num_buckets_, hf_.x, hf_.y, d_table_,
                                dynamic_allocator_->getContextPtr());
  }

  ~GpuSlabHash() {
    CHECK_CUDA_ERROR(cudaSetDevice(DEVICE_IDX));
    CHECK_CUDA_ERROR(cudaFree(d_table_));
  }

  // returns some debug information about the slab hash
  std::string to_string();
  double computeLoadFactor(int flag);

  void buildBulk(KeyT* d_key, ValueT* d_value, uint32_t num_keys) {
    const uint32_t num_blocks = (num_keys + BLOCKSIZE_ - 1) / BLOCKSIZE_;
    // calling the kernel for bulk build:
    CHECK_CUDA_ERROR(cudaSetDevice(DEVICE_IDX));
    build_table_kernel<KeyT, ValueT>
        <<<num_blocks, BLOCKSIZE_>>>(d_key, d_value, num_keys, gpu_context_);
  }

  void searchIndividual(KeyT* d_query, ValueT* d_result, uint32_t num_queries) {
    CHECK_CUDA_ERROR(cudaSetDevice(DEVICE_IDX));
    const uint32_t num_blocks = (num_queries + BLOCKSIZE_ - 1) / BLOCKSIZE_;
    search_table<KeyT, ValueT><<<num_blocks, BLOCKSIZE_>>>(
        d_query, d_result, num_queries, gpu_context_);
  }

  void searchBulk(KeyT* d_query, ValueT* d_result, uint32_t num_queries) {
    CHECK_CUDA_ERROR(cudaSetDevice(DEVICE_IDX));
    const uint32_t num_blocks = (num_queries + BLOCKSIZE_ - 1) / BLOCKSIZE_;
    search_table_bulk<KeyT, ValueT><<<num_blocks, BLOCKSIZE_>>>(
        d_query, d_result, num_queries, gpu_context_);
  }

  void deleteIndividual(KeyT* d_key, uint32_t num_keys) {
    CHECK_CUDA_ERROR(cudaSetDevice(DEVICE_IDX));
    const uint32_t num_blocks = (num_keys + BLOCKSIZE_ - 1) / BLOCKSIZE_;
    delete_table_keys<KeyT, ValueT>
        <<<num_blocks, BLOCKSIZE_>>>(d_key, num_keys, gpu_context_);
  }
};

template <typename KeyT,
          typename ValueT,
          uint32_t DEVICE_IDX,
          SlabHashType SlabHashT>
std::string GpuSlabHash<KeyT, ValueT, DEVICE_IDX, SlabHashT>::to_string() {
  std::string result;
  result += " ==== GpuSlabHash: \n";
  result += "\t Running on device \t\t " + std::to_string(DEVICE_IDX) + "\n";
  result +=
      "\t SlabHashType:     \t\t " + gpu_context_.getSlabHashTypeName() + "\n";
  result += "\t Number of buckets:\t\t " + std::to_string(num_buckets_) + "\n";
  result +=
      "\t d_table_ address: \t\t " +
      std::to_string(reinterpret_cast<uint64_t>(static_cast<void*>(d_table_))) +
      "\n";
  result += "\t hash function = \t\t (" + std::to_string(hf_.x) + ", " +
            std::to_string(hf_.y) + ")\n";
  return result;
}

template <typename KeyT,
          typename ValueT,
          uint32_t DEVICE_IDX,
          SlabHashType SlabHashT>
double GpuSlabHash<KeyT, ValueT, DEVICE_IDX, SlabHashT>::computeLoadFactor(
    int flag = 0) {
  uint32_t* h_bucket_count = new uint32_t[num_buckets_];
  uint32_t* d_bucket_count;
  CHECK_CUDA_ERROR(
      cudaMalloc((void**)&d_bucket_count, sizeof(uint32_t) * num_buckets_));
  CHECK_CUDA_ERROR(
      cudaMemset(d_bucket_count, 0, sizeof(uint32_t) * num_buckets_));

  const auto& dynamic_alloc = gpu_context_.getAllocatorContext();
  const uint32_t num_super_blocks = dynamic_alloc.num_super_blocks_;
  uint32_t* h_count_super_blocks = new uint32_t[num_super_blocks];
  uint32_t* d_count_super_blocks;
  CHECK_CUDA_ERROR(cudaMalloc((void**)&d_count_super_blocks,
                              sizeof(uint32_t) * num_super_blocks));
  CHECK_CUDA_ERROR(
      cudaMemset(d_count_super_blocks, 0, sizeof(uint32_t) * num_super_blocks));
  //---------------------------------
  // counting the number of inserted elements:
  const uint32_t blocksize = 128;
  const uint32_t num_blocks = (num_buckets_ * 32 + blocksize - 1) / blocksize;
  bucket_count_kernel<KeyT, ValueT>
      <<<num_blocks, blocksize>>>(gpu_context_, d_bucket_count, num_buckets_);
  CHECK_CUDA_ERROR(cudaMemcpy(h_bucket_count, d_bucket_count,
                              sizeof(uint32_t) * num_buckets_,
                              cudaMemcpyDeviceToHost));

  int total_elements_stored = 0;
  for (int i = 0; i < num_buckets_; i++)
    total_elements_stored += h_bucket_count[i];

  if (flag) {
    printf("## Total elements stored: %d (%lu bytes).\n", total_elements_stored,
           total_elements_stored * (sizeof(KeyT) + sizeof(ValueT)));
  }

  // counting total number of allocated memory units:
  int num_mem_units = dynamic_alloc.NUM_MEM_BLOCKS_PER_SUPER_BLOCK_ * 32;
  int num_cuda_blocks = (num_mem_units + blocksize - 1) / blocksize;
  compute_stats_allocators<<<num_cuda_blocks, blocksize>>>(d_count_super_blocks,
                                                           gpu_context_);

  CHECK_CUDA_ERROR(cudaMemcpy(h_count_super_blocks, d_count_super_blocks,
                              sizeof(uint32_t) * num_super_blocks,
                              cudaMemcpyDeviceToHost));

  // printing stats per super block:
  if (flag == 1) {
    int total_allocated = 0;
    for (int i = 0; i < num_super_blocks; i++) {
      printf(
          "(%d: %d -- %f) \t", i, h_count_super_blocks[i],
          double(h_count_super_blocks[i]) / double(1024 * num_mem_units / 32));
      if (i % 4 == 3)
        printf("\n");
      total_allocated += h_count_super_blocks[i];
    }
    printf("\n");
    printf("Total number of allocated memory units: %d\n", total_allocated);
  }
  // computing load factor
  int total_mem_units = num_buckets_;
  for (int i = 0; i < num_super_blocks; i++)
    total_mem_units += h_count_super_blocks[i];

  double load_factor =
      double(total_elements_stored * (sizeof(KeyT) + sizeof(ValueT))) /
      double(total_mem_units * WARP_WIDTH_ * sizeof(uint32_t));

  if (d_count_super_blocks)
    CHECK_ERROR(cudaFree(d_count_super_blocks));
  if (d_bucket_count)
    CHECK_ERROR(cudaFree(d_bucket_count));
  delete[] h_bucket_count;
  delete[] h_count_super_blocks;

  return load_factor;
}