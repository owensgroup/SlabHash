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
#include "slab_hash_global.cuh"
#include "concurrent/slab_hash.cuh"
 
template <typename KeyT, typename ValueT>
class gpu_hash_table {
 private:
  uint32_t max_keys_;
  uint32_t num_keys_;
  uint32_t num_buckets_;
  // size_t allocator_heap_size_;

  GpuSlabHash<KeyT, ValueT, SlabHashType::ConcurrentMap>* slab_hash_;

 public:
  // main arrays to hold keys, values, queries, results, etc.
  KeyT* d_key_;
  ValueT* d_value_;
  KeyT* d_query_;
  ValueT* d_result_;

  gpu_hash_table(uint32_t max_keys,
                 uint32_t num_buckets
                 /*uint32_t max_allocator_size*/)
      : max_keys_(max_keys),
        num_buckets_(num_buckets),
        // allocator_heap_size_(max_allocator_size),
        slab_hash_(nullptr) {
    std::cout << "gpu_hash_table constructor called.\n";
    // allocating key, value arrays:
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_key_, sizeof(KeyT) * max_keys_));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_value_, sizeof(ValueT) * max_keys_));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_query_, sizeof(KeyT) * max_keys_));
    CHECK_CUDA_ERROR(
        cudaMalloc((void**)&d_result_, sizeof(ValueT) * max_keys_));

    // slab hash:
    slab_hash_ = new GpuSlabHash<KeyT, ValueT, SlabHashType::ConcurrentMap>(
        num_buckets_ /*, allocator_heap_size_*/);
    std::cout << slab_hash_->to_string() << std::endl;
  }

  ~gpu_hash_table() {
    CHECK_CUDA_ERROR(cudaFree(d_key_));
    CHECK_CUDA_ERROR(cudaFree(d_value_));
    CHECK_CUDA_ERROR(cudaFree(d_query_));
    CHECK_CUDA_ERROR(cudaFree(d_result_));

    // slab hash:
    delete (slab_hash_);
  }

  float hash_build(KeyT* h_key, ValueT* h_value, uint32_t num_keys) {
    // moving key-values to the device:
    CHECK_CUDA_ERROR(cudaMemcpy(d_key_, h_key, sizeof(KeyT) * num_keys,
                                cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_value_, h_value, sizeof(ValueT) * num_keys,
                                cudaMemcpyHostToDevice));

    num_keys_ = num_keys;

    float temp_time = 0.0f;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    // calling slab-hash's bulk build procedure:
    slab_hash_->bulk_build(d_key_, d_value_, num_keys);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&temp_time, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return temp_time;
  }
};