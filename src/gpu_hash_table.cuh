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
#include "slab_hash.cuh"

template <typename KeyT, typename ValueT, uint32_t DEVICE_IDX>
class gpu_hash_table {
 private:
  uint32_t max_keys_;
  uint32_t num_buckets_;
  int64_t seed_;

  GpuSlabHash<KeyT, ValueT, DEVICE_IDX, ConcurrentMap<KeyT, ValueT>>*
      slab_hash_;

  // the dynamic allocator that is being used for slab hash
  DynamicAllocatorT* dynamic_allocator_;

 public:
  // main arrays to hold keys, values, queries, results, etc.
  KeyT* d_key_;
  ValueT* d_value_;
  KeyT* d_query_;
  ValueT* d_result_;

  gpu_hash_table(uint32_t max_keys, uint32_t num_buckets, const int64_t seed)
      : max_keys_(max_keys),
        num_buckets_(num_buckets),
        seed_(seed),
        slab_hash_(nullptr),
        dynamic_allocator_(nullptr) {
    int32_t devCount = 0;
    CHECK_CUDA_ERROR(cudaGetDeviceCount(&devCount));
    assert(DEVICE_IDX < devCount);

    CHECK_CUDA_ERROR(cudaSetDevice(DEVICE_IDX));

    // allocating key, value arrays:
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_key_, sizeof(KeyT) * max_keys_));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_value_, sizeof(ValueT) * max_keys_));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_query_, sizeof(KeyT) * max_keys_));
    CHECK_CUDA_ERROR(
        cudaMalloc((void**)&d_result_, sizeof(ValueT) * max_keys_));

    // allocate an initialize the allocator:
    dynamic_allocator_ = new DynamicAllocatorT();

    // slab hash:
    slab_hash_ =
        new GpuSlabHash<KeyT, ValueT, DEVICE_IDX, ConcurrentMap<KeyT, ValueT>>(
            num_buckets_, dynamic_allocator_, seed_);
    std::cout << slab_hash_->to_string() << std::endl;
  }

  ~gpu_hash_table() {
    CHECK_CUDA_ERROR(cudaSetDevice(DEVICE_IDX));
    CHECK_CUDA_ERROR(cudaFree(d_key_));
    CHECK_CUDA_ERROR(cudaFree(d_value_));
    CHECK_CUDA_ERROR(cudaFree(d_query_));
    CHECK_CUDA_ERROR(cudaFree(d_result_));

    // delete the dynamic allocator:
    delete dynamic_allocator_;

    // slab hash:
    delete (slab_hash_);
  }

  float hash_build(KeyT* h_key, ValueT* h_value, uint32_t num_keys) {
    // moving key-values to the device:
    CHECK_CUDA_ERROR(cudaSetDevice(DEVICE_IDX));
    CHECK_CUDA_ERROR(cudaMemcpy(d_key_, h_key, sizeof(KeyT) * num_keys,
                                cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_value_, h_value, sizeof(ValueT) * num_keys,
                                cudaMemcpyHostToDevice));

    float temp_time = 0.0f;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    // calling slab-hash's bulk build procedure:
    slab_hash_->buildBulk(d_key_, d_value_, num_keys);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&temp_time, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return temp_time;
  }

  float hash_search(KeyT* h_query, ValueT* h_result, uint32_t num_queries) {
    CHECK_CUDA_ERROR(cudaSetDevice(DEVICE_IDX));
    CHECK_CUDA_ERROR(cudaMemcpy(d_query_, h_query, sizeof(KeyT) * num_queries,
                                cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemset(d_result_, 0xFF, sizeof(ValueT) * num_queries));

    float temp_time = 0.0f;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // == calling slab hash's individual search:
    slab_hash_->searchIndividual(d_query_, d_result_, num_queries);
    //==

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&temp_time, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    CHECK_CUDA_ERROR(cudaMemcpy(h_result, d_result_,
                                sizeof(ValueT) * num_queries,
                                cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();
    return temp_time;
  }
  float hash_search_bulk(KeyT* h_query,
                         ValueT* h_result,
                         uint32_t num_queries) {
    CHECK_CUDA_ERROR(cudaSetDevice(DEVICE_IDX));
    CHECK_CUDA_ERROR(cudaMemcpy(d_query_, h_query, sizeof(KeyT) * num_queries,
                                cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemset(d_result_, 0xFF, sizeof(ValueT) * num_queries));

    float temp_time = 0.0f;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    //== slab hash's bulk search:
    slab_hash_->searchBulk(d_query_, d_result_, num_queries);
    //==

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&temp_time, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    CHECK_CUDA_ERROR(cudaMemcpy(h_result, d_result_,
                                sizeof(ValueT) * num_queries,
                                cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();
    return temp_time;
  }

  float hash_delete(KeyT* h_key, uint32_t num_keys) {
    CHECK_CUDA_ERROR(cudaSetDevice(DEVICE_IDX));
    CHECK_CUDA_ERROR(cudaMemcpy(d_key_, h_key, sizeof(KeyT) * num_keys,
                                cudaMemcpyHostToDevice));

    float temp_time = 0.0f;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    //=== slab hash's deletion:
    slab_hash_->deleteIndividual(d_key_, num_keys);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&temp_time, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return temp_time;
  }
  float measureLoadFactor(int flag = 0) {
    return slab_hash_->computeLoadFactor(flag);
  }
};