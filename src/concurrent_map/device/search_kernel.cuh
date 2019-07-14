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

//=== Individual search kernel:
template <typename KeyT, typename ValueT>
__global__ void search_table(
    KeyT* d_queries,
    ValueT* d_results,
    uint32_t num_queries,
    GpuSlabHashContext<KeyT, ValueT, SlabHashTypeT::ConcurrentMap> slab_hash) {
  uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  uint32_t laneId = threadIdx.x & 0x1F;

  if ((tid - laneId) >= num_queries) {
    return;
  }

  KeyT myQuery = 0;
  ValueT myResult = static_cast<ValueT>(SEARCH_NOT_FOUND);
  uint32_t myBucket = 0;
  bool to_search = false;
  if (tid < num_queries) {
    myQuery = d_queries[tid];
    myBucket = slab_hash.computeBucket(myQuery);
    to_search = true;
  }

  slab_hash.searchKey(to_search, laneId, myQuery, myResult, myBucket);

  // writing back the results:
  if (tid < num_queries) {
    d_results[tid] = myResult;
  }
}

//=== Bulk search kernel:
template <typename KeyT, typename ValueT>
__global__ void search_table_bulk(
    KeyT* d_queries,
    ValueT* d_results,
    uint32_t num_queries,
    GpuSlabHashContext<KeyT, ValueT, SlabHashTypeT::ConcurrentMap> slab_hash) {
  uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  uint32_t laneId = threadIdx.x & 0x1F;

  if ((tid - laneId) >= num_queries) {
    return;
  }

  KeyT myQuery = 0;
  ValueT myResult = static_cast<ValueT>(SEARCH_NOT_FOUND);
  uint32_t myBucket = 0;
  if (tid < num_queries) {
    myQuery = d_queries[tid];
    myBucket = slab_hash.computeBucket(myQuery);
  }

  slab_hash.searchKeyBulk(laneId, myQuery, myResult, myBucket);

  // writing back the results:
  if (tid < num_queries) {
    d_results[tid] = myResult;
  }
}