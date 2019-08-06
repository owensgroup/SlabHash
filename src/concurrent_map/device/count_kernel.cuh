/*
 * Copyright 2019 University of California, Davis
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

template <typename KeyT, typename ValueT>
__global__ void count_key(
    KeyT* d_queries,
    uint32_t* d_counts,
    uint32_t num_queries,
    GpuSlabHashContext<KeyT, ValueT, SlabHashTypeT::ConcurrentMap> slab_hash) {
  uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  uint32_t laneId = threadIdx.x & 0x1F;

  if ((tid - laneId) >= num_queries) {
    return;
  }

  KeyT myKey = 0;
  uint32_t myCount = 0;
  uint32_t myBucket = 0;
  bool to_count = false;

  if (tid < num_queries) {
    myKey = d_queries[tid];
    myBucket = slab_hash.computeBucket(myKey);
    to_count = true;
  }

  // count the keys:
  slab_hash.countKey(to_count, laneId, myKey, myCount, myBucket);

  // writing back the results:
  if (tid < num_queries) {
    d_counts[tid] = myCount;
  }
}