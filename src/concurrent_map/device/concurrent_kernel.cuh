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

template <typename KeyT, typename ValueT>
__global__ void batched_operations(
    uint32_t* d_operations,
    uint32_t* d_results,
    uint32_t num_operations,
    GpuSlabHashContext<KeyT, ValueT, SlabHashTypeT::ConcurrentMap> slab_hash) {
  uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  uint32_t laneId = threadIdx.x & 0x1F;

  if ((tid - laneId) >= num_operations)
    return;

  // initializing the memory allocator on each warp:
  AllocatorContextT local_allocator_ctx(slab_hash.getAllocatorContext());
  local_allocator_ctx.initAllocator(tid, laneId);

  uint32_t myOperation = 0;
  uint32_t myKey = 0;
  uint32_t myValue = 0;
  uint32_t myBucket = 0;

  if (tid < num_operations) {
    myOperation = d_operations[tid];
    myKey = myOperation & 0x3FFFFFFF;
    myBucket = slab_hash.computeBucket(myKey);
    myOperation = myOperation >> 30;
    // todo: should be changed to a more general case
    myValue = myKey;  // for the sake of this benchmark
  }

  bool to_insert = (myOperation == 1) ? true : false;
  bool to_delete = (myOperation == 2) ? true : false;
  bool to_search = (myOperation == 3) ? true : false;

  // first insertions:
  slab_hash.insertPair(to_insert, laneId, myKey, myValue, myBucket, local_allocator_ctx);

  // second deletions:
  slab_hash.deleteKey(to_delete, laneId, myKey, myBucket);

  // finally search queries:
  slab_hash.searchKey(to_search, laneId, myKey, myValue, myBucket);

  if (myOperation == 3 && myValue != SEARCH_NOT_FOUND) {
    d_results[tid] = myValue;
  }
}