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
/*
 *
 */
template <typename KeyT, typename ValueT>
__global__ void build_table_kernel(
    KeyT* d_key,
    ValueT* d_value,
    uint32_t num_keys,
    GpuSlabHashContext<KeyT, ValueT, SlabHashTypeT::ConcurrentMap> slab_hash) {
  uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  uint32_t laneId = threadIdx.x & 0x1F;

  if ((tid - laneId) >= num_keys) {
    return;
  }

  AllocatorContextT local_allocator_ctx(slab_hash.getAllocatorContext());
  local_allocator_ctx.initAllocator(tid, laneId);

  KeyT myKey = 0;
  ValueT myValue = 0;
  uint32_t myBucket = 0;
  bool to_insert = false;

  if (tid < num_keys) {
    myKey = d_key[tid];
    myValue = d_value[tid];
    myBucket = slab_hash.computeBucket(myKey);
    to_insert = true;
  }

  slab_hash.insertPair(to_insert, laneId, myKey, myValue, myBucket, local_allocator_ctx);
}

template <typename KeyT, typename ValueT>
__global__ void build_table_with_unique_keys_kernel(
    KeyT* d_key,
    ValueT* d_value,
    uint32_t num_keys,
    GpuSlabHashContext<KeyT, ValueT, SlabHashTypeT::ConcurrentMap> slab_hash) {
  uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  uint32_t laneId = threadIdx.x & 0x1F;

  if ((tid - laneId) >= num_keys) {
    return;
  }

  AllocatorContextT local_allocator_ctx(slab_hash.getAllocatorContext());
  local_allocator_ctx.initAllocator(tid, laneId);

  KeyT myKey = 0;
  ValueT myValue = 0;
  uint32_t myBucket = 0;
  bool to_insert = false;

  if (tid < num_keys) {
    myKey = d_key[tid];
    myValue = d_value[tid];
    myBucket = slab_hash.computeBucket(myKey);
    to_insert = true;
  }

  slab_hash.insertPairUnique(
      to_insert, laneId, myKey, myValue, myBucket, local_allocator_ctx);
}