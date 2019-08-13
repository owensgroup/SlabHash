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
 * This kernel can be used to compute the total number of elements and the total number of
 * slabs per bucket. The final results per bucket is stored in d_pairs_count_result and
 * d_slabs_count_result arrays respectively
 */
template <typename KeyT, typename ValueT>
__global__ void bucket_count_kernel(
    GpuSlabHashContext<KeyT, ValueT, SlabHashTypeT::ConcurrentMap> slab_hash,
    uint32_t* d_pairs_count_result,
    uint32_t* d_slabs_count_result,
    uint32_t num_buckets) {
  using SlabHashT = ConcurrentMapT<KeyT, ValueT>;
  // global warp ID
  uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  uint32_t wid = tid >> 5;
  // assigning a warp per bucket
  if (wid >= num_buckets) {
    return;
  }

  uint32_t laneId = threadIdx.x & 0x1F;

  // initializing the memory allocator on each warp:
  slab_hash.getAllocatorContext().initAllocator(tid, laneId);

  uint32_t pairs_count = 0;
  uint32_t slabs_count = 1;

  uint32_t src_unit_data = *slab_hash.getPointerFromBucket(wid, laneId);

  pairs_count += __popc(__ballot_sync(0xFFFFFFFF, src_unit_data != EMPTY_KEY) &
                        SlabHashT::REGULAR_NODE_KEY_MASK);
  uint32_t next = __shfl_sync(0xFFFFFFFF, src_unit_data, 31, 32);

  while (next != SlabHashT::EMPTY_INDEX_POINTER) {
    // counting pairs
    src_unit_data = *slab_hash.getPointerFromSlab(next, laneId);
    pairs_count += __popc(__ballot_sync(0xFFFFFFFF, src_unit_data != EMPTY_KEY) &
                          SlabHashT::REGULAR_NODE_KEY_MASK);
    next = __shfl_sync(0xFFFFFFFF, src_unit_data, 31, 32);
    // counting slabs
    slabs_count++;
  }
  // writing back the results:
  if (laneId == 0) {
    d_pairs_count_result[wid] = pairs_count;
    d_slabs_count_result[wid] = slabs_count;
  }
}
