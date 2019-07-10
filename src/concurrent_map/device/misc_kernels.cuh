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
 * This kernel can be used to compute total number of elements within each
 * bucket. The final results per bucket is stored in d_count_result array
 */
template <typename KeyT, typename ValueT>
__global__ void bucket_count_kernel(
    GpuSlabHashContext<KeyT, ValueT, SlabHashTypeT::ConcurrentMap> slab_hash,
    uint32_t* d_count_result,
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

  uint32_t count = 0;

  uint32_t src_unit_data = *slab_hash.getPointerFromBucket(wid, laneId);

  count += __popc(__ballot_sync(0xFFFFFFFF, src_unit_data != EMPTY_KEY) &
                  SlabHashT::REGULAR_NODE_KEY_MASK);
  uint32_t next = __shfl_sync(0xFFFFFFFF, src_unit_data, 31, 32);

  while (next != SlabHashT::EMPTY_INDEX_POINTER) {
    src_unit_data = *slab_hash.getPointerFromSlab(next, laneId);
    count += __popc(__ballot_sync(0xFFFFFFFF, src_unit_data != EMPTY_KEY) &
                    SlabHashT::REGULAR_NODE_KEY_MASK);
    next = __shfl_sync(0xFFFFFFFF, src_unit_data, 31, 32);
  }
  // writing back the results:
  if (laneId == 0) {
    d_count_result[wid] = count;
  }
}

/*
 * This kernel goes through all allocated bitmaps for a slab_hash's allocator
 * and store number of allocated slabs.
 * TODO: this should be moved into allocator's codebase (violation of layers)
 */
template <typename KeyT, typename ValueT>
__global__ void compute_stats_allocators(
    uint32_t* d_count_super_block,
    GpuSlabHashContext<KeyT, ValueT, SlabHashTypeT::ConcurrentMap> slab_hash) {
  uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;

  int num_bitmaps = slab_hash.getAllocatorContext().NUM_MEM_BLOCKS_PER_SUPER_BLOCK_ * 32;
  if (tid >= num_bitmaps) {
    return;
  }

  for (int i = 0; i < slab_hash.getAllocatorContext().num_super_blocks_; i++) {
    uint32_t read_bitmap = *(slab_hash.getAllocatorContext().getPointerForBitmap(i, tid));
    atomicAdd(&d_count_super_block[i], __popc(read_bitmap));
  }
}