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

//================================================
// Individual Count Unit:
//================================================
template <typename KeyT, typename ValueT>
__device__ __forceinline__ void
GpuSlabHashContext<KeyT, ValueT, SlabHashTypeT::ConcurrentMap>::countKey(
    bool& to_be_searched,
    const uint32_t& laneId,
    const KeyT& myKey,
    uint32_t& myCount,
    const uint32_t bucket_id) {
  using SlabHashT = ConcurrentMapT<KeyT, ValueT>;
  uint32_t work_queue = 0;
  uint32_t last_work_queue = work_queue;
  uint32_t next = SlabHashT::A_INDEX_POINTER;

  while ((work_queue = __ballot_sync(0xFFFFFFFF, to_be_searched))) {
    next = (last_work_queue != work_queue) ? SlabHashT::A_INDEX_POINTER
                                           : next;
    uint32_t src_lane = __ffs(work_queue) - 1;
    uint32_t src_bucket = __shfl_sync(0xFFFFFFFF, bucket_id, src_lane, 32);
    uint32_t wanted_key = __shfl_sync(0xFFFFFFFF,
                                      *reinterpret_cast<const uint32_t*>(
                                          reinterpret_cast<const unsigned char*>(&myKey)),
                                      src_lane,
                                      32);
    const uint32_t src_unit_data = (next == SlabHashT::A_INDEX_POINTER)
                                       ? *(getPointerFromBucket(src_bucket, laneId))
                                       : *(getPointerFromSlab(next, laneId));
    const int wanted_key_count = __popc(__ballot_sync(0xFFFFFFFF, src_unit_data == wanted_key) &
                           SlabHashT::REGULAR_NODE_KEY_MASK);
    
    if(laneId == src_lane) //count
      myCount += wanted_key_count;

    uint32_t next_ptr = __shfl_sync(0xFFFFFFFF, src_unit_data, 31, 32); //iterate
    if (next_ptr == SlabHashT::EMPTY_INDEX_POINTER){
      if(laneId == src_lane){
        to_be_searched = false;
      }
    }
    else{
      next = next_ptr;
    }

    last_work_queue = work_queue;
  }
}

