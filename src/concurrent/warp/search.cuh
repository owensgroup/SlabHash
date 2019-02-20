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
__device__ __forceinline__ void
GpuSlabHashContext<KeyT, ValueT, SlabHashType::ConcurrentMap>::searchKey(
    bool& to_be_searched,
    const uint32_t& laneId,
    const KeyT& myKey,
    ValueT& myValue,
    const uint32_t bucket_id) {
  uint32_t work_queue = 0;
  uint32_t last_work_queue = work_queue;
  uint32_t next = A_INDEX_POINTER;

  while ((work_queue = __ballot_sync(0xFFFFFFFF, to_be_searched))) {
    next = (last_work_queue != work_queue)
               ? A_INDEX_POINTER
               : next;  // a successfull insertion in the warp
    uint32_t src_lane = __ffs(work_queue) - 1;
    uint32_t src_bucket = __shfl_sync(0xFFFFFFFF, bucket_id, src_lane, 32);
    uint32_t wanted_key =
        __shfl_sync(0xFFFFFFFF,
                    *reinterpret_cast<const uint32_t*>(
                        reinterpret_cast<const unsigned char*>(&myKey)),
                    src_lane, 32);
    const uint32_t src_unit_data =
        (next == A_INDEX_POINTER) ? *(getPointerFromBucket(src_bucket, laneId))
                                  : *(getPointerFromSlab(next, laneId));
    int found_lane =
        __ffs(__ballot_sync(0xFFFFFFFF, src_unit_data == wanted_key) &
              REGULAR_NODE_KEY_MASK) -
        1;
    if (found_lane < 0) {  // not found
      uint32_t next_ptr = __shfl_sync(0xFFFFFFFF, src_unit_data, 31, 32);
      if (next_ptr == EMPTY_INDEX_POINTER) {  // not found
        if (laneId == src_lane) {
          myValue = static_cast<ValueT>(SEARCH_NOT_FOUND);
          to_be_searched = false;
        }
      } else {
        next = next_ptr;
      }
    } else {  // found the key:
      uint32_t found_value =
          __shfl_sync(0xFFFFFFFF, src_unit_data, found_lane + 1, 32);
      if (laneId == src_lane) {
        myValue = *reinterpret_cast<const ValueT*>(
            reinterpret_cast<const unsigned char*>(&found_value));
        to_be_searched = false;
      }
    }
    last_work_queue = work_queue;
  }
}