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
__device__ __forceinline__ bool
GpuSlabHashContext<KeyT, ValueT, SlabHashTypeT::ConcurrentMap>::deleteKey(
    bool& to_be_deleted,
    const uint32_t& laneId,
    const KeyT& myKey,
    const uint32_t bucket_id) {
  // delete the first instance of key

  using SlabHashT = ConcurrentMapT<KeyT, ValueT>;
  uint32_t work_queue = 0;
  uint32_t last_work_queue = 0;
  uint32_t next = SlabHashT::A_INDEX_POINTER;
  bool successful_deletion = false;

  while ((work_queue = __ballot_sync(0xFFFFFFFF, to_be_deleted))) {
    // to know whether it is a base node, or a regular node
    next = (last_work_queue != work_queue) ? SlabHashT::A_INDEX_POINTER
                                           : next;  // a successfull insertion in the warp
    uint32_t src_lane = __ffs(work_queue) - 1;
    uint32_t src_key = __shfl_sync(0xFFFFFFFF,
                                   *reinterpret_cast<const uint32_t*>(
                                       reinterpret_cast<const unsigned char*>(&myKey)),
                                   src_lane,
                                   32);
    uint32_t src_bucket = __shfl_sync(0xFFFFFFFF, bucket_id, src_lane, 32);
    // starting with a base node OR regular node:
    // need to define different masks to extract super block index, memory block
    // index, and the memory unit index

    const uint32_t src_unit_data = (next == SlabHashT::A_INDEX_POINTER)
                                       ? *(getPointerFromBucket(src_bucket, laneId))
                                       : *(getPointerFromSlab(next, laneId));

    // looking for the item to be deleted:
    uint32_t isFound = (__ballot_sync(0xFFFFFFFF, src_unit_data == src_key)) &
                       SlabHashT::REGULAR_NODE_KEY_MASK;

    if (isFound == 0) {  // no matching slot found:
      uint32_t next_ptr = __shfl_sync(0xFFFFFFFF, src_unit_data, 31, 32);
      if (next_ptr == SlabHashT::EMPTY_INDEX_POINTER) {
        // not found:
        if (laneId == src_lane)
          to_be_deleted = false;
      } else {
        next = next_ptr;
      }
    } else {  // The wanted key found:
      int dest_lane = __ffs(isFound & SlabHashT::REGULAR_NODE_KEY_MASK) - 1;
      if (laneId == src_lane) {
        uint32_t* p = (next == SlabHashT::A_INDEX_POINTER)
                          ? getPointerFromBucket(src_bucket, dest_lane)
                          : getPointerFromSlab(next, dest_lane);

        uint64_t old_pair = atomicExch((unsigned long long int*)p, EMPTY_PAIR_64);
        uint32_t deleted_key = static_cast<uint32_t>(old_pair);
        successful_deletion = deleted_key == src_key;
        to_be_deleted = false;
      }
    }
    last_work_queue = work_queue;
  }
  return successful_deletion;
}