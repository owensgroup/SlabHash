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
#include "slab_hash_global.cuh"

/*
 * each thread inserts a key-value pair into the hash table
 * it is assumed all threads within a warp are present and collaborating with
 * each other with a warp-cooperative work sharing (WCWS) strategy.
 */
template <typename KeyT, typename ValueT>
__device__ __forceinline__ void insert_pair(
    bool& to_be_inserted,
    uint32_t& laneId,
    KeyT& myKey,
    ValueT& myValue,
    uint32_t bucket_id,
    GpuSlabHash<KeyT, ValueT, SlabHashType::ConcurrentMap>& slab_hash
    /*slab_alloc::context_alloc<1>& context*/) {
  uint32_t work_queue = 0;
  uint32_t last_work_queue = 0;
  uint32_t next = A_INDEX_POINTER;

  while ((work_queue = __ballot_sync(0xFFFFFFFF, to_be_inserted))) {
    // to know whether it is a base node, or a regular node
    next = (last_work_queue != work_queue)
               ? A_INDEX_POINTER
               : next;  // a successfull insertion in the warp
    uint32_t src_lane = __ffs(work_queue) - 1;
    uint32_t src_bucket = __shfl_sync(0xFFFFFFFF, bucket_id, src_lane, 32);

    uint32_t src_unit_data =
        (next == A_INDEX_POINTER)
            ? *(reinterpret_cast<uint32_t*>(reinterpret_cast<unsigned char*>(
                    slab_hash.getDeviceTablePointer() /*d_buckets*/)) +
                (src_bucket * BASE_UNIT_SIZE + laneId))
            : 0 /* *(reinterpret_cast<uint32_t*>(
                    reinterpret_cast<unsigned char*>(context.d_super_blocks)) +
                slab_alloc::address_decoder<1>(next) + laneId) */
        ;

    uint64_t old_key_value_pair = 0;

    uint32_t isEmpty = (__ballot_sync(0xFFFFFFFF, src_unit_data == EMPTY_KEY)) &
                       REGULAR_NODE_KEY_MASK;
    if (isEmpty == 0) {  // no empty slot available:
      uint32_t next_ptr = __shfl_sync(0xFFFFFFFF, src_unit_data, 31, 32);
      if (next_ptr == EMPTY_INDEX_POINTER) {
        // allocate a new node:
        uint32_t new_node_ptr =
            0 /*slab_alloc::warp_allocate<1>(context, laneId)*/;
        // TODO: experiment if it's better to use lane 0 instead
        if (laneId == 31) {
          uint32_t* p =
              (next == A_INDEX_POINTER)
                  ? reinterpret_cast<uint32_t*>(
                        reinterpret_cast<unsigned char*>(
                            slab_hash.getDeviceTablePointer() /*d_buckets*/)) +
                        (src_bucket * BASE_UNIT_SIZE + 31)
                  : nullptr /* reinterpret_cast<uint32_t*>(
                        reinterpret_cast<unsigned char*>(
                            context.d_super_blocks) +
                        (slab_alloc::address_decoder<1>(next) + 31))*/
              ;

          uint32_t temp =
              atomicCAS((unsigned int*)p, EMPTY_INDEX_POINTER, new_node_ptr);
          // check whether it was successful, and
          // free the allocated memory otherwise
          /* == temp: if (temp != EMPTY_INDEX_POINTER)

            slab_alloc::free_untouched<1>(context, new_node_ptr);*/
        }
      } else {
        next = next_ptr;
      }
    } else {  // there is an empty slot available
      int dest_lane = __ffs(isEmpty & REGULAR_NODE_KEY_MASK) - 1;
      if (laneId == src_lane) {
        uint32_t* p =
            (next == A_INDEX_POINTER)
                ? reinterpret_cast<uint32_t*>(reinterpret_cast<unsigned char*>(
                      slab_hash.getDeviceTablePointer() /*d_buckets*/)) +
                      (src_bucket * BASE_UNIT_SIZE + dest_lane)
                : nullptr /* reinterpret_cast<uint32_t*>(reinterpret_cast<unsigned
                      char*>( context.d_super_blocks)) +
                      (slab_alloc::address_decoder<1>(next) + dest_lane)*/
            ;
        old_key_value_pair =
            atomicCAS((unsigned long long int*)p, EMPTY_PAIR_64,
                      ((uint64_t)(*reinterpret_cast<uint32_t*>(
                           reinterpret_cast<unsigned char*>(&myValue)))
                       << 32) |
                          *reinterpret_cast<uint32_t*>(
                              reinterpret_cast<unsigned char*>(&myKey)));
        if (old_key_value_pair == EMPTY_PAIR_64)
          to_be_inserted = false;  // succesfful insertion
      }
    }
    last_work_queue = work_queue;
  }
}