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

namespace SlabHash_NS {
/*
 * search for a key (and/or an empty spot) in a single slab, returns the laneId
 * if found, otherwise returns -1
 */
template <typename KeyT, class SlabHashT>
__device__ __forceinline__ int32_t
findKeyOrEmptyPerWarp(const KeyT& src_key, const uint32_t read_data_chunk) {
  uint32_t isEmpty =
      (__ballot_sync(0xFFFFFFFF, (read_data_chunk == EMPTY_KEY) ||
                                     (read_data_chunk == src_key)));
  return __ffs(isEmpty & SlabHashT::REGULAR_NODE_KEY_MASK) - 1;
}

// search for just the key
template <typename KeyT, class SlabHashT>
__device__ __forceinline__ int32_t
findKeyPerWarp(const KeyT& src_key, const uint32_t read_data_chunk) {
  uint32_t isEmpty = __ballot_sync(0xFFFFFFFF, (read_data_chunk == src_key));
  return __ffs(isEmpty & SlabHashT::REGULAR_NODE_KEY_MASK) - 1;
}

// search for an empty spot
template <typename KeyT, class SlabHashT>
__device__ __forceinline__ int32_t
findEmptyPerWarp(const uint32_t read_data_chunk) {
  uint32_t isEmpty = __ballot_sync(0xFFFFFFFF, (read_data_chunk == EMPTY_KEY));
  return __ffs(isEmpty & SlabHashT::REGULAR_NODE_KEY_MASK) - 1;
}
};  // namespace SlabHash_NS