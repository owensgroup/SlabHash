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

// only works with up to 32-bit key/values
template <typename KeyT, typename ValueT>
struct key_value_pair {
  KeyT key;
  ValueT value;
};

template <typename KeyT, typename ValueT>
struct __align__(32) concurrent_slab {
  static constexpr uint32_t NUM_ELEMENTS_PER_SLAB = 15u;
  key_value_pair<KeyT, ValueT> data[NUM_ELEMENTS_PER_SLAB];
  uint32_t ptr_index[2];
};

enum class SlabHashType { ConcurrentMap, PhaseConcurrentMap };

template <typename KeyT, typename ValueT, SlabHashType SlabHashT>
class GpuSlabHash;
