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

#include <cassert>
#include <iostream>
#include "slab_hash_global.cuh"

template <typename KeyT, typename ValueT>
class GpuSlabHash<KeyT, ValueT, SlabHashType::ConcurrentMap> {
 private:
  // the used hash function:
  uint32_t num_buckets_;
  // size_t 		allocator_heap_size_;
  // hash_function hf_;
  // bucket data structures:
  concurrent_slab<KeyT, ValueT>* d_table_;
  // dynamic memory allocator:
  // slab_alloc::context_alloc<1> ctx_alloc_;
 public:
  GpuSlabHash() : num_buckets_(10), d_table_(nullptr) {
    std::cout << " == slab hash concstructor called" << std::endl;
  }

  ~GpuSlabHash() {}
};