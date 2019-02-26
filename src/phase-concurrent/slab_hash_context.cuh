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
 * This is the main class that will be shallowly copied into the device to be
 * used at runtime. This class does not own the allocated memory on the gpu
 * (i.e., d_table_)
 */
template <typename KeyT, typename ValueT>
class GpuSlabHashContext<KeyT, ValueT, SlabHashType::PhaseConcurrentMap> {
 public:
  GpuSlabHashContext()
      : num_buckets_(0), hash_x_(0), hash_y_(0), d_table_(nullptr) {
    // a single slab on a ConcurrentMap should be 128 bytes
    printf("phase concurrent slab size is %d\n",
           sizeof(phase_concurrent_slab<KeyT, ValueT>));
    // assert(sizeof(phase_concurrent_slab<KeyT, ValueT>) ==
    //        (WARP_WIDTH_ * sizeof(uint32_t)));
  }

  static size_t getSlabUnitSize() {
    return sizeof(phase_concurrent_slab<KeyT, ValueT>);
  }

  static std::string getSlabHashTypeName() {
    return std::string("PhaseConcurrentMap");
  }

  __host__ void initParameters(const uint32_t num_buckets,
                               const uint32_t hash_x,
                               const uint32_t hash_y,
                               int8_t* d_table,
                               AllocatorContextT* allocator_ctx) {
    num_buckets_ = num_buckets;
    hash_x_ = hash_x;
    hash_y_ = hash_y;
    d_table_ = reinterpret_cast<phase_concurrent_slab<KeyT, ValueT>*>(d_table);
    dynamic_allocator_ = *allocator_ctx;
  }

 private:
  // this function should be operated in a warp-wide fashion
  // TODO: add required asserts to make sure this is true in tests/debugs
  __device__ __forceinline__ SlabAllocAddressT
  allocateSlab(const uint32_t& laneId) {
    return dynamic_allocator_.warpAllocate(laneId);
  }

  // a thread-wide function to free the slab that was just allocated
  __device__ __forceinline__ void freeSlab(const SlabAllocAddressT slab_ptr) {
    dynamic_allocator_.freeUntouched(slab_ptr);
  }

  // === members:
  uint32_t num_buckets_;
  uint32_t hash_x_;
  uint32_t hash_y_;
  phase_concurrent_slab<KeyT, ValueT>* d_table_;
  // a copy of dynamic allocator's context to be used on the GPU
  AllocatorContextT dynamic_allocator_;
};