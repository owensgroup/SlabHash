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
class GpuSlabHashContext<KeyT, ValueT, SlabHashType::ConcurrentMap> {
 public:
  // fixed known parameters:
  static constexpr uint32_t PRIME_DIVISOR_ = 4294967291u;

  GpuSlabHashContext()
      : num_buckets_(0), hash_x_(0), hash_y_(0), d_table_(nullptr) {}

  __host__ void initParameters(const uint32_t num_buckets,
                               const uint32_t hash_x,
                               const uint32_t hash_y,
                               concurrent_slab<KeyT, ValueT>* d_table,
                               AllocatorContextT* allocator_ctx) {
    num_buckets_ = num_buckets;
    hash_x_ = hash_x;
    hash_y_ = hash_y;
    d_table_ = d_table;
    dynamic_allocator_ = *allocator_ctx;
  }

  __device__ __host__ __forceinline__ AllocatorContextT& getAllocatorContext() {
    return dynamic_allocator_;
  }

  __device__ __host__ __forceinline__ concurrent_slab<KeyT, ValueT>*
  getDeviceTablePointer() {
    return d_table_;
  }

  __device__ __host__ __forceinline__ uint32_t
  computeBucket(const KeyT& key) const {
    return (((hash_x_ ^ key) + hash_y_) % PRIME_DIVISOR_) % num_buckets_;
  }

  // threads in a warp cooperate with each other to insert key-value pairs
  // into the slab hash
  __device__ __forceinline__ void insertPair(bool& to_be_inserted,
                                             const uint32_t& laneId,
                                             const KeyT& myKey,
                                             const ValueT& myValue,
                                             const uint32_t bucket_id);

  // threads in a warp cooeparte with each other to search for keys
  // if found, it returns the corresponding value, else SEARCH_NOT_FOUND
  // is returned 
  __device__ __forceinline__ void searchKey(bool& to_be_searched,
                                            const uint32_t& laneId,
                                            const KeyT& myKey,
                                            ValueT& myValue,
                                            const uint32_t bucket_id);

  // threads in a warp cooperate with each other to search for keys.
  // the main difference with above function is that it is assumed all 
  // threads have something to search for
  __device__ __forceinline__ void searchKeyBulk(const uint32_t& laneId,
                                                const KeyT& myKey,
                                                ValueT& myValue,
                                                const uint32_t bucket_id);

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

  __device__ __forceinline__ uint32_t* getPointerFromSlab(
      const SlabAddressT& slab_address,
      const uint32_t laneId) {
    return dynamic_allocator_.getPointerFromSlab(slab_address, laneId);
  }

  __device__ __forceinline__ uint32_t* getPointerFromBucket(
      const uint32_t bucket_id,
      const uint32_t laneId) {
    return reinterpret_cast<uint32_t*>(d_table_) + bucket_id * BASE_UNIT_SIZE +
           laneId;
  }

  // === members:
  uint32_t num_buckets_;
  uint32_t hash_x_;
  uint32_t hash_y_;
  concurrent_slab<KeyT, ValueT>* d_table_;
  // a copy of dynamic allocator's context to be used on the GPU
  AllocatorContextT dynamic_allocator_;
};