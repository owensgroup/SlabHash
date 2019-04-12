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

#include <stdio.h>
#include <time.h>
#include <unistd.h>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>
#include "gpu_hash_table.cuh"
#include "slab_hash.cuh"
//=======================================
#define DEVICE_ID 0
//=======================================

template <typename KeyT>
__global__ void print_table(
    GpuSlabHashContext<KeyT, KeyT, SlabHashTypeT::ConcurrentSet> slab_hash) {
  uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  uint32_t wid = tid >> 5;
  uint32_t laneId = threadIdx.x & 0x1F;

  if (wid >= slab_hash.getNumBuckets()) {
    return;
  }

  // initializing the memory allocator on each warp:
  slab_hash.getAllocatorContext().initAllocator(tid, laneId);

  if (tid == 0) {
    printf(" == Printing the base array\n");
    SlabIterator<KeyT> iter(slab_hash);
    for (int i = 0; i < iter.cur_size_; i++) {
      if ((i & 0x1F) == 0)
        printf(" == bucket %d:\n", i >> 5);
      printf("%8x, ", *(iter.cur_ptr_ + i));
      if ((i & 0x7) == 0x7)
        printf("\n");
    }
    printf("\n");

    printf(" == Printing the rest of slabs:\n");
    while (iter.next()) {
      for (int i = 0; i < iter.cur_size_; i++) {
        if ((i & 0x1F) == 0)
          printf(" == bucket %d:\n", iter.cur_bucket_);
        printf("%8x, ", *(iter.cur_ptr_ + i));
        if ((i & 0x7) == 0x7)
          printf("\n");
      }
      printf("\n");
    }
  }
}

//=======================================
int main(int argc, char** argv) {
  //=========
  int devCount;
  cudaGetDeviceCount(&devCount);
  cudaDeviceProp devProp;
  if (devCount) {
    cudaSetDevice(DEVICE_ID);  // be changed later
    cudaGetDeviceProperties(&devProp, DEVICE_ID);
  }
  printf("Device: %s\n", devProp.name);

  //======================================
  // Building my hash table:
  //======================================
  uint32_t num_buckets = 2;

  using KeyT = uint32_t;

  std::vector<KeyT> h_key = {2,  4,  6,  8,  10, 1,  3,  5,  7,  9,
                             11, 13, 15, 17, 19, 21, 23, 25, 27, 29,
                             31, 33, 35, 37, 39, 41, 43, 45, 47, 49,
                             51, 53, 55, 57, 59, 61, 63, 65, 67};
  uint32_t num_keys = h_key.size();

  const int64_t seed = 1;
  std::mt19937 rng(seed);
  std::shuffle(h_key.begin(), h_key.end(), rng);

  gpu_hash_table<KeyT, KeyT, SlabHashTypeT::ConcurrentSet>
      hash_table(num_keys, num_buckets, DEVICE_ID, seed, false, /*identity_hash*/ true);

  float build_time = hash_table.hash_build(h_key.data(), nullptr, num_keys);

  const uint32_t num_blocks = 1;
  const uint32_t num_threads = 128;
  print_table<KeyT><<<num_blocks, num_threads>>>(
      hash_table.slab_hash_->getSlabHashContext());

  return 0;
}