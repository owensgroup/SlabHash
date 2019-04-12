/*
 * Copyright 2018 Saman Ashkiani
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

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <iostream>
#include "gpu_hash_table.cuh"
#include "slab_alloc.cuh"
#include "slab_hash.cuh"
#define DEVICE_ID 0

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

  auto my_hash_table =
      new gpu_hash_table<uint32_t, uint32_t,
                         SlabHashTypeT::ConcurrentMap>(100, 10, DEVICE_ID,/*seed = */ 1);

  std::vector<uint32_t> h_key{10, 5, 1};
  std::vector<uint32_t> h_value{100, 50, 10};

  my_hash_table->hash_build(h_key.data(), h_value.data(), h_key.size());
  // auto slab_alloc = new SlabAllocLight<8, 32, 1>();
  // printf("slab alloc constructed\n");

  // delete slab_alloc;

  // auto slab_hash =
  //     new GpuSlabHash<uint32_t, uint32_t, SlabHashType::ConcurrentMap>();
  // std::cout << slab_hash->to_string() << std::endl;
  delete my_hash_table;
  return 0;
}