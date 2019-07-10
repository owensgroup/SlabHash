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
void GpuSlabHash<KeyT, ValueT, SlabHashTypeT::ConcurrentSet>::buildBulk(
    KeyT* d_key,
    ValueT* d_value,
    uint32_t num_keys) {
  const uint32_t num_blocks = (num_keys + BLOCKSIZE_ - 1) / BLOCKSIZE_;
  // calling the kernel for bulk build:
  CHECK_CUDA_ERROR(cudaSetDevice(device_idx_));
  cset::build_table_kernel<KeyT>
      <<<num_blocks, BLOCKSIZE_>>>(d_key, num_keys, gpu_context_);
}

template <typename KeyT, typename ValueT>
void GpuSlabHash<KeyT, ValueT, SlabHashTypeT::ConcurrentSet>::searchIndividual(
    KeyT* d_query,
    ValueT* d_result,
    uint32_t num_queries) {
  CHECK_CUDA_ERROR(cudaSetDevice(device_idx_));
  const uint32_t num_blocks = (num_queries + BLOCKSIZE_ - 1) / BLOCKSIZE_;
  cset::search_table<KeyT>
      <<<num_blocks, BLOCKSIZE_>>>(d_query, d_result, num_queries, gpu_context_);
}

template <typename KeyT, typename ValueT>
std::string GpuSlabHash<KeyT, ValueT, SlabHashTypeT::ConcurrentSet>::to_string() {
  std::string result;
  result += " ==== GpuSlabHash: \n";
  result += "\t Running on device \t\t " + std::to_string(device_idx_) + "\n";
  result += "\t SlabHashType:     \t\t " + gpu_context_.getSlabHashTypeName() + "\n";
  result += "\t Number of buckets:\t\t " + std::to_string(num_buckets_) + "\n";
  result += "\t d_table_ address: \t\t " +
            std::to_string(reinterpret_cast<uint64_t>(static_cast<void*>(d_table_))) +
            "\n";
  result += "\t hash function = \t\t (" + std::to_string(hf_.x) + ", " +
            std::to_string(hf_.y) + ")\n";
  return result;
}
