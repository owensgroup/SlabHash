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
#include <gtest/gtest.h>
#include <stdio.h>
#include <cstdlib>
#include <iostream>
#include <random>
#include <unordered_map>

#include "CommandLine.h"
#include "gpu_hash_table.cuh"
#include "slab_alloc.cuh"
#include "slab_hash.cuh"

size_t g_gpu_device_idx{0};  // the gpu device to run tests on

TEST(ConcurrentMap, Construction) {
  gpu_hash_table<uint32_t, uint32_t, SlabHashTypeT::ConcurrentMap> cmap(
      100, 10, g_gpu_device_idx, /*seed = */ 1);

  std::vector<uint32_t> h_key{10, 5, 1};
  std::vector<uint32_t> h_value{100, 50, 10};

  cmap.hash_build(h_key.data(), h_value.data(), h_key.size());
}

TEST(BulkBuild, IndividualSearch) {
  using KeyT = uint32_t;
  using ValueT = uint32_t;
  const uint32_t num_keys = 137;
  const uint32_t num_buckets = 2;
  // creating the data structures:
  gpu_hash_table<KeyT, ValueT, SlabHashTypeT::ConcurrentMap> cmap(
      num_keys, num_buckets, g_gpu_device_idx, /*seed = */ 1);

  // creating key-value pairs:
  std::vector<KeyT> h_key;
  h_key.reserve(num_keys);
  std::vector<ValueT> h_value;
  h_value.reserve(num_keys);
  for (uint32_t i_key = 0; i_key < num_keys; i_key++) {
    h_key.push_back(13 + i_key);
    h_value.push_back(1000 + h_key.back());
  }

  // building the slab hash, and the host's data structure:
  cmap.hash_build(h_key.data(), h_value.data(), h_key.size());

  // generating random queries
  const auto num_queries = num_keys;
  std::random_device rd;
  std::mt19937 rng(rd());
  std::vector<KeyT> h_query(h_key);
  std::shuffle(h_query.begin(), h_query.end(), rng);
  std::vector<ValueT> cmap_results(num_queries);

  // searching for the queries:
  cmap.hash_search(h_query.data(), cmap_results.data(), num_queries);

  // validating the results:
  std::unordered_map<KeyT, ValueT> hash_map;
  for (uint32_t i_key = 0; i_key < num_keys; i_key++) {
    hash_map.insert(std::make_pair(h_key[i_key], h_value[i_key]));
  }

  for (uint32_t i = 0; i < num_queries; i++) {
    auto cmap_result = cmap_results[i];
    auto expected_result = hash_map[h_query[i]];
    ASSERT_EQ(expected_result, cmap_result);
  }
}

TEST(BulkBuild, BulkSearch) {
  using KeyT = uint32_t;
  using ValueT = uint32_t;
  const uint32_t num_keys = 137;
  const uint32_t num_buckets = 2;
  // creating the data structures:
  gpu_hash_table<KeyT, ValueT, SlabHashTypeT::ConcurrentMap> cmap(
      num_keys, num_buckets, g_gpu_device_idx, /*seed = */ 1);

  // creating key-value pairs:
  std::vector<KeyT> h_key;
  h_key.reserve(num_keys);
  std::vector<ValueT> h_value;
  h_value.reserve(num_keys);
  for (uint32_t i_key = 0; i_key < num_keys; i_key++) {
    h_key.push_back(13 + i_key);
    h_value.push_back(1000 + h_key.back());
  }

  // building the slab hash, and the host's data structure:
  cmap.hash_build(h_key.data(), h_value.data(), h_key.size());

  // generating random queries
  const auto num_queries = num_keys;
  std::random_device rd;
  std::mt19937 rng(rd());
  std::vector<KeyT> h_query(h_key);
  std::shuffle(h_query.begin(), h_query.end(), rng);
  std::vector<ValueT> cmap_results(num_queries);

  // searching for the queries:
  cmap.hash_search_bulk(h_query.data(), cmap_results.data(), num_queries);

  // validating the results:
  std::unordered_map<KeyT, ValueT> hash_map;
  for (uint32_t i_key = 0; i_key < num_keys; i_key++) {
    hash_map.insert(std::make_pair(h_key[i_key], h_value[i_key]));
  }

  for (uint32_t i = 0; i < num_queries; i++) {
    auto cmap_result = cmap_results[i];
    auto expected_result = hash_map[h_query[i]];
    ASSERT_EQ(expected_result, cmap_result);
  }
}

TEST(BulkBuild, IndividualCount) {
  using KeyT = uint32_t;
  using ValueT = uint32_t;
  const uint32_t num_unique = 2014;
  const uint32_t num_buckets = 12;
  const uint32_t max_count = 32;

  // rng
  std::random_device rd;
  std::mt19937 rng(rd());

  // random key counts
  uint32_t num_keys = 0;
  std::vector<uint32_t> h_count;
  h_count.reserve(num_unique);
  for (uint32_t i_key = 0; i_key < num_unique; i_key++) {
    uint32_t key_count = rng() % max_count;
    h_count.push_back(key_count);
    num_keys += key_count;
  }

  // creating key-value pairs:
  std::vector<KeyT> h_key;
  h_key.reserve(num_keys);
  std::vector<KeyT> h_value;
  h_value.reserve(num_keys);
  std::vector<KeyT> h_key_unique;
  h_key_unique.reserve(num_unique);
  for (uint32_t i_key = 0; i_key < num_unique; i_key++) {
    KeyT myKey = 13 + i_key;
    ValueT myValue = 1000 + myKey;
    h_key_unique.push_back(myKey);
    for (uint32_t i_count = 0; i_count < h_count[i_key]; i_count++) {
      h_key.push_back(myKey);
      h_value.push_back(myValue);
    }
  }

  // creating the data structures:
  gpu_hash_table<KeyT, ValueT, SlabHashTypeT::ConcurrentMap> cmap(
      num_keys, num_buckets, g_gpu_device_idx, /*seed = */ 1);

  // building the slab hash, and the host's data structure:
  cmap.hash_build(h_key.data(), h_value.data(), h_key.size());

  // generating random queries
  const auto num_queries = num_unique;
  std::vector<KeyT> h_query(h_key_unique);
  std::shuffle(h_query.begin(), h_query.end(), rng);
  std::vector<ValueT> cmap_results(num_queries);

  // getting count per query:
  cmap.hash_count(h_query.data(), cmap_results.data(), num_queries);

  // validating the results:
  std::unordered_map<KeyT, ValueT> count_map;
  for (uint32_t i_key = 0; i_key < num_unique; i_key++) {
    count_map.insert(std::make_pair(h_key_unique[i_key], h_count[i_key]));
  }

  for (uint32_t i = 0; i < num_queries; i++) {
    auto cmap_result = cmap_results[i];
    auto expected_result = count_map[h_query[i]];
    ASSERT_EQ(expected_result, cmap_result);
  }
}

TEST(UniqueBulkBuild, IndividualCount) {
  using KeyT = uint32_t;
  using ValueT = uint32_t;
  const uint32_t num_unique = 2014;
  const uint32_t num_buckets = 12;
  const uint32_t max_count = 32;

  // rng
  std::random_device rd;
  std::mt19937 rng(rd());

  // random key counts
  uint32_t num_keys = 0;
  std::vector<uint32_t> h_count;
  h_count.reserve(num_unique);
  for (uint32_t i_key = 0; i_key < num_unique; i_key++) {
    uint32_t key_count = rng() % max_count;
    h_count.push_back(key_count);
    num_keys += key_count;
  }

  // creating key-value pairs:
  std::vector<KeyT> h_key;
  h_key.reserve(num_keys);
  std::vector<KeyT> h_value;
  h_value.reserve(num_keys);
  std::vector<KeyT> h_key_unique;
  h_key_unique.reserve(num_unique);
  for (uint32_t i_key = 0; i_key < num_unique; i_key++) {
    KeyT myKey = 13 + i_key;
    ValueT myValue = 1000 + myKey;
    h_key_unique.push_back(myKey);
    for (uint32_t i_count = 0; i_count < h_count[i_key]; i_count++) {
      h_key.push_back(myKey);
      h_value.push_back(myValue);
    }
  }

  // creating the data structures:
  gpu_hash_table<KeyT, ValueT, SlabHashTypeT::ConcurrentMap> cmap(
      num_keys, num_buckets, g_gpu_device_idx, /*seed = */ 1);

  // building the unique-keys slab hash, and the host's data structure:
  cmap.hash_build_with_unique_keys(h_key.data(), h_value.data(), h_key.size());

  // generating random queries
  const auto num_queries = num_unique;
  std::vector<KeyT> h_query(h_key_unique);
  std::shuffle(h_query.begin(), h_query.end(), rng);
  std::vector<ValueT> cmap_results(num_queries);

  // getting count per query:
  cmap.hash_count(h_query.data(), cmap_results.data(), num_queries);

  // validating the results:
  std::unordered_map<KeyT, ValueT> count_map;
  for (uint32_t i_key = 0; i_key < num_unique; i_key++) {
    count_map.insert(std::make_pair(h_key_unique[i_key], h_count[i_key]));
  }

  for (uint32_t i = 0; i < num_queries; i++) {
    auto cmap_result = cmap_results[i];
    auto expected_result = (count_map[h_query[i]] != 0) ? 1 : 0;
    ASSERT_EQ(expected_result, cmap_result);
  }
}

TEST(BulkBuild, IndividualDelete) {
  using KeyT = uint32_t;
  using ValueT = uint32_t;
  const uint32_t num_keys = 137;
  const uint32_t num_buckets = 2;
  // creating the data structures:
  gpu_hash_table<KeyT, ValueT, SlabHashTypeT::ConcurrentMap> cmap(
      num_keys, num_buckets, g_gpu_device_idx, /*seed = */ 1);

  // creating key-value pairs:
  std::vector<KeyT> h_key;
  h_key.reserve(num_keys);
  std::vector<ValueT> h_value;
  h_value.reserve(num_keys);
  for (uint32_t i_key = 0; i_key < num_keys; i_key++) {
    h_key.push_back(13 + i_key);
    h_value.push_back(1000 + h_key.back());
  }

  // building the slab hash:
  cmap.hash_build(h_key.data(), h_value.data(), h_key.size());

  // generating random keys to delete:
  const auto num_deletion = num_keys;
  const auto extend_fact = 4;
  std::random_device rd;
  std::mt19937 rng(rd());
  std::vector<KeyT> h_deleted_keys;
  h_deleted_keys.reserve(num_deletion * extend_fact);
  for (uint32_t i_key = 0; i_key < num_deletion * extend_fact; i_key++) {
    h_deleted_keys.push_back(13 + i_key);
  }
  std::shuffle(h_deleted_keys.begin(), h_deleted_keys.end(), rng);

  // delete the keys:
  cmap.hash_delete(h_deleted_keys.data(), num_deletion);

  // query all keys:
  const auto num_queries = num_keys;
  std::vector<KeyT> h_query(h_key);
  std::vector<ValueT> cmap_results(num_queries);

  // searching for the queries:
  cmap.hash_search_bulk(h_query.data(), cmap_results.data(), num_queries);

  // validating the results:
  std::unordered_map<KeyT, ValueT> hash_map;
  for (uint32_t i_key = 0; i_key < num_keys; i_key++) {
    hash_map.insert(std::make_pair(h_key[i_key], h_value[i_key]));
  }
  for (uint32_t i_key = 0; i_key < num_deletion; i_key++) {
    hash_map.erase(h_deleted_keys[i_key]);
  }

  for (uint32_t i = 0; i < num_queries; i++) {
    auto cmap_result = cmap_results[i];
    auto expected_result_it = hash_map.find(h_query[i]);
    auto expected_result = expected_result_it == hash_map.end()
                               ? SEARCH_NOT_FOUND
                               : expected_result_it->second;
    ASSERT_EQ(expected_result, cmap_result);
  }
}

int main(int argc, char** argv) {
  if (cmdOptionExists(argv, argc + argv, "-device")) {
    g_gpu_device_idx = atoi(getCmdOption(argv, argv + argc, "-device"));
  }

  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}