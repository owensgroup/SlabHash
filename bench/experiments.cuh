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
#include <fstream>
#include <iostream>

#include "batched_data_gen.h"
#include "gpu_hash_table.cuh"
#include "rapidjson/document.h"
#include "rapidjson/ostreamwrapper.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"

// ======= Part 1: bulk experiments

/*
 * In this experiment, we compute the bulk build time versus different load
 * factors.
 */
template <typename KeyT, typename ValueT>
void load_factor_bulk_experiment(uint32_t num_keys,
                                 uint32_t num_queries,
                                 std::string filename,
                                 uint32_t device_idx,
                                 float query_ratio,
                                 uint32_t num_iter = 1,
                                 bool run_cudpp = false,
                                 int num_sample_lf = 10,
                                 float steps = 0.1f,
                                 bool verbose = false) {
  rapidjson::Document doc;
  doc.SetObject();

  int devCount;
  cudaGetDeviceCount(&devCount);
  cudaDeviceProp devProp;
  if (devCount) {
    cudaSetDevice(device_idx);
    cudaGetDeviceProperties(&devProp, device_idx);
  }

  rapidjson::Value main_object(rapidjson::kObjectType);
  main_object.AddMember(
      "device_name",
      rapidjson::Value().SetString(devProp.name, 20, doc.GetAllocator()),
      doc.GetAllocator());
  rapidjson::Value object_array(rapidjson::kArrayType);

  uint32_t* h_key = new uint32_t[num_keys + num_queries];
  uint32_t* h_value = new uint32_t[num_keys + num_queries];
  uint32_t* h_query = new uint32_t[num_queries];
  uint32_t* h_result = new uint32_t[num_queries];

  const uint32_t num_elements_per_unit = 15;
  // === generating random key-values
  BatchedDataGen key_gen(num_keys + num_queries, num_keys + num_queries);
  key_gen.generate_random_keys(std::time(nullptr), /*num_msb = */ 0, true);
  auto f = [](uint32_t key) { return ~key; };

  std::vector<float> expected_chain_list(num_sample_lf);
  expected_chain_list[0] = 0.1f;
  for (int i = 1; i < num_sample_lf; i++) {
    expected_chain_list[i] = expected_chain_list[i - 1] + steps;
  }

  // query ratios to be tested against (fraction of queries that actually exist
  // in the data structure):

  uint32_t experiment_id = 0;

  for (int i_expected_chain = 0; i_expected_chain < num_sample_lf;
       i_expected_chain++) {
    float expected_chain = expected_chain_list[i_expected_chain];
    uint32_t expected_elements_per_bucket =
        expected_chain * num_elements_per_unit;
    uint32_t num_buckets = (num_keys + expected_elements_per_bucket - 1) /
                           expected_elements_per_bucket;

    // ==== Running CUDPP hash table with the same load factor:
    if (run_cudpp) {
      // cudpp_hash_table cudpp_hash(h_key, h_value, num_keys, num_queries,
      //                             load_factor, false, false);
      // float cudpp_build_time = cudpp_hash.hash_build();
    }

    // === generating random queries with a fixed ratio existing in keys
    uint32_t num_existing = static_cast<uint32_t>(query_ratio * num_queries);
    auto buffer_ptr =
        key_gen.getSingleBatchPointer(num_keys, num_queries, num_existing);

    for (int i = 0; i < num_keys; i++) {
      h_key[i] = buffer_ptr[i];
      h_value[i] = f(h_key[i]);
    }

    for (int i = 0; i < num_queries; i++) {
      h_query[i] = buffer_ptr[num_keys + i];
    }

    float build_time = 0.0f;
    float search_time = 0.0f;
    float search_time_bulk = 0.0f;
    double load_factor = 0.0;
    for (int iter = 0; iter < num_iter; iter++) {
      // building the hash table:
      gpu_hash_table<KeyT, ValueT, SlabHashTypeT::ConcurrentMap> hash_table(
          num_keys, num_buckets, device_idx, time(nullptr));

      build_time += hash_table.hash_build(h_key, h_value, num_keys);

      // measuring the exact load factor in slab hash
      load_factor += hash_table.measureLoadFactor();

      if (verbose) {
        printf(
            "%d: num_element = %u, load_factor = %.2f, query_ratio = %.2f, "
            "build_rate: %.2f M elements/s\n",
            experiment_id, num_keys, load_factor, query_ratio,
            double(num_keys) / build_time / 1000.0);
      }
      // performing the queries:
      search_time += hash_table.hash_search(h_query, h_result, num_queries);
      search_time_bulk +=
          hash_table.hash_search_bulk(h_query, h_result, num_queries);

      // CUDPP hash table:
      if (run_cudpp) {
        // float cudpp_search_time =
        //     cudpp_hash.lookup_hash_table(h_query, num_queries);
      }
    }
    build_time /= num_iter;
    search_time /= num_iter;
    search_time_bulk /= num_iter;
    load_factor /= num_iter;

    rapidjson::Value object(rapidjson::kObjectType);
    object.AddMember("id", rapidjson::Value().SetInt(experiment_id++),
                     doc.GetAllocator());
    object.AddMember("num_keys", rapidjson::Value().SetInt(num_keys),
                     doc.GetAllocator());
    object.AddMember("num_buckets", rapidjson::Value().SetInt(num_buckets),
                     doc.GetAllocator());
    object.AddMember("num_queries", rapidjson::Value().SetInt(num_queries),
                     doc.GetAllocator());
    object.AddMember("build_time_ms", rapidjson::Value().SetDouble(build_time),
                     doc.GetAllocator());
    object.AddMember(
        "build_rate_mps",
        rapidjson::Value().SetDouble(double(num_keys) / build_time / 1000.0),
        doc.GetAllocator());
    object.AddMember("search_time_bulk_ms",
                     rapidjson::Value().SetDouble(search_time_bulk),
                     doc.GetAllocator());
    object.AddMember("search_rate_bulk_mps",
                     rapidjson::Value().SetDouble(double(num_queries) /
                                                  search_time_bulk / 1000.0),
                     doc.GetAllocator());
    object.AddMember("search_time_ms",
                     rapidjson::Value().SetDouble(search_time),
                     doc.GetAllocator());
    object.AddMember("search_rate_mps",
                     rapidjson::Value().SetDouble(double(num_queries) /
                                                  search_time / 1000.0),
                     doc.GetAllocator());
    object.AddMember("query_ratio", rapidjson::Value().SetDouble(query_ratio),
                     doc.GetAllocator());
    object.AddMember("load_factor", rapidjson::Value().SetDouble(load_factor),
                     doc.GetAllocator());
    object.AddMember("exp_chain_length",
                     rapidjson::Value().SetDouble(expected_chain),
                     doc.GetAllocator());
    object_array.PushBack(object, doc.GetAllocator());
  }

  // adding the array of objects into the document:
  main_object.AddMember("trial", object_array, doc.GetAllocator());
  doc.AddMember("slab_hash", main_object, doc.GetAllocator());
  // writing back the results as a json file
  std::ofstream ofs(filename);
  rapidjson::OStreamWrapper osw(ofs);
  rapidjson::Writer<rapidjson::OStreamWrapper> writer(osw);

  doc.Accept(writer);

  if (h_key)
    delete[] h_key;
  if (h_value)
    delete[] h_value;
  if (h_query)
    delete[] h_query;
  if (h_result)
    delete[] h_result;
}

/*
* In this experiment, a single experiment is performed:
  Inputs: number of elements, expected chain length (# buckets)
*/
template <typename KeyT, typename ValueT>
void singleton_experiment(uint32_t num_keys,
                          uint32_t num_queries,
                          float expected_chain_length,
                          std::string filename,
                          uint32_t device_idx,
                          float query_ratio,
                          uint32_t num_iter,
                          bool run_cudpp = false,
                          bool verbose = false) {
  rapidjson::Document doc;
  doc.SetObject();

  int devCount;
  cudaGetDeviceCount(&devCount);
  cudaDeviceProp devProp;
  if (devCount) {
    cudaSetDevice(device_idx);
    cudaGetDeviceProperties(&devProp, device_idx);
  }

  rapidjson::Value main_object(rapidjson::kObjectType);
  main_object.AddMember(
      "device_name",
      rapidjson::Value().SetString(devProp.name, 20, doc.GetAllocator()),
      doc.GetAllocator());
  rapidjson::Value object_array(rapidjson::kArrayType);

  KeyT* h_key = new KeyT[num_keys + num_queries];
  ValueT* h_value = new ValueT[num_keys + num_queries];
  KeyT* h_query = new KeyT[num_queries];
  ValueT* h_result = new ValueT[num_queries];

  uint32_t experiment_id = 0;
  const uint32_t num_elements_per_unit = 15;
  // === generating random key-values
  BatchedDataGen key_gen(num_keys + num_queries, num_keys + num_queries);
  key_gen.generate_random_keys(std::time(nullptr), /*num_msb = */ 0, true);
  auto f = [](uint32_t key) { return ~key; };

  const uint32_t expected_elements_per_bucket =
      expected_chain_length * num_elements_per_unit;
  const uint32_t num_buckets = (num_keys + expected_elements_per_bucket - 1) /
                               expected_elements_per_bucket;

  // === generating random queries with a fixed ratio existing in keys
  const uint32_t num_existing =
      static_cast<uint32_t>(query_ratio * num_queries);
  auto buffer_ptr =
      key_gen.getSingleBatchPointer(num_keys, num_queries, num_existing);

  for (int i = 0; i < num_keys; i++) {
    h_key[i] = buffer_ptr[i];
    h_value[i] = f(h_key[i]);
  }

  for (int i = 0; i < num_queries; i++) {
    h_query[i] = buffer_ptr[num_keys + i];
  }

  float build_time = 0.0f;
  float search_time = 0.0f;
  float search_time_bulk = 0.0f;
  double load_factor = 0.0;
  for (int iter = 0; iter < num_iter; iter++) {
    // building the hash table:
    gpu_hash_table<KeyT, ValueT, SlabHashTypeT::ConcurrentMap> hash_table(
        num_keys, num_buckets, device_idx, time(nullptr));

    build_time += hash_table.hash_build(h_key, h_value, num_keys);

    // measuring the exact load factor in slab hash
    load_factor += hash_table.measureLoadFactor();

    // performing the queries:
    search_time += hash_table.hash_search(h_query, h_result, num_queries);
    search_time_bulk +=
        hash_table.hash_search_bulk(h_query, h_result, num_queries);

    // CUDPP hash table:
    if (run_cudpp) {
      // float cudpp_search_time =
      //     cudpp_hash.lookup_hash_table(h_query, num_queries);
    }
  }
  build_time /= num_iter;
  search_time /= num_iter;
  search_time_bulk /= num_iter;
  load_factor /= num_iter;

  if (verbose) {
    printf(
        "num_element = %u, load_factor = %.2f, query_ratio = %.2f, num_buckets "
        "= %d\nbuild_rate: %.2f M elements/s\nsearch rate = %.2f M "
        "queries/s\nbulk search rate = %.2f M queries/s\n",
        num_keys, load_factor, query_ratio, num_buckets,
        double(num_keys) / build_time / 1000.0,
        double(num_queries) / search_time / 1000.0,
        double(num_queries) / search_time_bulk / 1000.0);
  }

  rapidjson::Value object(rapidjson::kObjectType);
  object.AddMember("id", rapidjson::Value().SetInt(experiment_id++),
                   doc.GetAllocator());
  object.AddMember("num_keys", rapidjson::Value().SetInt(num_keys),
                   doc.GetAllocator());
  object.AddMember("num_buckets", rapidjson::Value().SetInt(num_buckets),
                   doc.GetAllocator());
  object.AddMember("num_queries", rapidjson::Value().SetInt(num_queries),
                   doc.GetAllocator());
  object.AddMember("build_time_ms", rapidjson::Value().SetDouble(build_time),
                   doc.GetAllocator());
  object.AddMember(
      "build_rate_mps",
      rapidjson::Value().SetDouble(double(num_keys) / build_time / 1000.0),
      doc.GetAllocator());
  object.AddMember("search_time_bulk_ms",
                   rapidjson::Value().SetDouble(search_time_bulk),
                   doc.GetAllocator());
  object.AddMember("search_rate_bulk_mps",
                   rapidjson::Value().SetDouble(double(num_queries) /
                                                search_time_bulk / 1000.0),
                   doc.GetAllocator());
  object.AddMember("search_time_ms", rapidjson::Value().SetDouble(search_time),
                   doc.GetAllocator());
  object.AddMember(
      "search_rate_mps",
      rapidjson::Value().SetDouble(double(num_queries) / search_time / 1000.0),
      doc.GetAllocator());
  object.AddMember("query_ratio", rapidjson::Value().SetDouble(query_ratio),
                   doc.GetAllocator());
  object.AddMember("load_factor", rapidjson::Value().SetDouble(load_factor),
                   doc.GetAllocator());
  object.AddMember("exp_chain_length",
                   rapidjson::Value().SetDouble(expected_chain_length),
                   doc.GetAllocator());
  object_array.PushBack(object, doc.GetAllocator());

  // adding the array of objects into the document:
  main_object.AddMember("trial", object_array, doc.GetAllocator());
  doc.AddMember("slab_hash", main_object, doc.GetAllocator());
  // writing back the results as a json file
  std::ofstream ofs(filename);
  rapidjson::OStreamWrapper osw(ofs);
  rapidjson::Writer<rapidjson::OStreamWrapper> writer(osw);

  doc.Accept(writer);

  if (h_key)
    delete[] h_key;
  if (h_value)
    delete[] h_value;
  if (h_query)
    delete[] h_query;
  if (h_result)
    delete[] h_result;
}

/*
 * In this experiment, we assume that the expected chain length is fixed, number
 * of elements within the data structure changes.
 */
template <typename KeyT, typename ValueT>
void build_search_bulk_experiment(uint32_t num_keys_start,
                                  uint32_t num_keys_end,
                                  std::string filename,
                                  float expected_chain,
                                  float existing_ratio,
                                  uint32_t device_idx = 0,
                                  int num_iter = 1,
                                  bool run_cudpp = false,
                                  bool verbose = false) {
  // computing the bulk build time versus different load factors (equivalently
  // expected chain lengths) defining the modes:

  rapidjson::Document doc;
  doc.SetObject();

  int devCount;
  cudaGetDeviceCount(&devCount);
  cudaDeviceProp devProp;
  if (devCount) {
    cudaSetDevice(device_idx);
    cudaGetDeviceProperties(&devProp, device_idx);
  }

  rapidjson::Value main_object(rapidjson::kObjectType);
  main_object.AddMember(
      "device_name",
      rapidjson::Value().SetString(devProp.name, 20, doc.GetAllocator()),
      doc.GetAllocator());
  rapidjson::Value object_array(rapidjson::kArrayType);

  uint32_t max_queries = num_keys_end;
  KeyT* h_key = new KeyT[2 * num_keys_end];
  ValueT* h_value = new ValueT[2 * num_keys_end];
  KeyT* h_query = new ValueT[max_queries];
  ValueT* h_result = new ValueT[max_queries];

  const uint32_t num_elements_per_unit = 15;  // todo: change this

  // === generating random key-values
  BatchedDataGen key_gen(num_keys_end + max_queries,
                         num_keys_end + max_queries);
  key_gen.generate_random_keys(std::time(nullptr), /*num_msb = */ 0,
                               /*unique = */ false);

  auto f = [](uint32_t key) { return ~key; };

  uint32_t experiment_id = 0;

  for (uint32_t num_keys = num_keys_start; num_keys <= num_keys_end;
       num_keys <<= 1) {
    uint32_t num_queries = num_keys;
    const uint32_t expected_elements_per_bucket =
        expected_chain * num_elements_per_unit;
    const uint32_t num_buckets = (num_keys + expected_elements_per_bucket - 1) /
                                 expected_elements_per_bucket;

    float build_time = 0.0f;
    // float cudpp_build_time = 0.0f;
    float search_time_bulk = 0.0f;
    float search_time = 0.0f;
    // float cudpp_search_time = 0.0f;
    double load_factor_total = 0.0;

    // === generating random queries with a fixed ratio existing in keys
    const uint32_t num_existing =
        static_cast<uint32_t>(existing_ratio * num_queries);
    auto buffer_ptr =
        key_gen.getSingleBatchPointer(num_keys, num_queries, num_existing);

    for (int i = 0; i < num_keys; i++) {
      h_key[i] = buffer_ptr[i];
      h_value[i] = f(h_key[i]);
    }

    for (int i = 0; i < num_queries; i++) {
      h_query[i] = buffer_ptr[num_keys + i];
    }

    for (int iter = 0; iter < num_iter; iter++) {
      // building the hash table:
      gpu_hash_table<KeyT, ValueT, SlabHashTypeT::ConcurrentMap> hash_table(
          num_keys, num_buckets, device_idx, time(nullptr));

      build_time += hash_table.hash_build(h_key, h_value, num_keys);

      // measuring the exact load factor in slab hash
      double load_factor = hash_table.measureLoadFactor();
      load_factor_total += load_factor;

      // ==== Running CUDPP hash table with the same load factor:
      // cudpp_hash_table cudpp_hash(h_key, h_value, num_keys, num_queries,
      //                             load_factor, false, false);
      // cudpp_build_time += cudpp_hash.hash_build();

      // performing the queries:
      // individual implementation:
      search_time += hash_table.hash_search(h_query, h_result, num_queries);

      // bulk implementation:
      search_time_bulk +=
          hash_table.hash_search_bulk(h_query, h_result, num_queries);

      // CUDPP hash table:
      // cudpp_search_time += cudpp_hash.lookup_hash_table(h_query,
      // num_queries);
    }
    // computing averages:
    build_time /= num_iter;
    // cudpp_build_time /= num_iter;
    // cudpp_search_time /= num_iter;
    search_time /= num_iter;
    search_time_bulk /= num_iter;
    load_factor_total /= num_iter;

    if (verbose) {
      printf("num_keys = %d, build_rate_mps = %.3f\n", num_keys,
             double(num_keys) / build_time / 1000.0);
    }
    // storing the results:
    rapidjson::Value object(rapidjson::kObjectType);
    object.AddMember("id", rapidjson::Value().SetInt(experiment_id++),
                     doc.GetAllocator());
    object.AddMember("iter", rapidjson::Value().SetInt(num_iter),
                     doc.GetAllocator());
    object.AddMember("num_keys", rapidjson::Value().SetInt(num_keys),
                     doc.GetAllocator());
    object.AddMember("num_buckets", rapidjson::Value().SetInt(num_buckets),
                     doc.GetAllocator());    
    object.AddMember("num_queries", rapidjson::Value().SetInt(num_queries),
                     doc.GetAllocator());
    object.AddMember("build_time_ms", rapidjson::Value().SetDouble(build_time),
                     doc.GetAllocator());
    object.AddMember(
        "build_rate_mps",
        rapidjson::Value().SetDouble(double(num_keys) / build_time / 1000.0),
        doc.GetAllocator());
    object.AddMember("search_time_bulk_ms",
                     rapidjson::Value().SetDouble(search_time_bulk),
                     doc.GetAllocator());
    object.AddMember("search_rate_bulk_mps",
                     rapidjson::Value().SetDouble(double(num_queries) /
                                                  search_time_bulk / 1000.0),
                     doc.GetAllocator());
    object.AddMember("search_time_ms",
                     rapidjson::Value().SetDouble(search_time),
                     doc.GetAllocator());
    object.AddMember("search_rate_mps",
                     rapidjson::Value().SetDouble(double(num_queries) /
                                                  search_time / 1000.0),
                     doc.GetAllocator());
    object.AddMember("query_ratio",
                     rapidjson::Value().SetDouble(existing_ratio),
                     doc.GetAllocator());
    object.AddMember("load_factor",
                     rapidjson::Value().SetDouble(load_factor_total),
                     doc.GetAllocator());
    object.AddMember("exp_chain_length",
                     rapidjson::Value().SetDouble(expected_chain),
                     doc.GetAllocator());
    object.AddMember("num_buckets", rapidjson::Value().SetDouble(num_buckets),
                     doc.GetAllocator());
    object_array.PushBack(object, doc.GetAllocator());
  }

  // adding the array of objects into the document:
  main_object.AddMember("trial", object_array, doc.GetAllocator());
  doc.AddMember("slab_hash", main_object, doc.GetAllocator());
  // writing back the results as a json file
  std::ofstream ofs(filename);
  rapidjson::OStreamWrapper osw(ofs);
  rapidjson::Writer<rapidjson::OStreamWrapper> writer(osw);

  doc.Accept(writer);

  if (h_key)
    delete[] h_key;
  if (h_value)
    delete[] h_value;
  if (h_query)
    delete[] h_query;
  if (h_result)
    delete[] h_result;
}

template <typename KeyT, typename ValueT>
void concurrent_batched_op_load_factor_experiment(uint32_t max_key_num,
                                                  uint32_t batch_size,
                                                  uint32_t num_batches,
                                                  uint32_t num_initial_batches,
                                                  float a_insert,
                                                  float b_delete,
                                                  float c_search_exist,
                                                  std::string filename,
                                                  uint32_t device_idx = 0,
                                                  float lf_steps = 0.1,
                                                  uint32_t num_sample_lf = 100,
                                                  int num_iter = 1,
                                                  bool verbose = false) {
  rapidjson::Document doc;
  doc.SetObject();

  int devCount;
  cudaGetDeviceCount(&devCount);
  cudaDeviceProp devProp;
  if (devCount) {
    cudaSetDevice(device_idx);
    cudaGetDeviceProperties(&devProp, device_idx);
  }

  rapidjson::Value main_object(rapidjson::kObjectType);
  main_object.AddMember(
      "device_name",
      rapidjson::Value().SetString(devProp.name, 20, doc.GetAllocator()),
      doc.GetAllocator());
  rapidjson::Value object_array(rapidjson::kArrayType);

  const uint32_t num_keys = num_batches * batch_size;
  uint32_t* h_result = new uint32_t[max_key_num];

  // generating different load factors
  std::vector<float> expected_chain_list(num_sample_lf);
  expected_chain_list[0] = 0.1f;
  for (int i = 1; i < num_sample_lf; i++) {
    expected_chain_list[i] = expected_chain_list[i - 1] + lf_steps;
  }

  const uint32_t num_elements_per_unit = 15;  // todo: change this

  uint32_t experiment_id = 0;

  for (int i_expected_chain = 0; i_expected_chain < num_sample_lf;
       i_expected_chain++) {
    float init_build_time = 0.0f;
    float concurrent_time = 0.0f;
    double init_load_factor = 0.0;
    double final_load_factor = 0.0;
    float expected_chain = expected_chain_list[i_expected_chain];
    uint32_t expected_elements_per_bucket =
        expected_chain * num_elements_per_unit;
    uint32_t num_buckets = (num_keys + expected_elements_per_bucket - 1) /
                           expected_elements_per_bucket;

    // building the hash table:
    for (int iter = 0; iter < num_iter; iter++) {
      // building the hash table:
      gpu_hash_table<KeyT, ValueT, SlabHashTypeT::ConcurrentMap> hash_table(
          num_keys, num_buckets, device_idx, time(nullptr));

      BatchedDataGen key_gen(max_key_num, batch_size);
      key_gen.generate_random_keys(std::time(nullptr), /*num_msb = */ 2,
                                   /*unique = */ true);

      // initial building phase:
      int batch_id;
      for (batch_id = 0; batch_id < num_initial_batches; batch_id++) {
        // Pure insertion
        key_gen.next_batch(1.0f, 0.0f, 0.0f);
        init_build_time += hash_table.batched_operations(
            key_gen.h_batch_buffer_, h_result, batch_size, batch_id);
      }
      // init_build_time /= num_initial_batches;
      init_load_factor += hash_table.measureLoadFactor();

      // concurrent update phase:
      for (; batch_id < num_batches; batch_id++) {
        key_gen.next_batch(a_insert, b_delete, c_search_exist);
        concurrent_time += hash_table.batched_operations(
            key_gen.h_batch_buffer_, h_result, batch_size, batch_id);
      }

      final_load_factor += hash_table.measureLoadFactor();
    }
    // === computing average values:
    init_build_time /= (num_iter * num_initial_batches);
    concurrent_time /= (num_iter * (num_batches - num_initial_batches));
    init_load_factor /= num_iter;
    final_load_factor /= num_iter;

    if (verbose) {
      printf(
          "expected = %.2f, init_load_factor = %.2f, final_load_factor = %.2f, "
          "concurrent_rate = %.3f M op/s\n",
          expected_chain, init_load_factor, final_load_factor,
          float(batch_size) / concurrent_time / 1000.0f);
    }

    // storing the results:
    rapidjson::Value object(rapidjson::kObjectType);
    object.AddMember("id", rapidjson::Value().SetInt(experiment_id++),
                     doc.GetAllocator());
    object.AddMember("num_keys", rapidjson::Value().SetInt(num_keys),
                     doc.GetAllocator());
    object.AddMember("num_buckets", rapidjson::Value().SetInt(num_buckets),
                     doc.GetAllocator());
    object.AddMember("batch_size", rapidjson::Value().SetInt(batch_size),
                     doc.GetAllocator());
    object.AddMember("num_batches", rapidjson::Value().SetInt(num_batches),
                     doc.GetAllocator());
    object.AddMember("num_init_batches",
                     rapidjson::Value().SetInt(num_initial_batches),
                     doc.GetAllocator());
    object.AddMember("init_load_factor",
                     rapidjson::Value().SetDouble(init_load_factor),
                     doc.GetAllocator());
    object.AddMember("final_load_factor",
                     rapidjson::Value().SetDouble(final_load_factor),
                     doc.GetAllocator());
    object.AddMember("insert_ratio", rapidjson::Value().SetDouble(a_insert),
                     doc.GetAllocator());
    object.AddMember("delete_ratio", rapidjson::Value().SetDouble(b_delete),
                     doc.GetAllocator());
    object.AddMember("search_exist_ratio",
                     rapidjson::Value().SetDouble(c_search_exist),
                     doc.GetAllocator());
    object.AddMember("search_non_exist_ratio",
                     rapidjson::Value().SetDouble(
                         1.0 - (a_insert + b_delete + c_search_exist)),
                     doc.GetAllocator());
    object.AddMember("initial_time_ms",
                     rapidjson::Value().SetDouble(init_build_time),
                     doc.GetAllocator());
    object.AddMember("initial_rate_mps",
                     rapidjson::Value().SetDouble(double(batch_size) /
                                                  init_build_time / 1000.0),
                     doc.GetAllocator());
    object.AddMember("concurrent_time_ms",
                     rapidjson::Value().SetDouble(concurrent_time),
                     doc.GetAllocator());
    object.AddMember("concurrent_rate_mps",
                     rapidjson::Value().SetDouble(double(batch_size) /
                                                  concurrent_time / 1000.0),
                     doc.GetAllocator());
    object_array.PushBack(object, doc.GetAllocator());
  }

  // adding the array of objects into the document:
  main_object.AddMember("trial", object_array, doc.GetAllocator());
  doc.AddMember("slab_hash", main_object, doc.GetAllocator());
  // writing back the results as a json file
  std::ofstream ofs(filename);
  rapidjson::OStreamWrapper osw(ofs);
  rapidjson::Writer<rapidjson::OStreamWrapper> writer(osw);

  doc.Accept(writer);

  if (h_result)
    delete[] h_result;
}