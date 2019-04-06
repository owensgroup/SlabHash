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

#include "gpu_hash_table.cuh"
#include "rapidjson/document.h"
#include "rapidjson/ostreamwrapper.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"
using namespace rapidjson;
// ======= Part 1: bulk experiments

/*
 * In this experiment, we compute the bulk build time versus different load
 * factors.
 */
template <typename KeyT, typename ValueT>
void load_factor_bulk_experiment(uint32_t num_keys,
                                 uint32_t num_queries,
                                 std::string filename,
                                 uint32_t algmode = 0,
                                 int num_sample_lf = 10,
                                 float steps = 0.1f) {
  Document doc;
  // doc.Parse(json);
  doc.SetObject();

  // Value object(kObjectType);
  // object.AddMember("Math", Value().SetInt(40), doc.GetAllocator());
  // object.AddMember("Science", "70", doc.GetAllocator());
  // object.AddMember("English", "50", doc.GetAllocator());
  // object.AddMember("Social Science", "70", doc.GetAllocator());
  // doc.AddMember("Marks", object, doc.GetAllocator());

  // doc.Accept(writer);
  // std::cout << s.GetString() << std::endl;

  printf(
      "hash,algmode,num_keys,num_queries,load_factor,chain,build_time,"
      "build_rate,query_ratio,search_time,search_rate\n");

  uint32_t* h_key = new uint32_t[num_keys + num_queries];
  uint32_t* h_value = new uint32_t[num_keys + num_queries];
  uint32_t* h_query = new uint32_t[num_queries];
  uint32_t* h_result = new uint32_t[num_queries];
  // uint32_t*	h_correct_result	= new uint32_t[num_queries];

  const uint32_t num_elements_per_unit = 15;
  // my warp allocator initial size:

  // === generating random key-values
  // RandomSequenceOfUnique myRNG(rand(), rand());
  // todo : remember to change this part
  for (int i = 0; i < (num_keys + num_queries); i++) {
    h_key[i] = i;
    h_value[i] = h_key[i];
  }

  std::vector<float> expected_chain_list(num_sample_lf);
  expected_chain_list[0] = 0.1f;
  for (int i = 1; i < num_sample_lf; i++)
    expected_chain_list[i] = expected_chain_list[i - 1] + steps;

  std::vector<float> query_ratio_list{0.0f, 1.0f};

  uint32_t experiment_id = 0;

  for (int i_expected_chain = 0; i_expected_chain < num_sample_lf;
       i_expected_chain++) {
    float expected_chain = expected_chain_list[i_expected_chain];
    uint32_t expected_elements_per_bucket =
        expected_chain * num_elements_per_unit;
    uint32_t num_buckets = (num_keys + expected_elements_per_bucket - 1) /
                           expected_elements_per_bucket;

    // building the hash table:
    gpu_hash_table<KeyT, ValueT, DEVICE_ID, SlabHashTypeT::ConcurrentMap>
        hash_table(num_keys, num_buckets, time(NULL));

    float build_time = hash_table.hash_build(h_key, h_value, num_keys);

    // my_hash_table::gpu_hash_table hash_table(num_keys, num_buckets,
    //                                          max_allocator_size);
    // float init_time = hash_table.init();
    // float build_time = hash_table.hash_build_bulk(h_key, h_value, num_keys);
    double load_factor = hash_table.measureLoadFactor();

    // ==== Running CUDPP hash table with the same load factor:
    // cudpp_hash_table cudpp_hash(h_key, h_value, num_keys, num_queries,
    //                             load_factor, false, false);
    // float cudpp_build_time = cudpp_hash.hash_build();

    printf("chain = %.2f, lf = %.2f, query ratios: : \n", expected_chain,
           load_factor);
    for (auto query_ratio : query_ratio_list) {
      printf("%.1f, ", query_ratio);
      // === generating random queries with a fixed ratio existing in keys
      uint32_t num_existing = static_cast<uint32_t>(query_ratio * num_queries);

      for (int i = 0; i < num_existing; i++) {
        h_query[i] = h_key[num_keys - 1 - i];
        // h_correct_result[i] = h_query[i];
      }

      for (int i = 0; i < (num_queries - num_existing); i++) {
        h_query[num_existing + i] = h_key[num_keys + i];
        // h_correct_result[num_existing + i] = SEARCH_NOT_FOUND;
      }
      // permuting the queries:
      // randomPermute(h_query, num_queries); // ==== removed

      // performing the queries:
      // float search_time_bulk =
      //     hash_table.hash_search_bulk(h_query, h_result, num_queries);
      float search_time =
          hash_table.hash_search(h_query, h_result, num_queries);      
      float search_time_bulk =
          hash_table.hash_search_bulk(h_query, h_result, num_queries);

      // // CUDPP hash table:
      // float cudpp_search_time =
      //     cudpp_hash.lookup_hash_table(h_query, num_queries);

      rapidjson::Value object(kObjectType);
      object.AddMember("num_keys", rapidjson::Value().SetInt(num_keys),
                       doc.GetAllocator());
      object.AddMember("num_queries", rapidjson::Value().SetInt(num_queries),
                       doc.GetAllocator());
      object.AddMember("build_time_ms",
                       rapidjson::Value().SetDouble(build_time),
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
      std::string key_name(std::to_string(experiment_id++));
      doc.AddMember(rapidjson::Value().SetString(
                        key_name.c_str(), key_name.size(), doc.GetAllocator()),
                    object, doc.GetAllocator());

      printf("SlabHash,%d,%d,%d,%.2f,%.2f,%.3f,%.3f,%.2f,%.3f,%.3f\n", algmode,
             num_keys, num_queries, load_factor, expected_chain, build_time,
             double(num_keys) / build_time / 1000.0, query_ratio,
             search_time_bulk, double(num_queries) / search_time_bulk / 1000.0);
      // fprintf(fptr, "CUDPP,%d,%d,%d,%.2f,%.2f,%.3f,%.3f,%.2f,%.3f,%.3f\n",
      //         algmode, num_keys, num_queries, load_factor, expected_chain,
      //         cudpp_build_time, double(num_keys) / cudpp_build_time /
      //         1000.0, query_ratio, cudpp_search_time, double(num_queries) /
      //         cudpp_search_time / 1000.0);
    }
    // printf(" ==> Build: %.2f M elements/s, CUDPP: %.2f M elements/s\n",
    //        double(num_keys) / build_time / 1000.0,
    //        double(num_keys) / cudpp_build_time / 1000.0);
    printf(" ==> Build: %.2f M elements/s\n",
           double(num_keys) / build_time / 1000.0);
  }

  //  rapidjson::StringBuffer s;
  //  rapidjson::Writer<rapidjson::StringBuffer> writer(s);
  // doc.Accept(writer);
  // std::cout << s.GetString() << std::endl;
  std::cout << " ============= " << filename << std::endl;
  // const std::string base_name("bench/");
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