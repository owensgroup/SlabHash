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

#include <stdint.h>
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <numeric>
#include <random>
#include <unordered_set>

class BatchedDataGen {
 private:
  uint32_t* h_key_ref_;
  uint32_t* h_index_ref_;
  uint32_t num_ref_;
  uint32_t edge_index_;
  uint32_t* temp_buffer_;

  uint32_t batch_counter_;
  uint32_t num_insert_;
  uint32_t num_delete_;
  uint32_t num_search_exist_;
  uint32_t num_search_non_exist_;

 public:
  uint32_t batch_size_;
  uint32_t* h_batch_buffer_;

  BatchedDataGen(uint32_t num_ref_, uint32_t batch_size);
  ~BatchedDataGen();
  void shuffle(uint32_t* input, uint32_t size);
  void shuffle_pairs(uint32_t* input, uint32_t* values, uint32_t size);
  void generate_random_keys();
  void generate_random_keys(int seed, int num_msb, bool ensure_uniqueness);
  uint32_t* getSingleBatchPointer(uint32_t num_keys,
                                  uint32_t num_queries,
                                  uint32_t num_existing);
  uint32_t* getKeyRefPointer() { return h_key_ref_; }
  uint32_t get_edge_index();
  void set_edge_index(uint32_t new_edge_index);
  uint32_t* next_batch(float a_insert, float b_delete, float c_search_exist);
  uint32_t getBatchCounter() { return batch_counter_; }
  void print_batch();
  void print_reference();
  void compute_batch_contents(float a_insert,
                              float b_delete,
                              float c_search_exist);
};