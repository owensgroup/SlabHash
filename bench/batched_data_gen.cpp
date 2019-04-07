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

#include "batched_data_gen.h"

batched_bench::batched_bench(uint32_t num_ref, uint32_t batch_size)
    : num_insert_(0),
      num_delete_(0),
      num_search_exist_(0),
      num_search_non_exist_(0),
      edge_index_(0),
      batch_counter_(0) {
  num_ref_ = num_ref;
  batch_size_ = batch_size;
  h_key_ref_ = new uint32_t[num_ref_];
  h_index_ref_ = new uint32_t[num_ref_];
  std::iota(h_index_ref_, h_index_ref_ + num_ref_, 0);
  h_batch_buffer_ = new uint32_t[batch_size_];
  temp_buffer_ = new uint32_t[batch_size_];
}

batched_bench::~batched_bench() {
  if (h_key_ref_)
    delete[] h_key_ref_;
  if (h_index_ref_)
    delete[] h_index_ref_;
  if (h_batch_buffer_)
    delete[] h_batch_buffer_;
  if (temp_buffer_)
    delete[] temp_buffer_;
}

void batched_bench::shuffle(uint32_t* input, uint32_t size) {
  for (int i = 0; i < size; i++) {
    unsigned int rand1 = rand();
    unsigned int rand2 = (rand() << 15) + rand1;
    unsigned int swap = i + (rand2 % (size - i));

    unsigned int temp = input[i];
    input[i] = input[swap];
    input[swap] = temp;
  }
}

void batched_bench::shuffle_pairs(uint32_t* input,
                                  uint32_t* values,
                                  uint32_t size) {
  for (int i = 0; i < size; i++) {
    unsigned int rand1 = rand();
    unsigned int rand2 = (rand() << 15) + rand1;
    unsigned int swap = i + (rand2 % (size - i));

    unsigned int temp = input[i];
    input[i] = input[swap];
    input[swap] = temp;

    temp = values[i];
    values[i] = values[swap];
    values[swap] = temp;
  }
}

void batched_bench::generate_random_keys() {
  std::iota(h_key_ref_, h_key_ref_ + num_ref_, 0);
  // shuffle(h_key_ref_, num_ref_);
  std::random_shuffle(h_key_ref_, h_key_ref_ + num_ref_);
}

void batched_bench::generate_random_keys(int seed, int num_msb) {
  std::mt19937 rng(seed);
  for (int i = 0; i < num_ref_; i++) {
    h_key_ref_[i] =
        (rng() &
         (0xFFFFFFFF >> num_msb));  // except for the most significant two bits
  }
}

uint32_t batched_bench::get_edge_index() {
  return edge_index_;
}

void batched_bench::set_edge_index(uint32_t new_edge_index) {
  if (new_edge_index < num_ref_)
    edge_index_ = new_edge_index;
}

void batched_bench::compute_batch_contents(float a_insert,
                                           float b_delete,
                                           float c_search_exist) {
  assert(a_insert + b_delete + c_search_exist <= 1.0f);
  num_insert_ = static_cast<uint32_t>(a_insert * batch_size_);
  num_delete_ = static_cast<uint32_t>(b_delete * batch_size_);
  num_search_exist_ = static_cast<uint32_t>(c_search_exist * batch_size_);
  num_search_non_exist_ =
      batch_size_ - (num_insert_ + num_delete_ + num_search_exist_);
}

uint32_t* batched_bench::next_batch(float a_insert,
                                    float b_delete,
                                    float c_search_exist) {
  compute_batch_contents(a_insert, b_delete, c_search_exist);

  std::random_shuffle(h_index_ref_, h_index_ref_ + edge_index_);
  std::random_shuffle(h_index_ref_ + edge_index_, h_index_ref_ + num_ref_);

  uint32_t output_offset = 0;

  // search queries that actually exist in the data structure
  // choosing the first num_search_exist_ from the beginning of the references:
  // code 3 for search queries
  for (int i = 0; i < num_search_exist_; i++) {
    h_batch_buffer_[output_offset + i] =
        (0xC0000000 | h_key_ref_[h_index_ref_[i]]);
  }
  output_offset += num_search_exist_;

  // search queries that do not exist in the data structure
  // choose the last num_search_non_exist_ from the end of the references:
  // code 3 for search queries
  for (int i = 0; i < num_search_non_exist_; i++) {
    h_batch_buffer_[output_offset + i] =
        (0xC0000000 | h_key_ref_[h_index_ref_[num_ref_ - i - 1]]);
  }
  output_offset += num_search_non_exist_;

  // inserting new items:
  // code 1:
  // the first num_isnert_ elements after the edge:
  for (int i = 0; i < num_insert_; i++) {
    temp_buffer_[i] = h_index_ref_[edge_index_ + i];
    h_batch_buffer_[output_offset + i] =
        (0x40000000 | h_key_ref_[temp_buffer_[i]]);
  }
  output_offset += num_insert_;

  // deleting previously inserted elements:
  // code 2:
  for (int i = 0; i < num_delete_; i++) {
    temp_buffer_[num_insert_ + i] = h_index_ref_[edge_index_ - i - 1];
    h_batch_buffer_[output_offset + i] =
        (0x80000000 | h_key_ref_[temp_buffer_[num_insert_ + i]]);
  }

  // shuffling the output buffer:
  std::random_shuffle(h_batch_buffer_, h_batch_buffer_ + batch_size_);

  // updating the edge index:
  std::copy(temp_buffer_, temp_buffer_ + batch_size_,
            h_index_ref_ + edge_index_ - num_delete_);
  edge_index_ += (num_insert_ - num_delete_);

  batch_counter_++;
  return h_batch_buffer_;
}

void batched_bench::print_batch() {
  printf("Batch %d:\n", batch_counter_);
  for (int i = 0; i < batch_size_; i++) {
    printf("(%d, %d), ", h_batch_buffer_[i] >> 30,
           h_batch_buffer_[i] & 0x3FFFFFFF);
    if (i % 10 == 9)
      printf("\n");
  }
  printf("\n");
}

void batched_bench::print_reference() {
  printf("Reference keys:");
  for (int i = 0; i < num_ref_; i++) {
    printf("%d, ", h_key_ref_[i]);
    if (i % 16 == 31)
      printf("\n");
  }
  printf("\n");
}