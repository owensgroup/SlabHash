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
#include <unistd.h>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include "experiments.cuh"
//
inline char* getCmdOption(char** begin, char** end, const std::string& option) {
  char** itr = std::find(begin, end, option);
  if (itr != end && ++itr != end) {
    return *itr;
  }
  return 0;
}
//=======================
inline bool cmdOptionExists(char** begin,
                            char** end,
                            const std::string& option) {
  return std::find(begin, end, option) != end;
}

int main(int argc, char** argv) {
  int mode = 0;  // type of experiment
  uint32_t num_iter = 1;
  bool verbose = false;
  int device_idx = 0;
  uint32_t num_keys = (1 << 22);
  uint32_t n_start = 20;  // num_keys = 1 << n_start;
  uint32_t n_end = 20;
  uint32_t num_queries = num_keys;
  float expected_chain = 0.6f;
  float existing_ratio = 1.0f;

  // mode 1 parameters:
  float lf_bulk_step = 0.1f;
  uint32_t lf_bulk_num_sample = 10;

  // mode 3 parameters:
  int num_batch = 2;
  int init_batch = 1;
  float insert_ratio = 0.1f;
  float delete_ratio = 0.1f;
  float search_exist_ratio = 0.4f;
  float lf_conc_step = 0.1f;
  int lf_conc_num_sample = 10;

  if (cmdOptionExists(argv, argc + argv, "-mode"))
    mode = atoi(getCmdOption(argv, argv + argc, "-mode"));
  if (cmdOptionExists(argv, argc + argv, "-num_key"))
    num_keys = atoi(getCmdOption(argv, argv + argc, "-num_key"));
  if (cmdOptionExists(argv, argc + argv, "-num_query"))
    num_queries = atoi(getCmdOption(argv, argv + argc, "-num_query"));
  else {
    num_queries = num_keys;
  }

  if (cmdOptionExists(argv, argc + argv, "-expected_chain"))
    expected_chain = atof(getCmdOption(argv, argv + argc, "-expected_chain"));
  assert(expected_chain > 0);
  if (cmdOptionExists(argv, argc + argv, "-query_ratio"))
    existing_ratio = atof(getCmdOption(argv, argv + argc, "-query_ratio"));
  if (cmdOptionExists(argv, argc + argv, "-verbose")) {
    verbose = (atoi(getCmdOption(argv, argv + argc, "-verbose")) != 0) ? true : false;
  }

  if (cmdOptionExists(argv, argc + argv, "-device"))
    device_idx = atoi(getCmdOption(argv, argv + argc, "-device"));
  if (cmdOptionExists(argv, argc + argv, "-iter")) {
    num_iter = atoi(getCmdOption(argv, argv + argc, "-iter"));
  }
  if (cmdOptionExists(argv, argc + argv, "-nStart")) {
    n_start = atoi(getCmdOption(argv, argv + argc, "-nStart"));
    // for mode 0:
    num_keys = (1 << n_start);
    num_queries = num_keys;
  }
  if (cmdOptionExists(argv, argc + argv, "-nEnd")) {
    n_end = atoi(getCmdOption(argv, argv + argc, "-nEnd"));
  }
  if (cmdOptionExists(argv, argc + argv, "-num_batch")) {
    num_batch = atoi(getCmdOption(argv, argv + argc, "-num_batch"));
  }
  if (cmdOptionExists(argv, argc + argv, "-init_batch")) {
    init_batch = atoi(getCmdOption(argv, argv + argc, "-init_batch"));
  }
  if (cmdOptionExists(argv, argc + argv, "-insert_ratio"))
    insert_ratio = atof(getCmdOption(argv, argv + argc, "-insert_ratio"));
  if (cmdOptionExists(argv, argc + argv, "-delete_ratio"))
    delete_ratio = atof(getCmdOption(argv, argv + argc, "-delete_ratio"));
  if (cmdOptionExists(argv, argc + argv, "-search_exist_ratio"))
    search_exist_ratio =
        atof(getCmdOption(argv, argv + argc, "-search_exist_ratio"));
  if (cmdOptionExists(argv, argc + argv, "-lf_conc_step"))
    lf_conc_step = atof(getCmdOption(argv, argv + argc, "-lf_conc_step"));
  if (cmdOptionExists(argv, argc + argv, "-lf_conc_num_sample"))
    lf_conc_num_sample =
        atoi(getCmdOption(argv, argv + argc, "-lf_conc_num_sample"));
  if (cmdOptionExists(argv, argc + argv, "-lf_bulk_step"))
    lf_bulk_step = atof(getCmdOption(argv, argv + argc, "-lf_bulk_step"));
  if (cmdOptionExists(argv, argc + argv, "-lf_bulk_num_sample"))
    lf_bulk_num_sample =
        atoi(getCmdOption(argv, argv + argc, "-lf_bulk_num_sample"));

  // input argument for the file to be used for storing the results
  std::string filename("");
  if (cmdOptionExists(argv, argc + argv, "-filename")) {
    filename.append(getCmdOption(argv, argv + argc, "-filename"));
    std::cout << filename << std::endl;
  } else {
    // setting the filename to be the current time:
    filename += "bench/";
    auto time = std::time(nullptr);
    auto tm = *std::localtime(&time);
    std::ostringstream temp;
    temp << std::put_time(&tm, "%d-%m-%Y_%H-%M-%S");
    filename += ("out_" + temp.str() + ".json");
  }

  //=========
  int devCount;
  cudaGetDeviceCount(&devCount);
  cudaDeviceProp devProp;
  if (devCount) {
    cudaSetDevice(device_idx);  // be changed later
    cudaGetDeviceProperties(&devProp, device_idx);
  }
  printf("Device: %s\n", devProp.name);
  printf("Experiment mode = %d\n", mode);

  using KeyT = uint32_t;
  using ValueT = uint32_t;

  // running the actual experiment
  switch (mode) {
    case 0:  // singleton experiment
      singleton_experiment<KeyT, ValueT>(num_keys, num_queries, expected_chain,
                                         filename, device_idx, existing_ratio,
                                         num_iter,
                                         /*run_cudpp = */ false, verbose);
      break;
    case 1:  // bulk build, num elements fixed, load factor changing
      load_factor_bulk_experiment<KeyT, ValueT>(
          num_keys, num_queries, filename, device_idx, existing_ratio, num_iter,
          false, lf_bulk_num_sample, lf_bulk_step);
      break;
    case 2:  // bulk build, load factor fixed, num elements changing
      build_search_bulk_experiment<KeyT, ValueT>(
          1 << n_start, 1 << n_end, filename, expected_chain, existing_ratio,
          device_idx, num_iter,
          /* run_cudpp = */ false,
          /* verbose = */ verbose);
      break;
    case 3:  // concurrent experiment:
      concurrent_batched_op_load_factor_experiment<KeyT, ValueT>(
          /*max_num_keys = */ 1 << n_end, /*batch_size = */ 1 << n_start,
          num_batch, init_batch, insert_ratio, delete_ratio, search_exist_ratio,
          filename, device_idx, lf_conc_step, lf_conc_num_sample, num_iter,
          verbose);
      break;
    default:
      std::cout << "Error: invalid mode." << std::endl;
      break;
  }
}