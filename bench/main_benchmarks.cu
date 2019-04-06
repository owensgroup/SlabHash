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

#include <cstdlib>
#include <unistd.h>
#include <stdio.h>
#include <iostream> 
#include <algorithm>
#include <ctime>
#include <iomanip> 
#include <string>
#include <sstream> 

#define DEVICE_ID 0 // todo: change this into a paramater 

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
  int mode = 0;
  int num_iter = 1;
  int n_start = 20;  // num_keys_start = 1 << n_start;
  int n_end = 20;    // num_keys_start = 1 << n_start;
  int num_samples = 10;
  float d_steps = 0.1f;
  uint32_t num_keys = (1 << 22);
  uint32_t num_queries = num_keys;
  float alpha = 1.0f;
  // for mode 5:
  uint32_t buckets = 1;
  float a_update = 1.0f;
  float c_search = 0.0f;

  uint32_t init_batches = 1;

  std::string filename("");
  if (cmdOptionExists(argv, argc + argv, "-mode"))
    mode = atoi(getCmdOption(argv, argv + argc, "-mode"));
  if (cmdOptionExists(argv, argc + argv, "-buckets"))
    buckets = atoi(getCmdOption(argv, argv + argc, "-buckets"));
  if (cmdOptionExists(argv, argc + argv, "-iter"))
    num_iter = atoi(getCmdOption(argv, argv + argc, "-iter"));
  if (cmdOptionExists(argv, argc + argv, "-nStart")) {
    n_start = atoi(getCmdOption(argv, argv + argc, "-nStart"));
    // for mode 0:
    num_keys = (1 << n_start);
    num_queries = num_keys;
  }
  if (cmdOptionExists(argv, argc + argv, "-nEnd"))
    n_end = atoi(getCmdOption(argv, argv + argc, "-nEnd"));
  if (cmdOptionExists(argv, argc + argv, "-batch"))
    init_batches = atoi(getCmdOption(argv, argv + argc, "-batch"));
  if (cmdOptionExists(argv, argc + argv, "-nSample"))
    num_samples = atoi(getCmdOption(argv, argv + argc, "-nSample"));
  if (cmdOptionExists(argv, argc + argv, "-dStep"))
    d_steps = atof(getCmdOption(argv, argv + argc, "-dStep"));
  if (cmdOptionExists(argv, argc + argv, "-alpha"))
    alpha = atof(getCmdOption(argv, argv + argc, "-alpha"));
  if (cmdOptionExists(argv, argc + argv, "-update"))
    a_update = float(atoi(getCmdOption(argv, argv + argc, "-update"))) / 100.0f;
  if (cmdOptionExists(argv, argc + argv, "-search"))
    c_search = float(atoi(getCmdOption(argv, argv + argc, "-search"))) / 100.0f;

  // input argument for the file to be used for storing the results
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
    cudaSetDevice(DEVICE_ID);  // be changed later
    cudaGetDeviceProperties(&devProp, DEVICE_ID);
  }
  printf("Device: %s\n", devProp.name);
  
  // running the actual experiment 
  load_factor_bulk_experiment<uint32_t, uint32_t>(num_keys, num_queries, filename, 0, 10, 0.1f);
}