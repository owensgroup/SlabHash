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

#include <cassert>
#include <iostream>
#include <random>
#include <typeinfo>

// global declarations
#include "slab_hash_global.cuh"

// global helper methods:
#include "slab_hash_helper_methods.cuh"

// class declaration:
#include "concurrent_map/cmap_class.cuh"
#include "concurrent_set/cset_class.cuh"
#include "slab_iterator.cuh"

// warp implementations of member functions:
#include "concurrent_map/warp/delete.cuh"
#include "concurrent_map/warp/insert.cuh"
#include "concurrent_map/warp/search.cuh"
#include "concurrent_map/warp/count.cuh"

#include "concurrent_set/cset_warp_operations.cuh"

// helper kernels:
#include "concurrent_map/device/build.cuh"
#include "concurrent_map/device/concurrent_kernel.cuh"
#include "concurrent_map/device/delete_kernel.cuh"
#include "concurrent_map/device/misc_kernels.cuh"
#include "concurrent_map/device/search_kernel.cuh"
#include "concurrent_map/device/count_kernel.cuh"
#include "concurrent_set/cset_helper_kernels.cuh"

// implementations:
#include "concurrent_map/cmap_implementation.cuh"
#include "concurrent_set/cset_implementation.cuh"