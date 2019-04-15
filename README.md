# SlabHash
A warp-oriented dynamic hash table for GPUs

##Publication:
This library is based on the original slab hash paper, initially proposed in the following IPDPS'18 paper:
* [Saman Ashkiani, Martin Farach-Colton, John Owens, *A Dynamic Hash Table for the GPU*, 2018 IEEE International Parallel and Distributed Processing Symposium (IPDPS)](https://ieeexplore.ieee.org/abstract/document/8425196)

This library is a rafactored and slightly redesigned version of the original code, so that it can be extended and be used in other research projects as well. It is still under continuous development. If you find any problem with the code, or suggestions for potential additions to the library, we will appreciate it if you can raise issues on github. We will address them as soon as possible. 

##Compilation
1. `git submodule init`
2. `git submodule update`
3. Make sure to edit `CMakeLists.txt` such that it reflects the GPU device's compute capability. For example, to include compute 3.5 you should have `option(SLABHASH_GENCODE_SM35 "GENCODE_SM35" ON)`.
4. `mkdir build && cd build`
5. `cmake ..`
6. 'make'

##Simple benchmarking
A simplified set of benchmark scenarios are available through a Python API that can be used as follows: Once the code is successfully compiled you can run the following python code from the `build` directory: `python3 ../bench/bencher.py -m <experiment mode> -d <device index>`, where experiment mode and device to be used are chosen. So far, the following experiments are added:

* mode 0: singleton experiment, where the hash table is built given a fixed load factor (which set by using a parameter for expected chain length, or equivalently total number of initial buckets).
* mode 1: load factor experiment, where a series of scenarios are simulated. In each case, total number of elements to be inserted into the hash table are constant, but the load factor (number of buckets) varies from case to case.
* mode 2: variable sized tables experiment, where the load factor (number of buckets) is fixed, but the total number of elements to be inserted into the table is variable.
* mode 3: concurrent experiment, where a series of batches of operations are used in the data structure: each batch is consisted of `(insert_ratio, delete_ratio, search_exist_ratio, search_not_exist_ratio)` as it operation distribution. For example, a tuple of (0.1, 0.1, 0.4, 0.4) would mean that 10% of each batch's operations are new elements to be inserted, 10% are deletion of previously inserted elements (in previous batches), 40% are search queries of elements that are previously inserted, and the final 40% are search queries of elements that are not stored in the data structure at all. Simulation starts with a few number of initial batches with 100% of operations as insertion, and then the rest of the batches with its given probability distribution.