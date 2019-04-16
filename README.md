# SlabHash
A warp-oriented dynamic hash table for GPUs

## Publication:
This library is based on the original slab hash paper, initially proposed in the following IPDPS'18 paper:
* [Saman Ashkiani, Martin Farach-Colton, John Owens, *A Dynamic Hash Table for the GPU*, 2018 IEEE International Parallel and Distributed Processing Symposium (IPDPS)](https://ieeexplore.ieee.org/abstract/document/8425196)

This library is a rafactored and slightly redesigned version of the original code, so that it can be extended and be used in other research projects as well. It is still under continuous development. If you find any problem with the code, or suggestions for potential additions to the library, we will appreciate it if you can raise issues on github. We will address them as soon as possible. 

## Compilation
1. `git submodule init`
2. `git submodule update`
3. Make sure to edit `CMakeLists.txt` such that it reflects the GPU device's compute capability. For example, to include compute 3.5 you should have `option(SLABHASH_GENCODE_SM35 "GENCODE_SM35" ON)`.
4. `mkdir build && cd build`
5. `cmake ..`
6. `make`

## High level API
In order to use this code, it is required to include [https://github.com/owensgroup/SlabHash/blob/master/src/slab_hash.cuh](`src/slab_hash.cuh`), which itself will include all required variations of the GpuSlabHash main class.
We have provided a simple application class [https://github.com/owensgroup/SlabHash/blob/master/src/gpu_hash_table.cuh](gpu_hash_table), where the right instance of `GpuSlabHash<KeyT, ValueT, SlabHashT>` is initialized.
This class is just an example of how to use the GpuSlabHash in various contexts.
Any other similar application level API should also own the dynamic memory allocator that is used by all instances of GpuSlabHash class (here just one). Finally, GpuSlabHash will be constructed with a pointer to the mentioned dynamic allocator.

There are a few variations of GpuSlabHash class. The most complete one at the moment is [https://github.com/owensgroup/SlabHash/blob/master/src/concurrent_map/cmap_class.cuh](`GpuSlabHash<KeyT, ValueT, SlabHashTypeT::ConcurrentMap>`) which is based on the initial idea of the slab hash proposed in the paper above.
This class partially owns all the memory allocated on the GPU to actually store all the contents, side by side all units allocated by the dynamic memory allocator. 
There is another class, named [https://github.com/owensgroup/SlabHash/blob/master/src/concurrent_map/cmap_class.cuh#L26](`GpuSlabHashContext`), which does not own any memory but has all the related member functions to use the data structure itself. The context class is the one which is used by GPU threads on the device. Here's an example of the way to use it for a [https://github.com/owensgroup/SlabHash/blob/master/src/concurrent_map/device/search_kernel.cuh](search kernel):

```
template <typename KeyT, typename ValueT>
__global__ void search_table(
    KeyT* d_queries,
    ValueT* d_results,
    uint32_t num_queries,
    GpuSlabHashContext<KeyT, ValueT, SlabHashTypeT::ConcurrentMap> slab_hash) {
  uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  uint32_t laneId = threadIdx.x & 0x1F;

  if ((tid - laneId) >= num_queries) {
    return;
  }

  // initializing the memory allocator on each warp:
  slab_hash.getAllocatorContext().initAllocator(tid, laneId);

  KeyT myQuery = 0;
  ValueT myResult = static_cast<ValueT>(SEARCH_NOT_FOUND);
  uint32_t myBucket = 0;
  bool to_search = false;
  if (tid < num_queries) {
    myQuery = d_queries[tid];
    myBucket = slab_hash.computeBucket(myQuery);
    to_search = true;
  }

  slab_hash.searchKey(to_search, laneId, myQuery, myResult, myBucket);

  // writing back the results:
  if (tid < num_queries) {
    d_results[tid] = myResult;
  }
}
```

## Simple benchmarking
A simplified set of benchmark scenarios are available through a Python API that can be used as follows: Once the code is successfully compiled you can run the following python code from the `build` directory: `python3 ../bench/bencher.py -m <experiment mode> -d <device index>`, where experiment mode and device to be used are chosen. So far, the following experiments are added:

* mode 0: singleton experiment, where the hash table is built given a fixed load factor (which set by using a parameter for expected chain length, or equivalently total number of initial buckets).
* mode 1: load factor experiment, where a series of scenarios are simulated. In each case, total number of elements to be inserted into the hash table are constant, but the load factor (number of buckets) varies from case to case.
* mode 2: variable sized tables experiment, where the load factor (number of buckets) is fixed, but the total number of elements to be inserted into the table is variable.
* mode 3: concurrent experiment, where a series of batches of operations are used in the data structure: each batch is consisted of `(insert_ratio, delete_ratio, search_exist_ratio, search_not_exist_ratio)` as it operation distribution. For example, a tuple of (0.1, 0.1, 0.4, 0.4) would mean that 10% of each batch's operations are new elements to be inserted, 10% are deletion of previously inserted elements (in previous batches), 40% are search queries of elements that are previously inserted, and the final 40% are search queries of elements that are not stored in the data structure at all. Simulation starts with a few number of initial batches with 100% of operations as insertion, and then the rest of the batches with its given probability distribution.

In the following, these benchmarks are run a few GPU architectures. It should be noted that majority of input parameters for these scenarios are not exposed as command line arguments in the python code. If interested to try with different set of settings, the reader should either use their corresponding C++ API (through `build/bin/benchmark` and with the parameters listed in [https://github.com/owensgroup/SlabHash/blob/master/bench/main_benchmarks.cu](bench/main_benchmark.cu)), or change these parameters in [https://github.com/owensgroup/SlabHash/blob/master/bench/bencher.py#L166](`bench/bencher.py`).

### NVIDIA Titan V:

Titan V has Volta architecture with 12GB of DRAM memory. In our setting, we have NVIDIA driver 410.104, and CUDA 10.0 running.

#### Mode 0:
```
python3 ../bench/bencher.py -m 0 -d 0


GPU hardware: TITAN V
===============================================================================================
Singleton experiment:
        Number of elements to be inserted: 4194304
        Number of buckets: 466034
        Expected chain length: 0.60
===============================================================================================
load factor     build rate(M/s)         search rate(M/s)        search rate bulk(M/s)
===============================================================================================
0.55            1525.352                4137.374                3241.468
```

#### Mod 1:
```
python3 ../bench/bencher.py -m 1 -d 0

GPU hardware: TITAN V
===============================================================================================
Load factor experiment:
        Total number of elements is fixed, load factor (number of buckets) is a variable
        Number of elements to be inserted: 4194304
         1.00 of 4194304 queries exist in the data structure
===============================================================================================
load factor     num buckets     build rate(M/s)         search rate(M/s)        search rate bulk(M/s)
===============================================================================================
0.06            4194304         1416.107                3851.094                3454.809
0.19            1398102         1454.223                3934.442                3575.244
0.25            1048576         1466.819                3978.993                3603.156
0.37            699051          1491.658                4053.439                3629.898
0.44            599187          1508.881                4084.385                3512.300
0.55            466034          1527.094                4138.811                3239.865
0.60            419431          1528.536                4146.405                2877.604
0.65            349526          1522.836                4095.360                2125.584
0.65            262144          1476.884                3785.364                1318.751
0.65            279621          1481.709                3886.148                1436.972
0.66            233017          1451.372                3599.791                1164.226
0.66            322639          1512.172                4044.683                1811.162
0.66            220753          1431.386                3508.069                1110.930
0.69            199729          1408.241                3352.397                1024.753
0.70            190651          1413.983                3278.603                991.955
0.73            174763          1403.611                3149.785                934.420
0.74            167773          1381.567                3085.426                903.303
0.75            155345          1367.470                2973.300                850.200
0.76            149797          1363.288                2914.719                823.777
0.76            139811          1349.699                2808.064                777.419
```

#### Mode 2:
```
python3 ../bench/bencher.py -m 2 -d 0

GPU hardware: TITAN V
===============================================================================================
Table size experiment:
        Table's expected chain length is fixed, and total number of elements is variable
        Expected chain length = 0.60

        1.00 of 262144 queries exist in the data structure
===============================================================================================
(num keys, num buckets, load factor)    build rate(M/s)         search rate(M/s)        search rate bulk(M/s)
===============================================================================================
(262144, 29128, 0.55)                     2640.026              4571.429                3529.513
(524288, 58255, 0.55)                     2473.430              4701.291                3207.518
(1048576, 116509, 0.55)                   2011.170              4821.660                3431.563
(2097152, 233017, 0.55)                   1673.630              4426.912                3475.236
(4194304, 466034, 0.55)                   1530.160              4154.290                3431.204
(8388608, 932068, 0.55)                   1464.140              3996.341                3214.361
```

#### Mode 3:
```
python3 ../bench/bencher.py -m 3 -d 0

GPU hardware: TITAN V
===============================================================================================
Concurrent experiment:
        variable load factor, fixed number of elements
        Operation ratio: (insert, delete, search) = (0.10, 0.10, [0.40, 0.40])
===============================================================================================
batch_size = 262144, init num batches = 3, final num batches = 4
===============================================================================================
init lf         final lf        num buckets     init build rate(M/s)    concurrent rate(Mop/s)
===============================================================================================
0.05            0.05            1048576         1427.426                2669.273
0.14            0.14            349526          1526.934                2826.777
0.19            0.19            262144          1590.783                2801.642
0.28            0.28            174763          1714.166                2952.072
0.33            0.33            149797          1781.644                3000.733
0.42            0.42            116509          1937.406                3119.574
0.47            0.47            104858          1992.379                3088.990
0.55            0.55            87382           2099.257                3144.722
0.59            0.58            80660           2137.415                3166.602
0.64            0.62            69906           2160.717                2986.511
```

### Titan Xp

Titan Xp has Pascal architecture with 12GB of DRAM memory. In our setting, we have NVIDIA driver 410.104, and CUDA 10.0 running.

### Tesla K40c

Tesla K40c has Kepler architecture with compute capability 3.5 and 12GB of DRAM. In our setting, we have NVIDIA driver 410.72 and CUDA 10.0.

#### Mode 0:
```
python3 ../bench/bencher.py -m 0 -d 2

GPU hardware: Tesla K40c
===============================================================================================
Singleton experiment:
        Number of elements to be inserted: 4194304
        Number of buckets: 466034
        Expected chain length: 0.60
===============================================================================================
load factor     build rate(M/s)         search rate(M/s)        search rate bulk(M/s)
===============================================================================================
0.55            545.779         764.014         831.575
``` 

#### Mode 1:
```
python3 ../bench/bencher.py -m 1 -d 2

GPU hardware: Tesla K40c
===============================================================================================
Load factor experiment:
        Total number of elements is fixed, load factor (number of buckets) is a variable
        Number of elements to be inserted: 4194304
         1.00 of 4194304 queries exist in the data structure
===============================================================================================
load factor     num buckets     build rate(M/s)         search rate(M/s)        search rate bulk(M/s)
===============================================================================================
0.06            4194304         427.761         737.781         797.139
0.19            1398102         539.284         758.641         828.134
0.25            1048576         548.825         769.378         841.300
0.37            699051          551.950         769.572         841.694
0.44            599187          551.411         769.604         841.559
0.55            466034          546.190         764.509         831.907
0.60            419431          540.693         758.150         819.574
0.65            349526          521.354         734.935         777.110
0.65            262144          467.077         660.569         675.041
0.65            279621          480.977         679.845         701.025
0.66            233017          443.047         621.548         630.487
0.66            322639          508.520         719.334         753.231
0.66            220753          432.415         603.049         610.993
0.69            199729          414.232         571.586         578.291
0.70            190651          406.401         557.613         564.020
0.73            174763          391.686         532.063         538.003
0.74            167773          384.449         520.422         525.573
0.75            155345          371.302         498.036         504.311
0.76            149797          364.787         487.541         492.959
0.76            139811          352.283         467.503         472.981
```

#### Mode 2:
```
python3 ../bench/bencher.py -m 2 -d 2

GPU hardware: Tesla K40c
===============================================================================================
Table size experiment:
        Table's expected chain length is fixed, and total number of elements is variable
        Expected chain length = 0.60

        1.00 of 262144 queries exist in the data structure
===============================================================================================
(num keys, num buckets, load factor)    build rate(M/s)         search rate(M/s)        search rate bulk(M/s)
===============================================================================================
(262144, 29128, 0.55)                      538.062              742.231         823.234
(524288, 58255, 0.55)                      547.301              755.789         829.696
(1048576, 116509, 0.55)                    550.168              761.621         832.457
(2097152, 233017, 0.55)                    547.768              763.422         831.348
(4194304, 466034, 0.55)                    546.646              764.558         832.098
(8388608, 932068, 0.55)                    544.300              764.801         832.008
```
#### Mode 3:
```
python3 ../bench/bencher.py -m 3 -d 2

GPU hardware: Tesla K40c
===============================================================================================
Concurrent experiment:
        variable load factor, fixed number of elements
        Operation ratio: (insert, delete, search) = (0.10, 0.10, [0.40, 0.40])
===============================================================================================
batch_size = 262144, init num batches = 3, final num batches = 4
===============================================================================================
init lf         final lf        num buckets     init build rate(M/s)    concurrent rate(Mop/s)
===============================================================================================
0.05            0.05            1048576         502.381         649.592
0.14            0.14            349526          507.926         656.305
0.19            0.19            262144          509.950         660.272
0.28            0.28            174763          511.659         663.212
0.33            0.33            149797          512.075         662.354
0.42            0.42            116509          513.390         664.073
0.47            0.47            104858          511.723         657.781
0.55            0.55            87382           509.052         649.026
0.59            0.58            80660           501.725         639.850
0.64            0.62            69906           493.702         601.822

```