cuda_add_executable(benchmark main_benchmarks.cu
			OPTIONS ${GENCODE} ${CUDA_VERBOSE_PTXAS})
