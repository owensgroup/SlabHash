cuda_add_executable(cmap_test cmap_test.cu 
		OPTIONS ${GENCODE} ${CUDA_VERBOSE_PTXAS})
target_link_libraries(cmap_test gtest)

cuda_add_executable(test_slab_hash test_slab_hash.cu 
		OPTIONS ${GENCODE} ${CUDA_VERBOSE_PTXAS})

cuda_add_executable(concurrent_map concurrent_map.cu
		OPTIONS ${GENCODE} ${CUDA_VERBOSE_PTXAS})

cuda_add_executable(concurrent_set concurrent_set.cu
                OPTIONS ${GENCODE} ${CUDA_VERBOSE_PTXAS})

cuda_add_executable(iterator_test iterator_test.cu
		OPTIONS ${GENCODE} ${CUDA_VERBOSE_PTXAS})
