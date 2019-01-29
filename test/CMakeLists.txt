cuda_add_executable(test_slab_hash test_slab_hash.cu 
		OPTIONS ${GENCODE} ${CUDA_VERBOSE_PTXAS})
