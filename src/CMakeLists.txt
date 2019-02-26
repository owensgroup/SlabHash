set(CUHFILES
	slab_hash_global.cuh
	slab_hash.cuh)

cuda_add_library(slab_hash STATIC
	${CUHFILES}
	${CUFILES}
	OPTIONS ${GENCODE} ${VERBOSE_PTXAS})
