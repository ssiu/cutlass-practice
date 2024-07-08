CUTLASS_DIR = cutlass
NVCC := $(shell which nvcc 2>/dev/null)

ifeq ($(NVCC),)
		$(error nvcc not found.)
endif

NVCC_FLAGS = -O3 -std=c++17
NVCC_LDFLAGS = -lcublas

GPU_COMPUTE_CAPABILITY = $(shell __nvcc_device_query) # assume if NVCC is present, then this likely is too
GPU_COMPUTE_CAPABILITY := $(strip $(GPU_COMPUTE_CAPABILITY))

NVCC_FLAGS += --generate-code arch=compute_$(GPU_COMPUTE_CAPABILITY),code=[compute_$(GPU_COMPUTE_CAPABILITY),sm_$(GPU_COMPUTE_CAPABILITY)]

.PHONY: all transpose.o

all: transpose.o

transpose.o: transpose/run_transpose.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $< -I${CUTLASS_DIR}/include
