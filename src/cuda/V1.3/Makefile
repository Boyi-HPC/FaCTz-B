SHELL := /bin/sh

REPO_ROOT := $(abspath ../../..)
BUILD_DIR ?= $(REPO_ROOT)/build
BUILD_TYPE ?= Release
JOBS ?= 8
CUDA_ARCHITECTURES ?= 70;75;80;86
CUDACXX ?= $(shell command -v nvcc 2>/dev/null)
ifeq ($(strip $(CUDACXX)),)
CUDACXX := /usr/local/cuda-12/bin/nvcc
endif

FTK_DIR ?= $(REPO_ROOT)/external/ftk/install/lib/cmake/FTK
COMPRESS_TARGET := cpszg_2d_v1_3
DECOMPRESS_TARGET := decpszg_2d_v1_3
TARGETS := $(COMPRESS_TARGET) $(DECOMPRESS_TARGET)
COMPRESS_PROGRAM := $(BUILD_DIR)/bin/$(COMPRESS_TARGET)
DECOMPRESS_PROGRAM := $(BUILD_DIR)/bin/$(DECOMPRESS_TARGET)
RUN_DIR ?= $(BUILD_DIR)/runs/v1.3
U ?= $(REPO_ROOT)/data/uf.dat
V ?= $(REPO_ROOT)/data/vf.dat
R1 ?= 2400
R2 ?= 3600
EB ?= 0.1

.PHONY: all configure build compress decompress run clean

all: build

configure:
	cmake -S "$(REPO_ROOT)" -B "$(BUILD_DIR)" \
		-DCMAKE_BUILD_TYPE="$(BUILD_TYPE)" \
		-DCMAKE_CUDA_COMPILER="$(CUDACXX)" \
		-DCMAKE_CUDA_ARCHITECTURES="$(CUDA_ARCHITECTURES)" \
		-DFTK_DIR="$(FTK_DIR)" \
		-DFACTZ_BUILD_CUDA=ON \
		-DFACTZ_BUILD_CUDA_NAIVE=OFF \
		-DFACTZ_BUILD_CUDA_V1=OFF \
		-DFACTZ_BUILD_CUDA_V1_1=OFF \
		-DFACTZ_BUILD_CUDA_V1_2=OFF \
		-DFACTZ_BUILD_CUDA_V1_3=ON \
		-DFACTZ_BUILD_CUDA_V1_4=OFF

build: configure
	cmake --build "$(BUILD_DIR)" --parallel "$(JOBS)" --target $(TARGETS)

compress: build
	@test -f "$(U)" || { echo "Missing U input: $(U)" >&2; exit 1; }
	@test -f "$(V)" || { echo "Missing V input: $(V)" >&2; exit 1; }
	@mkdir -p "$(RUN_DIR)"
	@ln -sf "$(U)" "$(RUN_DIR)/U.dat"
	@ln -sf "$(V)" "$(RUN_DIR)/V.dat"
	cd "$(RUN_DIR)" && "$(COMPRESS_PROGRAM)" U.dat V.dat "$(R1)" "$(R2)" "$(EB)"

decompress: build
	@test -f "$(RUN_DIR)/U.dat.cucpsz" || { echo "Missing compressed input: $(RUN_DIR)/U.dat.cucpsz" >&2; exit 1; }
	cd "$(RUN_DIR)" && "$(DECOMPRESS_PROGRAM)" U.dat.cucpsz U.dat.out V.dat.out

run: build
	@test -f "$(U)" || { echo "Missing U input: $(U)" >&2; exit 1; }
	@test -f "$(V)" || { echo "Missing V input: $(V)" >&2; exit 1; }
	@mkdir -p "$(RUN_DIR)"
	@ln -sf "$(U)" "$(RUN_DIR)/U.dat"
	@ln -sf "$(V)" "$(RUN_DIR)/V.dat"
	cd "$(RUN_DIR)" && "$(COMPRESS_PROGRAM)" U.dat V.dat "$(R1)" "$(R2)" "$(EB)"
	cd "$(RUN_DIR)" && "$(DECOMPRESS_PROGRAM)" U.dat.cucpsz U.dat.out V.dat.out

clean:
	@if test -f "$(BUILD_DIR)/CMakeCache.txt"; then cmake --build "$(BUILD_DIR)" --target clean; fi
