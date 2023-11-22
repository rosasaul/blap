################################################################################
#
# blap - deep learning library
#   "blap, sometimes it just hits you in the face."
#
# Copyright (C) 2017 Saul Rosa http://www.megaframe.org
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
################################################################################

CUDA_INSTALL_PATH = /usr/local/cuda

# Location of the CUDA Toolkit binaries and libraries
CUDA_INC_PATH  = $(CUDA_INSTALL_PATH)/include
CUDA_BIN_PATH  = $(CUDA_INSTALL_PATH)/bin
CUDA_LIB_PATH  = $(CUDA_INSTALL_PATH)/lib64

GENCODE_FLAGS := -gencode arch=compute_61,code=sm_61
NVLDFLAGS := -L$(CUDA_LIB_PATH) -lcudart
NVCCFLAGS := -m64
NVFLAGS := $(NVCCFLAGS) $(NVLDFLAGS) $(GENCODE_FLAGS)

# Common binaries
NVCC            = $(CUDA_BIN_PATH)/nvcc
GCC             = g++ 

CFLAGS=-Wall -Wno-unused-function -Wno-switch
LDFLAGS=-lpthread
TOOLS=tools
LIB=lib
LIBBLAP=$(LIB)/blap
BIN=bin
INC=include
SRC=src
FLAGS=$(LDFLAGS) -std=c++11 -g 

all: blap_train_gpu blap_test_gpu

# Test library

matrix_cuda:
	$(NVCC) $(TOOLS)/matrix_cuda.cu $(NVCCFLAGS) $(NVLDFLAGS) $(GENCODE_FLAGS) -o $(BIN)/$@


# CPU Based blap

blap_test: blap_test.o Model.o Utils.o ActivationFunctions.o InitFunctions.o
	$(GCC) $(LIBBLAP)_test.o $(LIBBLAP)/Model.o $(LIBBLAP)/Utils.o $(LIBBLAP)/ActivationFunctions.o $(LIBBLAP)/InitFunctions.o $(LDFLAGS) $(FLAGS) $(CFLAGS) -I$(INC) -o $(BIN)/$@

blap_train: blap_train.o Model.o Utils.o ActivationFunctions.o InitFunctions.o
	$(GCC) $(LIBBLAP)_train.o $(LIBBLAP)/Model.o $(LIBBLAP)/Utils.o $(LIBBLAP)/ActivationFunctions.o $(LIBBLAP)/InitFunctions.o $(LDFLAGS) $(FLAGS) $(CFLAGS) -I$(INC) -o $(BIN)/$@


# Blap GPU libraries

activation_functions.o:
	$(NVCC) src/blap/gpu/activation_functions.cu $(FLAGS) -c -I$(INC) $(NVFLAGS) -o $(LIBBLAP)/gpu/$@

math.o:
	$(NVCC) src/blap/gpu/math.cu $(FLAGS) -c -I$(INC) $(NVFLAGS) -o $(LIBBLAP)/gpu/$@

Model_gpu.o:
	$(NVCC) $(SRC)/blap/Model_gpu.cu $(FLAGS) -c -I$(INC) $(NVFLAGS) -o $(LIBBLAP)/$@

# Blap Test GPU

blap_test_gpu: blap_test_gpu.o Model_gpu.o Utils.o ActivationFunctions.o InitFunctions.o activation_functions.o math.o
	$(NVCC) $(LIB)/blap_test_gpu.o $(LIBBLAP)/Model_gpu.o $(LIBBLAP)/Utils.o $(LIBBLAP)/ActivationFunctions.o $(LIBBLAP)/InitFunctions.o $(LIBBLAP)/gpu/math.o $(LIBBLAP)/gpu/activation_functions.o $(FLAGS) -I$(INC) $(NVFLAGS) -o bin/$@

blap_test_gpu.o:
	$(NVCC) tools/blap_test_gpu.cu $(FLAGS) -c -I$(INC) $(NVFLAGS) -o $(LIB)/$@


# Blap Train GPU

blap_train_gpu: blap_train_gpu.o Model_gpu.o Utils.o ActivationFunctions.o InitFunctions.o activation_functions.o math.o
	$(NVCC) $(LIB)/blap_train_gpu.o $(LIBBLAP)/Model_gpu.o $(LIBBLAP)/Utils.o $(LIBBLAP)/ActivationFunctions.o $(LIBBLAP)/InitFunctions.o $(LIBBLAP)/gpu/math.o $(LIBBLAP)/gpu/activation_functions.o $(FLAGS) -I$(INC) $(NVFLAGS) -o bin/$@

blap_train_gpu.o:
	$(NVCC) tools/blap_train_gpu.cu $(FLAGS) -c -I$(INC) -o $(LIB)/$@


# Other .o files

blap_test.o:
	$(GCC) $(TOOLS)/blap_test.cpp $(FLAGS) -c -I$(INC)/blap -I$(INC) -o $(LIB)/$@

blap_train.o:
	$(GCC) $(TOOLS)/blap_train.cpp $(FLAGS) -c -I$(INC)/blap -I$(INC) -o $(LIB)/$@

Model.o:
	$(GCC) $(SRC)/blap/Model.cpp $(FLAGS) -c -I$(INC)/blap -I$(INC) -o $(LIBBLAP)/$@

Utils.o:
	$(GCC) $(SRC)/blap/Utils.cpp $(FLAGS) -c -I$(INC)/blap -I$(INC) -o $(LIBBLAP)/$@

ActivationFunctions.o:
	$(GCC) $(SRC)/blap/ActivationFunctions.cpp $(FLAGS) -c -I$(INC)/blap -I$(INC) -o $(LIBBLAP)/$@

InitFunctions.o:
	$(GCC) $(SRC)/blap/InitFunctions.cpp $(FLAGS) -c -I$(INC)/blap -I$(INC) -o $(LIBBLAP)/$@

.PHONY: clean

clean:
	-rm bin/* lib/*.o lib/blap/*.o lib/blap/gpu/*.o


