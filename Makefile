###############################################################################
# Copyright (c) 2015-6 Bryce Adelstein Lelbach aka wash <brycelelbach@gmail.com>
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
###############################################################################

# ARCH=AVX|AVX2|AVX512 (default: AVX)
# TEST=0|1             (default: 1)
# DEBUG=0|1            (default: 0)
# STATIC=0|1           (default: 1 if not debug)
# ASSERTS=0|1          (default: 1)

CXX=icpc

MKLFLAGS+=-mkl=sequential

ifeq ($(NERSC_HOST),cori)
  ifeq ($(ARCH),)
	ARCH=AVX2
  endif
endif
ifeq ($(NERSC_HOST),edison)
  ifeq ($(ARCH),)
	ARCH=AVX
  endif
endif

ifeq ($(ARCH),AVX512)
  BUILD_TYPE=AVX512
  CXXFLAGS+=-xMIC-AVX512 -DTSB_ARCH_AVX512
else ifeq ($(ARCH),AVX2)
  BUILD_TYPE=AVX2
  CXXFLAGS+=-xCORE-AVX2 -DTSB_ARCH_AVX2
else
  BUILD_TYPE=AVX
  CXXFLAGS+=-xAVX -DTSB_ARCH_AVX
endif

ifeq ($(TEST),)
  TEST=1
endif

ifeq ($(DEBUG),1)     # DEBUG == 1
  BUILD_TYPE:=$(BUILD_TYPE).DEBUG
  CXXFLAGS+=-O0 -ggdb
  ifeq ($(STATIC),1)
    $(error Debug builds must be dynamic.)
  endif               # DEBUG == 0
else
  BUILD_TYPE:=$(BUILD_TYPE).RELEASE
  CXXFLAGS+=-O3 -ipo -no-prec-div -fp-model fast=2
endif

ifneq ($(STATIC),0)   # STATIC == 1
  BUILD_TYPE:=$(BUILD_TYPE).STATIC
  CXXFLAGS+=-static
else                  # STATIC == 0
  BUILD_TYPE:=$(BUILD_TYPE).DYNAMIC
endif

ifneq ($(ASSERTS),0)  # ASSERTS == 1
  BUILD_TYPE:=$(BUILD_TYPE).ASSERTS-ON
else                  # ASSERTS == 0
  BUILD_TYPE:=$(BUILD_TYPE).ASSERTS-OFF
  CXXFLAGS+=-DNDEBUG
endif

# Disable icpc warning #2261: __assume expression with side effects discarded
CXXFLAGS+=-wd2261

CXXFLAGS+=-std=c++11 -qopenmp
CXXFLAGS+=-DTSB_BUILD_TYPE=\"$(BUILD_TYPE)\"
CXXFLAGS+=-DTSB_GIT_REVISION=\"$(shell git rev-parse --short HEAD)\"

# We disable IPO because ICPC dumps ~500k of raw IPO data into the assembly
# files otherwise.
ASMFLAGS+=-no-ipo -fcode-asm -fsource-asm

OPTREPORTFLAGS+=-no-ipo -qopt-report=5

# Disable icpc remark #10397: optimization reports are generated in *.optrpt
# files in the output location
OPTREPORTFLAGS+=-wd10397

SOURCES=$(shell ls -1 $(CURDIR)/*.cpp)
PROGRAMS=$(SOURCES:.cpp=)
DIRECTORY=$(CURDIR)/build

all: $(PROGRAMS)

asm: $(PROGRAMS:=.asm)

directory: $(DIRECTORY)/

clean:
	@echo "Cleaning build directory"
	@rm -f $(DIRECTORY)/*
	@if [ -d "$(DIRECTORY)" ]; then rmdir $(DIRECTORY); fi

.PHONY: directory asm clean

% : %.cpp directory
	@echo "********************************************************************************"
	@echo "Building $(*F)"
	$(CXX) $(if $(findstring mkl,$<),$(MKLFLAGS) $(CXXFLAGS),$(CXXFLAGS)) $< -o $(CURDIR)/build/$(*F)
	@echo "********************************************************************************"
	@if [ $(TEST) -eq 1 ]; then                                                                 \
	   echo "Running $(*F)";                                                                    \
	   OMP_NUM_THREADS=1 build/$(*F);                                                           \
	   echo "********************************************************************************"; \
	fi
	@echo

%.asm : %.cpp directory
	@echo "********************************************************************************"
	@echo "Generating assembly for $(*F)"
	$(CXX) $(if $(findstring mkl,$<),$(MKLFLAGS) $(CXXFLAGS),$(CXXFLAGS)) $(ASMFLAGS) -S $< -o $(CURDIR)/build/$(*F).asm
	@echo "********************************************************************************"
	@echo "Generating optimization report for $(*F)"
	$(CXX) $(if $(findstring mkl,$<),$(MKLFLAGS) $(CXXFLAGS),$(CXXFLAGS)) $(OPTREPORTFLAGS) $< -o $(CURDIR)/build/$(*F)
	@echo "********************************************************************************"
	@echo

$(DIRECTORY)/:
	@mkdir -p $@ 

