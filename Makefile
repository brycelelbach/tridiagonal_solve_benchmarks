###############################################################################
# Copyright (c) 2015-6 Bryce Adelstein Lelbach aka wash <brycelelbach@gmail.com>
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
###############################################################################

CXX=icpc

CXXFLAGS+=-mavx -std=c++11 -mkl=sequential

ifneq ($(OPENMP),0)
  CXXFLAGS+=-openmp
endif

ifeq ($(DEBUG),1)
  CXXFLAGS+=-O0 -ggdb
  ifeq ($(ASSERTS),0)
    CXXFLAGS+=-DNDEBUG
    CXXFLAGS+=-DBUILD_TYPE=\"Debug\ Dynamic\ Asserts\ Off\"
  else
    CXXFLAGS+=-DBUILD_TYPE=\"Debug\ Dynamic\ Asserts\ On\"
  endif

  ifeq ($(STATIC),0)
    $(error Debug builds must be dynamic.)
  endif
else
  CXXFLAGS+=-O3
  ifeq ($(STATIC),0)
    ifeq ($(ASSERTS),0)
      # STATIC == 0, ASSERTS == 0
      CXXFLAGS+=-DNDEBUG
      CXXFLAGS+=-DBUILD_TYPE=\"Release\ Dynamic\ Asserts\ Off\"
    else
      # STATIC == 0, ASSERTS == 1
      CXXFLAGS+=-DBUILD_TYPE=\"Release\ Dynamic\ Asserts\ On\"
    endif
  else
    ifeq ($(ASSERTS),0)
      # STATIC == 1, ASSERTS == 0
      CXXFLAGS+=-static -DNDEBUG
      CXXFLAGS+=-DBUILD_TYPE=\"Release\ Static\ Asserts\ Off\"
    else
      # STATIC == 1, ASSERTS == 1
      CXXFLAGS+=-static
      CXXFLAGS+=-DBUILD_TYPE=\"Release\ Static\ Asserts\ On\"
    endif
  endif
endif

SOURCES=$(shell ls -1 $(CURDIR)/*.cpp)
PROGRAMS=$(SOURCES:.cpp=)
DIRECTORIES=$(CURDIR)/build

all: directories $(PROGRAMS)

.PHONY: directories

directories: $(DIRECTORIES)/

$(DIRECTORIES)/:
	@mkdir -p $@ 

% : %.cpp
	@echo "**************************************************"
	@echo "Building $(*F)"
	$(CXX) $(CXXFLAGS) $< -o $(CURDIR)/build/$(*F)
	@echo "**************************************************"
	@echo "Running $(*F)"
	@OMP_NUM_THREADS=1 build/$(*F)
	@echo

clean:
	@echo "Cleaning build directory"
	@rm -f $(DIRECTORIES)/*


