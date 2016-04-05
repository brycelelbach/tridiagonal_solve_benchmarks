CXX=icpc

ifdef DEBUG
  CXXFLAGS+=-O0 -ggdb
  ifeq ($(ASSERTS),0)
    CXXFLAGS+=-DNDEBUG
    CXXFLAGS+=-DBUILD_TYPE=\"Debug\ Asserts\ Off\"
  else
    CXXFLAGS+=-DBUILD_TYPE=\"Debug\ Asserts\ On\"
  endif
else
  CXXFLAGS+=-O3 -static
  ifeq ($(ASSERTS),0)
    CXXFLAGS+=-DNDEBUG
    CXXFLAGS+=-DBUILD_TYPE=\"Release\ Asserts\ Off\"
  else
    CXXFLAGS+=-DBUILD_TYPE=\"Release\ Asserts\ On\"
  endif
endif


CXXFLAGS+=-mavx -std=c++11 -mkl=sequential
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
	@build/$(*F)
	@echo

clean:
	@echo "Cleaning build directory"
	@rm -f $(DIRECTORIES)/*
	@rmdir $(DIRECTORIES)


