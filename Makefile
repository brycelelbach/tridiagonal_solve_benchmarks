CXX=icpc

ifdef DEBUG
  CXXFLAGS+=-O0 -ggdb
else
  CXXFLAGS+=-O3 -static
  ifdef DISABLE_ASSERTS
    CXXFLAGS+=-DNDEBUG
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


