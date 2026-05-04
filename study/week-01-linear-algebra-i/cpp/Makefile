# Makefile — Week 01 C++ Programs
# Assad's 52-Week NVIDIA Curriculum
#
# Usage:
#   make           → build all three programs
#   make run       → build and run all three
#   make clean     → remove binaries

CXX      := g++
CXXFLAGS := -O2 -Wall -Wextra -std=c++17

TARGETS  := dot_product basis_vectors memory_demo

.PHONY: all run clean

all: $(TARGETS)

dot_product: dot_product.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

basis_vectors: basis_vectors.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

memory_demo: memory_demo.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

run: all
	@echo ""
	@echo "══════════════════════════════════════════"
	@echo "Running: dot_product"
	@echo "══════════════════════════════════════════"
	./dot_product

	@echo ""
	@echo "══════════════════════════════════════════"
	@echo "Running: basis_vectors"
	@echo "══════════════════════════════════════════"
	./basis_vectors

	@echo ""
	@echo "══════════════════════════════════════════"
	@echo "Running: memory_demo"
	@echo "══════════════════════════════════════════"
	./memory_demo

clean:
	rm -f $(TARGETS)
