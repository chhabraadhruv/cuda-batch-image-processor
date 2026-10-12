# cuda-batch-image-processor
# High-Performance Asynchronous Multi-Stream CUDA Image Processor

A high-performance C++/CUDA batch image processing engine implementing 2D Sobel edge detection. The project highlights core GPU acceleration techniques including tiled shared memory stencil convolutions, page-locked (pinned) memory transfers, dynamic multi-stream queueing, and Google C++ Style Guide compliance.

## Features
- **Tiled Shared Memory Kernel:** Eliminates redundant global memory transactions using a $16 \times 16$ thread block with a 1-pixel halo boundary.
- **Multi-Stream Asynchrony:** Overlaps Host-to-Device (H2D) transfer, GPU kernel computation, and Device-to-Host (D2H) transfer across concurrent `cudaStream_t` queues.
- **Zero-Copy Host Buffers:** Leverages `cudaMallocHost` for high-throughput PCIe memory transfers.
- **Command-Line Interface:** Standard POSIX argument parsing supporting runtime configuration.
- **Google C++ Style Guide:** Adheres to naming conventions (`PascalCase` functions, `camelCase` variables, `kPascalCase` constants), explicit namespace scoping, and strict RAII buffer management.

## Prerequisites
- NVIDIA CUDA Toolkit 11.0+
- C++14 compliant compiler (`g++` 7.5+)
- Python 3.8+ with `pandas`, `matplotlib`, `numpy`, and `Pillow` (for automated benchmarks)

## Building the Project

### Using Makefile
```bash
# Build both single-stream and multi-stream binaries
make all

# Clean build artifacts
make clean
