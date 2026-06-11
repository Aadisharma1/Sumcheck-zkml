# ResNet-18 Sumcheck Prover

## Prerequisites
* Python 3.10+ with PyTorch & Torchvision
* CMake 3.16+
* C++17 Compiler (GCC/Clang/MSVC)

## Execution Flow

1. **Offline Fusion & Weight Export**
   Run the PyTorch script to fuse BatchNorm layers and dump the raw binaries.
   ```bash
   python fusion_export.py
   ```
   *Expected output: A `fused_weights/` directory populated with `.bin` and `_meta.txt` files, along with `input.bin`.*

2. **Build the C++ Proving Backend**
   ```bash
   mkdir build && cd build
   cmake ..
   cmake --build . --config Release
   ```

3. **Run the Prover/Verifier**
   Pass the path of the fused weights to the binary.
   ```bash
   ./zkresnet ../fused_weights
   ```
