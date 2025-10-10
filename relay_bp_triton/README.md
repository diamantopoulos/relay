# Relay-BP-S Triton GPU Backend

Ultra-fast GPU implementation of the Relay-BP-S algorithm for quantum error correction decoding using PyTorch and Triton kernels. This implementation provides GPU acceleration while maintaining algorithmic fidelity with the Rust reference implementation.

## Overview

The Relay-BP-S (Relay Belief Propagation with Stopping) algorithm is a quantum error correction decoding method that combines belief propagation with ensemble decoding and disordered memory strengths to improve decoding performance on quantum error correction codes. This Triton implementation provides GPU acceleration for high-throughput quantum error correction applications.

## Key Features

- **GPU Acceleration**: Optimized Triton kernels for belief propagation message passing
- **Algorithmic Fidelity**: Maintains equivalence with the Rust reference implementation
- **Device-resident Decoding**: No host↔device copies during decode loop
- **Batch Processing**: Efficient parallel decoding of multiple syndromes
- **Memory Efficient**: Sparse matrix representation with edge-centric GPU layout
- **Flexible Precision**: Support for fp16/fp32 message precision
- **Autotuning**: Automatic kernel parameter optimization for performance
- **Interface Compatibility**: Seamless integration with existing `relay_bp` package
- **Observable Decoding**: Built-in support for logical error detection

## Installation

```bash
# Install from the project root
pip install -r requirements.txt

# Or install the package in development mode
pip install -e .
```

## Quick Start

### Basic Usage

```python
import torch
import numpy as np
from scipy.sparse import csr_matrix
from relay_bp_triton import RelayBPDecoder

# Create check matrix representing quantum code stabilizer generators
H_csr = csr_matrix(np.array([
    [1, 1, 0],  # First stabilizer generator
    [0, 1, 1]   # Second stabilizer generator
], dtype=np.uint8))

# Error probabilities for each qubit
error_priors = np.array([0.01, 0.01, 0.01])

# Create decoder with quantum error correction parameters
decoder = RelayBPDecoder(
    H_csr=H_csr,
    error_priors=error_priors,
    pre_iter=80,                    # T₀: iterations for ordered memory phase
    num_sets=100,                   # R: number of relay legs/ensembles
    set_max_iter=60,                # Tᵣ: iterations per relay set
    gamma0=0.65,                    # γ₀: ordered memory strength
    gamma_dist_interval=(-0.24, 0.66),  # Disordered γ sampling range
    stop_nconv=5,                   # S: number of solutions to collect
    stopping_criterion="nconv",    # "nconv" | "pre_iter" | "all"
    dtype_messages="fp16",          # Message precision
    device="cuda"
)

# Decode syndromes (quantum measurement outcomes)
syndromes = torch.tensor([[1, 1], [0, 1]], dtype=torch.uint8, device="cuda")
result = decoder.decode(syndromes)

print("Decoded errors:", result["errors"])
print("Solution weights:", result["weights"])
print("Valid solutions:", result["valid_mask"])
print("Iterations used:", result["iterations"])
```

### Using the Adapter Interface

For compatibility with existing `relay_bp` package tools:

```python
from relay_bp_triton import RelayDecoder, ObservableDecoderRunner

# Create adapter decoder
decoder = RelayDecoder(
    check_matrix=H_csr,
    error_priors=error_priors,
    gamma0=0.65,
    pre_iter=80,
    num_sets=100,
    set_max_iter=60,
    gamma_dist_interval=(-0.24, 0.66),
    stop_nconv=5,
    stopping_criterion="nconv"
)

# Decode single syndrome
syndrome = np.array([1, 1], dtype=np.uint8)
error_pattern = decoder.decode(syndrome)

# Batch processing with observable decoding
runner = ObservableDecoderRunner(decoder, observable_matrix)
results = runner.from_errors_decode_observables_detailed_batch(
    error_patterns, num_shots=1000
)
```

## API Reference

### Core Classes

#### `RelayBPDecoder`

Main decoder class implementing the Relay-BP-S algorithm with GPU acceleration.

**Parameters:**
- `H_csr`: Check matrix (C × V) in SciPy CSR format representing stabilizer generators
- `error_priors`: [V] error probabilities in (0, 0.5) for each qubit
- `pre_iter`: T₀ - iterations for first set (ordered memory phase, default: 80)
- `num_sets`: R - number of relay legs/ensembles (default: 100)
- `set_max_iter`: Tᵣ - iterations per relay set (default: 60)
- `gamma0`: γ₀ - ordered memory strength for first set (default: 0.65)
- `gamma_dist_interval`: (min, max) for disordered γ sampling in relay legs (default: (-0.24, 0.66))
- `stop_nconv`: S - number of valid solutions to collect before stopping (default: 5)
- `stopping_criterion`: "nconv" (default), "pre_iter", or "all" to control early stopping
- `normalized_min_sum_alpha`: α for normalized min-sum (0 < α ≤ 1, default: 1.0)
- `offset_min_sum_beta`: β for offset min-sum (mutually exclusive with α)
- `dtype_messages`: "fp16" or "fp32" for message precision (default: "fp16")
- `device`: "cuda" or "rocm" for GPU backend (default: "cuda")
- `seed`: RNG seed for γ sampling (default: 1234)
- `bitpack_output`: whether to return packed error bits for memory efficiency (default: False)
- `algo`: "relay" or "plain" algorithm mode (default: "relay")
- `perf`: "default", "throughput", or "realtime" performance mode (default: "default")
- `explicit_gammas`: optional array of shape (K, V) providing per-leg, per-variable γ values; if set, leg `l` uses row `l % K` instead of uniform sampling

**Methods:**
- `decode(syndromes)`: Decode batch of syndromes
  - Input: `syndromes` [B, C] syndrome bits (uint8)
  - Output: Dictionary with keys:
    - `"errors"`: [B, V] decoded errors (uint8) or packed if bitpack_output=True
    - `"weights"`: [B] solution weights (float32)
    - `"valid_mask"`: [B] valid solution mask (bool)
    - `"iterations"`: [B] number of iterations for each batch element (int32)

#### `RelayDecoder` (Adapter)

Adapter class for compatibility with the `relay_bp` package interface.

#### `ObservableDecoderRunner` (Adapter)

Batch processing class with observable decoding and logical error detection.

### Utility Functions

#### `CSRGraph`

GPU-optimized graph representation for sparse check matrices.

#### `bitpack_errors` / `bitunpack_errors`

Memory-efficient error pattern packing/unpacking utilities.

## Algorithm Details

The Relay-BP-S algorithm implements three key innovations for quantum error correction:

### 1. Disordered Memory Strengths (Gamma Mixing)
Breaks trapping sets by diversifying memory dynamics across ensemble members. Each relay leg uses a different memory strength γ sampled from a uniform distribution, preventing the decoder from getting stuck in local minima.

### 2. Ensemble Decoding
Explores broader solution space through multiple decoding attempts with different memory configurations, increasing the probability of finding valid solutions.

### 3. Relaying
Shares ensemble posteriors to accelerate convergence, using information from previous decoding attempts to guide subsequent ones.

### Decode Loop

1. **Pre-iterations (T₀)**: Run belief propagation with uniform memory strength γ₀
2. **Relay legs (R times)**:
   - Use explicit γ per leg if provided, else sample disordered γ values from a uniform distribution
   - Run belief propagation for Tᵣ iterations with memory mixing
   - Track best solutions (up to S solutions)
3. **Solution selection**: Return solution with minimum log-likelihood weight

### GPU Implementation

The Triton implementation uses optimized kernels for:
- **Check-to-variable message passing**: Two-pass min-sum algorithm with syndrome integration
- **Variable-to-check message passing**: Fused gamma mixing with memory-based updates
- **Edge-centric processing**: Efficient sparse matrix operations on GPU
- **Batch processing**: Parallel decoding of multiple syndromes

## Performance Features

- **Memory bandwidth optimized**: Coalesced loads and contiguous memory layout
- **Kernel autotuning**: Automatic optimization of launch parameters for different GPU architectures
- **Persistent kernels**: Realtime mode with continuous kernel execution
- **Batch processing**: Efficient parallel decoding with configurable batch sizes
- **Device resident**: No host↔device copies during decoding loop

## Testing and Validation

The implementation includes comprehensive equivalence testing against the Rust reference:

```bash
# Run equivalence tests
pytest tests/test_triton_against_rust.py

# Run performance tests
pytest tests/test_perf_real_circuit.py

# Run observable decoding tests
pytest tests/test_observable_decoding.py
```

Test coverage includes:
- Exact equivalence on small quantum codes
- Weight-based equivalence for different valid solutions
- Batch processing consistency
- Memory format validation (bitpacking)
- Numerical precision testing (fp16/fp32)
- Deterministic behavior verification
- Ensemble decoding validation
- High-degree constraint stress testing
- Non-convergence handling

## Examples

### Minimal Example
See `examples/minimal.py` for a complete working example with the 3-variable repetition code.

### Paper Study Replication
Use `relay_bp_paper_study.py` to replicate the methodology from the original paper:

```python
python relay_bp_paper_study.py --backend triton --dtype fp16
```

### Detailed Performance Analysis
Use `relay_bp_detailed.py` for comprehensive performance analysis:

```python
python relay_bp_detailed.py --backend triton --dtype fp16 --num_shots 10000
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Triton 2.0+
- CUDA-capable GPU (Compute Capability 7.0+) or ROCm-compatible GPU
- NumPy, SciPy
- Optional: `relay_bp` package for equivalence testing

## Architecture

The implementation consists of several key components:

- **`decoder.py`**: Main decoder class and orchestration logic
- **`kernels.py`**: Triton GPU kernels for belief propagation
- **`graph.py`**: GPU-optimized sparse matrix representation
- **`utils.py`**: Utility functions for error priors, gamma sampling, and validation
- **`adapter.py`**: Interface compatibility layer for `relay_bp` package

## License

Apache 2.0

## References

This implementation is based on the Relay-BP-S algorithm described in the original paper. The GPU acceleration maintains algorithmic fidelity with the Rust reference implementation while providing significant performance improvements for high-throughput quantum error correction applications.