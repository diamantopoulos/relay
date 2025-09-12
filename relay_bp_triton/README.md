# Relay-BP-S Triton GPU Backend

Ultra-fast GPU implementation of the Relay-BP-S algorithm for quantum error correction decoding using PyTorch and Triton kernels.

## Features

- **Device-resident decoding**: No host↔device copies inside decode loop
- **Triton kernels**: Optimized GPU kernels for belief propagation
- **Batch processing**: Efficient parallel decoding of multiple syndromes
- **Memory efficient**: Sparse matrix representation with edge-centric layout
- **Flexible precision**: Support for fp16/fp32 message precision
- **Autotuning**: Automatic kernel parameter optimization
- **Clean API**: Easy integration with SciPy CSR matrices

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
import torch
import numpy as np
from scipy.sparse import csr_matrix
from relay_bp_triton import RelayBPDecoder

# Create check matrix (3 variables, 2 checks)
H_csr = csr_matrix(np.array([
    [1, 1, 0],
    [0, 1, 1]
], dtype=np.uint8))

# Error priors
error_priors = np.array([0.01, 0.01, 0.01])

# Create decoder
decoder = RelayBPDecoder(
    H_csr=H_csr,
    error_priors=error_priors,
    pre_iter=80,
    num_sets=100,
    set_max_iter=60,
    gamma0=0.65,
    gamma_dist_interval=(-0.24, 0.66),
    stop_nconv=5,
    dtype_messages="fp16",
    device="cuda"
)

# Decode syndromes
syndromes = torch.tensor([[1, 1], [0, 1]], dtype=torch.uint8, device="cuda")
result = decoder.decode(syndromes)

print("Decoded errors:", result["errors"])
print("Weights:", result["weights"])
print("Valid solutions:", result["valid_mask"])
```

## API Reference

### RelayBPDecoder

Main decoder class implementing the Relay-BP-S algorithm.

#### Parameters

- `H_csr`: Check matrix (C × V) in SciPy CSR format
- `error_priors`: [V] error probabilities in (0, 0.5)
- `pre_iter`: T₀ - iterations for first set (default: 80)
- `num_sets`: R - number of relay legs/ensembles (default: 100)
- `set_max_iter`: Tᵣ - iterations per relay set (default: 60)
- `gamma0`: γ for first set - ordered memory (default: 0.65)
- `gamma_dist_interval`: (min, max) for disordered γ sampling (default: (-0.24, 0.66))
- `stop_nconv`: S - number of solutions to collect (default: 5)
- `normalized_min_sum_alpha`: α for normalized min-sum (0 < α ≤ 1)
- `offset_min_sum_beta`: β for offset min-sum (mutually exclusive with α)
- `dtype_messages`: "fp16" or "fp32" for message precision (default: "fp16")
- `device`: "cuda" or "rocm" (default: "cuda")
- `seed`: RNG seed for γ sampling (default: 1234)
- `bitpack_output`: whether to return packed error bits (default: False)

#### Methods

- `decode(syndromes)`: Decode batch of syndromes
  - Input: `syndromes` [B, C] syndrome bits (uint8)
  - Output: Dictionary with keys:
    - `"errors"`: [B, V] decoded errors (uint8) or packed if bitpack_output=True
    - `"weights"`: [B] solution weights (float32)
    - `"valid_mask"`: [B] valid solution mask (bool)

- `get_stats()`: Get decoder statistics and configuration

## Algorithm

The Relay-BP-S algorithm implements three key innovations:

1. **Disordered Memory Strengths**: Breaks trapping sets by diversifying memory dynamics
2. **Ensembling**: Explores broader solution space through multiple decoding attempts
3. **Relaying**: Shares ensemble posteriors to accelerate convergence

### Decode Loop

1. **Pre-iterations (T₀)**: Run BP with uniform memory strength γ₀
2. **Relay legs (R times)**:
   - Sample disordered γ values from uniform distribution
   - Run BP for Tᵣ iterations
   - Track best solutions (up to S solutions)
3. **Solution selection**: Return solution with minimum weight

## Performance

- **Memory bandwidth optimized**: Coalesced loads and contiguous memory layout
- **Kernel autotuning**: Automatic optimization of launch parameters
- **Batch processing**: Efficient parallel decoding
- **Device resident**: No host↔device copies during decoding

## Examples

See `examples/minimal.py` for a complete working example with the 3-variable repetition code.

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Triton 2.0+
- CUDA-capable GPU or ROCm-compatible GPU
- NumPy, SciPy

## License

Apache 2.0
