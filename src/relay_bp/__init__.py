# (C) Copyright IBM 2025
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""Base imports for relay bp library."""

from .bp import *
from .decoder import *
from .observable_decoder import *

# Triton GPU backend (optional)
try:
    from .triton import RelayBPDecoder, RelayDecoder as TritonRelayDecoder, ObservableDecoderRunner as TritonObservableDecoderRunner
    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False
    RelayBPDecoder = None
    TritonRelayDecoder = None
    TritonObservableDecoderRunner = None

# Backend selection utilities
def get_available_backends():
    """Get list of available backends."""
    backends = ["rust"]
    if _HAS_TRITON:
        backends.append("triton")
    return backends

def select_decoder(backend="rust", dtype="fp32", algorithm="relay"):
    """Select appropriate decoder class for the given backend and dtype.
    
    Args:
        backend: "rust" or "triton"
        dtype: "fp16", "fp32", or "fp64" (availability depends on backend)
        algorithm: "relay" or "plain" (only "relay" supported for triton)
        
    Returns:
        Decoder class appropriate for the backend and dtype
        
    Raises:
        ValueError: If requested backend/dtype combination is unavailable
        ImportError: If required dependencies are not installed
    """
    if backend == "triton":
        if not _HAS_TRITON:
            raise ImportError("Triton backend not available. Install with: pip install relay-bp[triton]")
        if dtype not in ("fp16", "fp32"):
            raise ValueError("Triton backend supports only dtype in {'fp16', 'fp32'}")
        if algorithm != "relay":
            raise ValueError("Triton backend currently only supports 'relay' algorithm")
        return TritonRelayDecoder  # Triton adapter class
        
    elif backend == "rust":
        if dtype not in ("fp32", "fp64"):
            raise ValueError("Rust backend supports only dtype in {'fp32', 'fp64'}")
        
        if algorithm == "relay":
            if dtype == "fp32":
                return RelayDecoderF32
            elif dtype == "fp64":
                return RelayDecoderF64
        elif algorithm == "plain":
            if dtype == "fp32":
                return MinSumBPDecoderF32
            elif dtype == "fp64":
                return MinSumBPDecoderF64
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
    else:
        raise ValueError(f"Unknown backend: {backend}. Available: {get_available_backends()}")

# Add backend info to module
__version__ = "0.1.0"
__all__ = [
    # Rust classes
    "RelayDecoderF32", "RelayDecoderF64", "RelayDecoderI32", "RelayDecoderI64",
    "MinSumBPDecoderF32", "MinSumBPDecoderF64", "MinSumBPDecoderI8", 
    "MinSumBPDecoderI16", "MinSumBPDecoderI32", "MinSumBPDecoderI64", "MinSumBPDecoderFixed",
    "DecodeResult", "ObservableDecoderRunner", "ObservableDecodeResult",
    # Triton classes (if available)
    "RelayBPDecoder", "TritonRelayDecoder", "TritonObservableDecoderRunner",
    # Utility functions
    "get_available_backends", "select_decoder",
]
