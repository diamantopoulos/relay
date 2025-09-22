# (C) Copyright IBM 2025
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Utility functions for Relay-BP-S decoding."""

import torch
import numpy as np
from typing import Tuple, Optional


def bitpack_errors(errors: torch.Tensor, bits_per_word: int = 32) -> torch.Tensor:
    """Pack error bits into words for memory efficiency.
    
    Args:
        errors: [B, V] binary tensor (0/1)
        bits_per_word: number of bits per packed word
        
    Returns:
        [B, (V + bits_per_word - 1) // bits_per_word] packed tensor
    """
    B, V = errors.shape
    words_per_batch = (V + bits_per_word - 1) // bits_per_word
    
    # Reshape to [B, words_per_batch, bits_per_word]
    padded_errors = torch.zeros(B, words_per_batch * bits_per_word, 
                               dtype=errors.dtype, device=errors.device)
    padded_errors[:, :V] = errors
    
    # Reshape and pack
    packed = padded_errors.view(B, words_per_batch, bits_per_word)
    
    # Convert to uint32 words
    result = torch.zeros(B, words_per_batch, dtype=torch.uint32, device=errors.device)
    
    for i in range(bits_per_word):
        result |= (packed[:, :, i].to(torch.uint32) << i)
    
    return result


def bitunpack_errors(packed: torch.Tensor, V: int, bits_per_word: int = 32) -> torch.Tensor:
    """Unpack error bits from words.
    
    Args:
        packed: [B, words_per_batch] packed tensor
        V: number of variables
        bits_per_word: number of bits per packed word
        
    Returns:
        [B, V] binary tensor (0/1)
    """
    B, words_per_batch = packed.shape
    
    # Convert to binary representation
    bits = torch.zeros(B, words_per_batch, bits_per_word, 
                      dtype=torch.uint8, device=packed.device)
    
    for i in range(bits_per_word):
        bits[:, :, i] = ((packed >> i) & 1).to(torch.uint8)
    
    # Flatten and truncate to V
    errors = bits.view(B, -1)[:, :V]
    
    return errors


def compute_log_prior_ratios(error_priors: np.ndarray) -> np.ndarray:
    """Compute log prior ratios from error probabilities.
    
    Args:
        error_priors: [V] error probabilities in (0, 0.5)
        
    Returns:
        [V] log prior ratios log((1-p)/p)
    """
    # Clip probabilities to avoid numerical issues
    p = np.clip(error_priors, 1e-10, 0.5 - 1e-10)
    
    # Compute log prior ratios
    log_ratios = np.log((1 - p) / p)
    
    return log_ratios.astype(np.float32)


def compute_decoding_weights(hard_dec: torch.Tensor, log_prior_ratios: torch.Tensor) -> torch.Tensor:
    """Compute decoding weights for solution selection.
    
    Args:
        hard_dec: [B, V] hard decisions (0/1)
        log_prior_ratios: [V] log prior ratios
        
    Returns:
        [B] decoding weights (lower is better)
    """
    # Weight = sum of log prior ratios for error bits
    weights = torch.sum(hard_dec.float() * log_prior_ratios.unsqueeze(0), dim=1)
    
    return weights


def sample_gamma_uniform(
    B: int, V: int, 
    gamma_min: float, gamma_max: float,
    device: str, seed: Optional[int] = None
) -> torch.Tensor:
    """Sample gamma values from uniform distribution.
    
    Args:
        B: batch size
        V: number of variables
        gamma_min: minimum gamma value
        gamma_max: maximum gamma value
        device: target device
        seed: random seed (optional)
        
    Returns:
        [B, V] gamma values
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    # Sample uniform random values
    uniform = torch.rand(B, V, device=device)
    
    # Scale to [gamma_min, gamma_max]
    gamma = gamma_min + (gamma_max - gamma_min) * uniform
    
    return gamma


def sample_gamma_scalar(
    B: int, V: int,
    gamma_value: float,
    device: str
) -> torch.Tensor:
    """Sample scalar gamma values (broadcast across variables).
    
    Args:
        B: batch size
        V: number of variables
        gamma_value: gamma value
        device: target device
        
    Returns:
        [B, V] gamma values (all the same)
    """
    return torch.full((B, V), gamma_value, device=device)


def validate_error_priors(error_priors: np.ndarray) -> np.ndarray:
    """Validate and clip error priors.
    
    Args:
        error_priors: [V] error probabilities
        
    Returns:
        [V] validated and clipped error probabilities
    """
    # Check shape
    if error_priors.ndim != 1:
        raise ValueError("error_priors must be 1D array")
    
    # Check range and clip
    clipped = np.clip(error_priors, 1e-10, 0.5 - 1e-10)
    
    # Warn if clipping occurred
    if np.any(clipped != error_priors):
        print("Warning: error_priors were clipped to avoid numerical issues")
    
    return clipped


def validate_csr_matrix(H_csr) -> bool:
    """Validate CSR matrix format and properties; coerce CSR if needed."""
    from scipy.sparse import csr_matrix, issparse

    if not issparse(H_csr):
        raise ValueError("Input must be a scipy.sparse matrix (any format)")

    # Coerce to CSR if it's e.g. csr_array/coo/csc/etc.
    if not isinstance(H_csr, csr_matrix):
        H_csr = H_csr.tocsr()

    if H_csr.shape[0] == 0 or H_csr.shape[1] == 0:
        raise ValueError("Matrix cannot be empty")
    if H_csr.nnz == 0:
        raise ValueError("Matrix cannot have zero non-zero elements")
    if not np.all((H_csr.data == 0) | (H_csr.data == 1)):
        print("Warning: Matrix contains non-binary values, will be treated as binary")

    return True



def get_device_info(device: str) -> dict:
    """Get device information for debugging.
    
    Args:
        device: device string ("cuda" or "rocm")
        
    Returns:
        Dictionary with device information
    """
    if device == "cuda":
        if torch.cuda.is_available():
            return {
                "device": device,
                "available": True,
                "name": torch.cuda.get_device_name(),
                "memory_total": torch.cuda.get_device_properties(0).total_memory,
                "memory_allocated": torch.cuda.memory_allocated(),
                "memory_reserved": torch.cuda.memory_reserved(),
            }
        else:
            return {"device": device, "available": False}
    
    elif device == "rocm":
        # TODO: Add ROCm device info
        return {"device": device, "available": torch.cuda.is_available()}
    
    else:
        raise ValueError(f"Unsupported device: {device}")


def format_memory_size(bytes: int) -> str:
    """Format memory size in human-readable format.
    
    Args:
        bytes: memory size in bytes
        
    Returns:
        Formatted string
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes < 1024.0:
            return f"{bytes:.1f} {unit}"
        bytes /= 1024.0
    return f"{bytes:.1f} PB"
