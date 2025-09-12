#!/usr/bin/env python3
"""
Minimal example of Relay-BP-S Triton decoder.

This example mirrors the tiny 3-var / 2-check repetition code from the Rust implementation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from scipy.sparse import csr_matrix
from relay_bp_triton import RelayBPDecoder


def create_repetition_code():
    """Create a 3-variable, 2-check repetition code.
    
    Check matrix H:
    [1 1 0]
    [0 1 1]
    
    This is a simple repetition code where:
    - Check 0: v0 + v1 = 0 (mod 2)
    - Check 1: v1 + v2 = 0 (mod 2)
    """
    H_dense = np.array([
        [1, 1, 0],
        [0, 1, 1]
    ], dtype=np.uint8)
    
    H_csr = csr_matrix(H_dense)
    return H_csr


def test_all_error_patterns():
    """Test all possible error patterns for the repetition code."""
    print("=" * 60)
    print("Relay-BP-S Triton Decoder - Minimal Example")
    print("=" * 60)
    print()
    
    # Create check matrix
    H_csr = create_repetition_code()
    print("Check matrix H:")
    print(H_csr.toarray())
    print()
    
    # Error priors (low error rate for testing)
    error_priors = np.array([0.01, 0.01, 0.01], dtype=np.float64)
    print(f"Error priors: {error_priors}")
    print()
    
    # Create decoder
    decoder = RelayBPDecoder(
        H_csr=H_csr,
        error_priors=error_priors,
        pre_iter=10,  # Reduced for testing
        num_sets=3,   # Reduced for testing
        set_max_iter=10,
        gamma0=0.1,
        gamma_dist_interval=(-0.1, 0.1),
        stop_nconv=1,
        normalized_min_sum_alpha=0.9,
        dtype_messages="fp32",
        device="cuda" if torch.cuda.is_available() else "cpu",
        seed=42
    )
    
    print("Decoder created successfully!")
    print(f"Device: {decoder.device}")
    print(f"Graph stats: {decoder.graph.get_stats()}")
    print()
    
    # Test all possible error patterns
    print("Testing all error patterns:")
    print("-" * 40)
    
    all_errors = []
    all_syndromes = []
    
    for error_pattern in range(2**3):  # 2^3 = 8 possible patterns
        # Convert to binary
        errors = np.array([(error_pattern >> i) & 1 for i in range(3)], dtype=np.uint8)
        all_errors.append(errors)
        
        # Compute syndrome
        syndrome = (H_csr @ errors) % 2
        all_syndromes.append(syndrome)
        
        print(f"Error pattern {error_pattern:03b}: {errors} -> syndrome: {syndrome}")
    
    print()
    
    # Batch decode all syndromes
    print("Batch decoding all syndromes:")
    print("-" * 40)
    
    syndromes_tensor = torch.tensor(all_syndromes, dtype=torch.uint8, device=decoder.device)
    
    # Decode
    result = decoder.decode(syndromes_tensor)
    
    errors_decoded = result["errors"]
    weights = result["weights"]
    valid_mask = result["valid_mask"]
    
    print(f"Decoded errors shape: {errors_decoded.shape}")
    print(f"Weights shape: {weights.shape}")
    print(f"Valid mask shape: {valid_mask.shape}")
    print()
    
    # Compare results
    print("Results comparison:")
    print("-" * 40)
    print(f"{'Pattern':<8} {'True':<8} {'Decoded':<8} {'Weight':<10} {'Valid':<6} {'Correct':<8}")
    print("-" * 60)
    
    correct_count = 0
    for i, true_errors in enumerate(all_errors):
        decoded_errors = errors_decoded[i].cpu().numpy()
        weight = weights[i].item()
        valid = valid_mask[i].item()
        correct = np.array_equal(true_errors, decoded_errors)
        
        if correct:
            correct_count += 1
        
        print(f"{i:03b}      {true_errors}    {decoded_errors}    {weight:8.3f}    {valid}      {correct}")
    
    print("-" * 60)
    print(f"Correct decodings: {correct_count}/{len(all_errors)} ({100*correct_count/len(all_errors):.1f}%)")
    print()
    
    # Test with some specific cases
    print("Specific test cases:")
    print("-" * 40)
    
    # Test case 1: No errors
    syndrome1 = torch.tensor([[0, 0]], dtype=torch.uint8, device=decoder.device)
    result1 = decoder.decode(syndrome1)
    print(f"No errors: syndrome [0,0] -> errors {result1['errors'][0].cpu().numpy()}, valid: {result1['valid_mask'][0].item()}")
    
    # Test case 2: Single error
    syndrome2 = torch.tensor([[1, 0]], dtype=torch.uint8, device=decoder.device)
    result2 = decoder.decode(syndrome2)
    print(f"Single error: syndrome [1,0] -> errors {result2['errors'][0].cpu().numpy()}, valid: {result2['valid_mask'][0].item()}")
    
    # Test case 3: Two errors
    syndrome3 = torch.tensor([[1, 1]], dtype=torch.uint8, device=decoder.device)
    result3 = decoder.decode(syndrome3)
    print(f"Two errors: syndrome [1,1] -> errors {result3['errors'][0].cpu().numpy()}, valid: {result3['valid_mask'][0].item()}")
    
    print()
    print("Minimal example completed successfully!")


def test_deterministic_mode():
    """Test deterministic mode (num_sets=1, gamma_dist_interval=(0,0))."""
    print("\n" + "=" * 60)
    print("Testing Deterministic Mode")
    print("=" * 60)
    
    H_csr = create_repetition_code()
    error_priors = np.array([0.01, 0.01, 0.01], dtype=np.float64)
    
    # Create deterministic decoder
    decoder = RelayBPDecoder(
        H_csr=H_csr,
        error_priors=error_priors,
        pre_iter=5,
        num_sets=1,  # Single set
        set_max_iter=5,
        gamma0=0.1,
        gamma_dist_interval=(0.0, 0.0),  # No randomness
        stop_nconv=1,
        dtype_messages="fp32",
        device="cuda" if torch.cuda.is_available() else "cpu",
        seed=42
    )
    
    # Test same syndrome multiple times
    syndrome = torch.tensor([[1, 0]], dtype=torch.uint8, device=decoder.device)
    
    print("Running same syndrome 3 times (should be identical):")
    for i in range(3):
        result = decoder.decode(syndrome)
        errors = result['errors'][0].cpu().numpy()
        weight = result['weights'][0].item()
        valid = result['valid_mask'][0].item()
        print(f"  Run {i+1}: errors {errors}, weight {weight:.3f}, valid {valid}")
    
    print("Deterministic mode test completed!")


if __name__ == "__main__":
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name()}")
    else:
        print("CUDA not available, using CPU")
    
    print()
    
    # Run tests
    test_all_error_patterns()
    test_deterministic_mode()
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
