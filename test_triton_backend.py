#!/usr/bin/env python3
"""
Test script for Relay-BP-S Triton backend.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'relay_bp_triton'))

import torch
import numpy as np
from scipy.sparse import csr_matrix

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from relay_bp_triton import RelayBPDecoder, CSRGraph
        from relay_bp_triton.utils import compute_log_prior_ratios
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_graph_creation():
    """Test graph creation from CSR matrix."""
    print("\nTesting graph creation...")
    
    try:
        # Create simple check matrix
        H_dense = np.array([
            [1, 1, 0],
            [0, 1, 1]
        ], dtype=np.uint8)
        H_csr = csr_matrix(H_dense)
        
        from relay_bp_triton import CSRGraph
        
        # Use CUDA if available, otherwise skip
        if torch.cuda.is_available():
            graph = CSRGraph(H_csr, device="cuda")
            print(f"✓ Graph created: {graph.C} checks, {graph.V} variables, {graph.E} edges")
            print(f"  Max degree check: {graph.max_deg_chk}, Max degree variable: {graph.max_deg_var}")
            
            # Validate graph
            graph.validate()
            print("✓ Graph validation passed")
        else:
            print("⚠ CUDA not available, skipping graph creation test (Triton requires GPU)")
        
        return True
    except Exception as e:
        print(f"✗ Graph creation failed: {e}")
        return False

def test_decoder_creation():
    """Test decoder creation."""
    print("\nTesting decoder creation...")
    
    try:
        # Create check matrix
        H_dense = np.array([
            [1, 1, 0],
            [0, 1, 1]
        ], dtype=np.uint8)
        H_csr = csr_matrix(H_dense)
        
        # Error priors
        error_priors = np.array([0.01, 0.01, 0.01], dtype=np.float64)
        
        from relay_bp_triton import RelayBPDecoder
        
        # Test CUDA decoder if available (Triton requires GPU)
        if torch.cuda.is_available():
            decoder_cuda = RelayBPDecoder(
                H_csr=H_csr,
                error_priors=error_priors,
                pre_iter=5,
                num_sets=2,
                set_max_iter=5,
                gamma0=0.1,
                gamma_dist_interval=(-0.1, 0.1),
                stop_nconv=1,
                dtype_messages="fp32",
                device="cuda",
                seed=42
            )
            print("✓ CUDA decoder created successfully")
        else:
            print("⚠ CUDA not available, skipping decoder creation test (Triton requires GPU)")
        
        return True
    except Exception as e:
        print(f"✗ Decoder creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_simple_decode():
    """Test simple decoding."""
    print("\nTesting simple decoding...")
    
    try:
        # Create check matrix
        H_dense = np.array([
            [1, 1, 0],
            [0, 1, 1]
        ], dtype=np.uint8)
        H_csr = csr_matrix(H_dense)
        
        # Error priors
        error_priors = np.array([0.01, 0.01, 0.01], dtype=np.float64)
        
        from relay_bp_triton import RelayBPDecoder
        
        # Use CUDA if available, otherwise skip this test
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if device == "cpu":
            print("⚠ CUDA not available, skipping decoding test (Triton requires GPU)")
            return True
        
        # Create decoder
        decoder = RelayBPDecoder(
            H_csr=H_csr,
            error_priors=error_priors,
            pre_iter=5,
            num_sets=2,
            set_max_iter=5,
            gamma0=0.1,
            gamma_dist_interval=(-0.1, 0.1),
            stop_nconv=1,
            dtype_messages="fp32",
            device=device,
            seed=42
        )
        
        # Test single syndrome
        syndrome = torch.tensor([[1, 0]], dtype=torch.uint8, device=device)
        result = decoder.decode(syndrome)
        
        print(f"✓ Decoding successful")
        print(f"  Input syndrome: {syndrome[0].cpu().numpy()}")
        print(f"  Decoded errors: {result['errors'][0].cpu().numpy()}")
        print(f"  Weight: {result['weights'][0].item():.3f}")
        print(f"  Valid: {result['valid_mask'][0].item()}")
        
        return True
    except Exception as e:
        print(f"✗ Simple decoding failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("Relay-BP-S Triton Backend Test Suite")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_graph_creation,
        test_decoder_creation,
        test_simple_decode,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
