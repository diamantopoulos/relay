# (C) Copyright IBM 2025
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Graph utilities for converting SciPy CSR matrices to device-optimized edge-centric representation."""

import torch
import numpy as np
from scipy.sparse import csr_matrix
from typing import Tuple, Optional

from scipy.sparse import csr_matrix, issparse

class CSRGraph:
    def __init__(self, H_csr: csr_matrix, device: str = "cuda"):
        if device == "cpu":
            raise RuntimeError("CPU not supported - Triton kernels require GPU. Use device='cuda' or 'rocm'")

        # Accept any scipy sparse, ensure CSR matrix
        if not issparse(H_csr):
            raise ValueError("Input must be a scipy.sparse matrix")
        if not isinstance(H_csr, csr_matrix):
            H_csr = H_csr.tocsr()

        self.device = device
        self.C, self.V = H_csr.shape
        self.E = H_csr.nnz
        
        # Validate input
        if not isinstance(H_csr, csr_matrix):
            raise ValueError("Input must be a scipy.sparse.csr_matrix")
        
        if H_csr.dtype != np.uint8 and H_csr.dtype != bool:
            # Convert to binary if needed
            H_csr = (H_csr != 0).astype(np.uint8)
        
        # Convert to device tensors
        self._build_edge_representation(H_csr)
    
    def _build_edge_representation(self, H_csr: csr_matrix):
        """Build edge-centric representation from CSR matrix."""
        # Get CSR data
        chk_ptr = H_csr.indptr.astype(np.int32)
        chk_edges = H_csr.indices.astype(np.int32)  # These are variable indices
        
        # Build variable-centric representation
        var_ptr = np.zeros(self.V + 1, dtype=np.int32)
        var_edges = np.zeros(self.E, dtype=np.int32)
        edge_var = np.zeros(self.E, dtype=np.int32)
        edge_chk = np.zeros(self.E, dtype=np.int32)
        
        # Build edge list and variable-centric pointers
        edge_idx = 0
        for chk_idx in range(self.C):
            start = chk_ptr[chk_idx]
            end = chk_ptr[chk_idx + 1]
            
            for local_idx in range(start, end):
                var_idx = chk_edges[local_idx]
                
                # Add to variable's edge list
                var_ptr[var_idx + 1] += 1
                
                # Store edge information
                edge_var[edge_idx] = var_idx
                edge_chk[edge_idx] = chk_idx
                var_edges[edge_idx] = edge_idx  # Will be updated after sorting
                
                edge_idx += 1
        
        # Convert to cumulative pointers for variables
        for i in range(1, self.V + 1):
            var_ptr[i] += var_ptr[i - 1]
        
        # Sort edges by variable index and update var_edges
        edge_order = np.argsort(edge_var)
        for i, edge_idx in enumerate(edge_order):
            var_idx = edge_var[edge_idx]
            var_edges[var_ptr[var_idx]] = edge_idx
            var_ptr[var_idx] += 1
        
        # Reset var_ptr to start positions
        for i in range(self.V - 1, 0, -1):
            var_ptr[i] = var_ptr[i - 1]
        var_ptr[0] = 0
        
        # Now build chk_edges as edge IDs (not variable indices)
        chk_edges_edge_ids = np.zeros(self.E, dtype=np.int32)
        edge_idx = 0
        for chk_idx in range(self.C):
            start = chk_ptr[chk_idx]
            end = chk_ptr[chk_idx + 1]
            
            for local_idx in range(start, end):
                chk_edges_edge_ids[local_idx] = edge_idx
                edge_idx += 1
        
        # Move to device
        self.chk_ptr = torch.from_numpy(chk_ptr).to(self.device)
        self.chk_edges = torch.from_numpy(chk_edges_edge_ids).to(self.device)  # Edge IDs
        self.var_ptr = torch.from_numpy(var_ptr).to(self.device)
        self.var_edges = torch.from_numpy(var_edges).to(self.device)
        self.edge_var = torch.from_numpy(edge_var).to(self.device)
        self.edge_chk = torch.from_numpy(edge_chk).to(self.device)
        
        # Compute degree statistics for autotuning
        self.max_deg_chk = int((chk_ptr[1:] - chk_ptr[:-1]).max())
        self.max_deg_var = int((var_ptr[1:] - var_ptr[:-1]).max())
    
    def get_device_tensors(self) -> Tuple[torch.Tensor, ...]:
        """Get all device tensors for kernel calls.
        
        Returns:
            Tuple of (chk_ptr, chk_edges, var_ptr, var_edges, edge_var, edge_chk)
        """
        return (self.chk_ptr, self.chk_edges, self.var_ptr, 
                self.var_edges, self.edge_var, self.edge_chk)
    
    def validate(self) -> bool:
        """Validate graph structure."""
        # Check that all edge indices are valid
        assert self.edge_var.max() < self.V, "Invalid variable index in edges"
        assert self.edge_chk.max() < self.C, "Invalid check index in edges"
        
        # Check that pointers are monotonic
        assert torch.all(self.chk_ptr[1:] >= self.chk_ptr[:-1]), "Invalid check pointers"
        assert torch.all(self.var_ptr[1:] >= self.var_ptr[:-1]), "Invalid variable pointers"
        
        # Check that edge counts match
        assert self.chk_ptr[-1] == self.E, "Check pointer mismatch"
        assert self.var_ptr[-1] == self.E, "Variable pointer mismatch"
        
        return True
    
    def get_stats(self) -> dict:
        """Get graph statistics for debugging."""
        return {
            "checks": self.C,
            "variables": self.V,
            "edges": self.E,
            "max_degree_check": self.max_deg_chk,
            "max_degree_variable": self.max_deg_var,
            "avg_degree_check": self.E / self.C,
            "avg_degree_variable": self.E / self.V,
        }
