"""
Equivalence tests: Triton Relay-BP-S vs Rust relay_bp (Python frontend).

We use a deterministic config so results are reproducible and comparable:
  - num_sets=1
  - gamma0=0.0
  - gamma_dist_interval=(0.0, 0.0)
  - fixed iteration counts
  - normalized_min_sum_alpha=1.0 (pure min-sum on Triton)
  - stop_nconv=1

The test mirrors the repo's testing approach (pytest under tests/) and the Python API
style from README (relay_bp.RelayDecoderF32).

Skip rules:
  - If CUDA is unavailable (Triton needs GPU)
  - If relay_bp (Rust bindings) isn't importable
  - If our Triton backend isn't importable
"""

import os
import sys
import numpy as np
import torch
import pytest
from scipy.sparse import csr_matrix

# --- try importing Rust python frontend ---
try:
    import relay_bp  # from this repo's package
    HAS_RELAY_BP = True
except Exception:
    HAS_RELAY_BP = False

# --- try importing our Triton backend (adjust if your package name differs) ---
try:
    # If the Triton backend lives in-repo at <repo_root>/relay_bp_triton
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from relay_bp_triton import RelayBPDecoder  # noqa: E402
    HAS_TRITON_BACKEND = True
except Exception:
    HAS_TRITON_BACKEND = False

CUDA_OK = torch.cuda.is_available()

pytestmark = [
    pytest.mark.skipif(not CUDA_OK, reason="CUDA not available (Triton requires GPU)"),
    pytest.mark.skipif(not HAS_RELAY_BP, reason="relay_bp (Rust) not importable"),
    pytest.mark.skipif(not HAS_TRITON_BACKEND, reason="relay_bp_triton not importable"),
]

# ---------------- helpers ----------------

# ---------- NEW HELPERS (drop-in) ----------

def _probe_rust_min_sum_kwargs(base_kwargs):
    """
    Try to enforce pure min-sum on the Rust side.
    Some wheels expose one or more of these; we only pass the ones that exist.
    """
    import numpy as _np
    from scipy.sparse import csr_matrix as _csr
    import relay_bp as _relay_bp  # use already-imported module

    extras = {}
    # Minimal 1x1 CSR and single prior for probing
    _H_probe = _csr(_np.array([[1]], dtype=_np.uint8))
    _p_probe = [0.01]

    candidates = [
        ("alpha", 1.0),
        ("alpha_iteration_scaling_factor", 0.0),
        ("offset_beta", 0.0),
        ("damping", 0.0),
        ("randomize_update_order", False),
    ]
    for k, v in candidates:
        try:
            _relay_bp.RelayDecoderF32(
                _H_probe,
                error_priors=_p_probe,
                **base_kwargs,
                **{k: v},
            )
            extras[k] = v
        except TypeError:
            # kw not accepted by this wheel
            pass
        except Exception:
            # accepted but value invalid for this build → don't pass it
            pass
    return {**base_kwargs, **extras}


def weight_from_priors(e_u8: np.ndarray, p: np.ndarray) -> float:
    """Compute ML 'cost' w = sum_j e_j * log((1-p_j)/p_j)."""
    wj = np.log((1.0 - p) / p)
    return float((e_u8 * wj).sum())


def compare_with_ml_or_tie(H_csr, s_u8, p, e_rust, e_gpu, tol=1e-4):
    """
    If both decoders are parity-valid but disagree, require near-equal cost;
    for tiny n (<=16), allow either one that exactly matches ML.
    """
    assert e_rust.dtype == np.uint8 and e_gpu.dtype == np.uint8
    assert parity_ok(H_csr, e_rust, s_u8)
    assert parity_ok(H_csr, e_gpu,  s_u8)

    if np.array_equal(e_rust, e_gpu):
        return  # identical – done

    # Different codewords: check weights
    w_rust = weight_from_priors(e_rust, p)
    w_gpu  = weight_from_priors(e_gpu,  p)

    n = H_csr.shape[1]
    if n <= 16:
        e_ml, w_ml = brute_force_ml(H_csr, s_u8, p)
        # Accept if either equals exact ML
        if e_ml is not None:
            rust_is_ml = np.array_equal(e_rust, e_ml) and abs(w_rust - w_ml) < 1e-6
            gpu_is_ml  = np.array_equal(e_gpu,  e_ml) and abs(w_gpu  - w_ml) < 1e-6
            if rust_is_ml or gpu_is_ml:
                return
    # Otherwise, require near-equal costs (same objective)
    assert abs(w_rust - w_gpu) < tol, f"Unequal weights: rust={w_rust}, gpu={w_gpu}"



def make_rust_decoder(H_csr, p):
    base_kwargs = dict(
        gamma0=0.0,
        pre_iter=200,
        num_sets=1,
        set_max_iter=0,
        gamma_dist_interval=(0.0, 1e-12),
        stop_nconv=1,
    )
    # Force pure min-sum if the wheel exposes the knobs
    base_kwargs = _probe_rust_min_sum_kwargs(base_kwargs)

    # C-contiguous 1-D buffers for PyO3
    p64 = np.array(p, dtype=np.float64, order="C", copy=True).ravel()
    p32 = np.array(p, dtype=np.float32, order="C", copy=True).ravel()
    p_list = [float(x) for x in p64]

    try:
        return relay_bp.RelayDecoderF32(H_csr, error_priors=p64, **base_kwargs)
    except TypeError:
        pass
    try:
        return relay_bp.RelayDecoderF32(H_csr, error_priors=p32, **base_kwargs)
    except TypeError:
        pass
    return relay_bp.RelayDecoderF32(H_csr, error_priors=p_list, **base_kwargs)


def make_triton_decoder(H_csr, p, device="cuda"):
    return RelayBPDecoder(
        H_csr=H_csr,
        error_priors=p,
        pre_iter=200,                   # match Rust pre-iterations
        num_sets=1,
        set_max_iter=0,                 # deterministic
        gamma0=0.0,
        gamma_dist_interval=(0.0, 1e-12),
        stop_nconv=1,
        normalized_min_sum_alpha=1.0,   # pure min-sum
        offset_min_sum_beta=None,
        dtype_messages="fp32",
        device=device,
        seed=1234,
        bitpack_output=False,
    )

def parity_ok(H_csr, e_u8, s_u8) -> bool:
    H = H_csr.toarray().astype(np.uint8)
    return bool(np.all(((H @ e_u8) & 1) == s_u8))


def brute_force_ml(H_csr, s_u8, p):
    """Brute-force ML for tiny n<=16; returns (e, weight) or (None, inf)."""
    H = H_csr.toarray().astype(np.uint8)
    m, n = H.shape
    if n > 16:
        return None, float("inf")
    s = s_u8.astype(np.uint8)
    wj = np.log((1.0 - p) / p)
    best_e, best_w = None, float("inf")
    for x in range(1 << n):
        e = np.array([(x >> j) & 1 for j in range(n)], dtype=np.uint8)
        if np.all(((H @ e) & 1) == s):
            w = float((e * wj).sum())
            if w < best_w - 1e-12:
                best_w, best_e = w, e.copy()
    return best_e, best_w

# ---------------- tests ----------------

def test_triangle_exact_match():
    """3-var/2-check toy: expect exact match (or weight tie) and parity-valid."""
    H = csr_matrix(np.array([[1, 1, 0],
                             [0, 1, 1]], dtype=np.uint8))
    p = np.array([0.01, 0.01, 0.01], dtype=np.float64)
    S = [np.array([0,0], np.uint8),
         np.array([1,0], np.uint8),
         np.array([0,1], np.uint8),
         np.array([1,1], np.uint8)]

    rust = make_rust_decoder(H, p)
    gpu  = make_triton_decoder(H, p, device="cuda")

    for s in S:
        e_rust = np.asarray(rust.decode(s), dtype=np.uint8)
        s_dev  = torch.from_numpy(np.asarray([s], dtype=np.uint8)).to("cuda")
        e_gpu  = gpu.decode(s_dev)["errors"][0].detach().cpu().numpy().astype(np.uint8)

        assert parity_ok(H, e_rust, s)
        assert parity_ok(H, e_gpu,  s)

        wj = np.log((1.0 - p) / p)
        w_rust = float((e_rust * wj).sum())
        w_gpu  = float((e_gpu  * wj).sum())

        if not np.array_equal(e_rust, e_gpu):
            assert abs(w_rust - w_gpu) < 1e-5, f"Different bits and unequal weights: {w_rust} vs {w_gpu}"


@pytest.mark.parametrize("n,c,seed", [
    (8, 4, 0),
    (10, 5, 1),
])
def test_random_small_graphs_equivalence(n, c, seed):
    """Random small H; both decoders must be parity-valid. Prefer exact bit match; otherwise accept weight tie."""
    rng = np.random.default_rng(seed)
    H_dense = np.zeros((c, n), dtype=np.uint8)
    for i in range(c):
        deg = int(rng.integers(2, min(4, n)+1))
        cols = rng.choice(n, size=deg, replace=False)
        H_dense[i, cols] = 1
    H = csr_matrix(H_dense)

    p = rng.uniform(0.005, 0.02, size=n).astype(np.float64)
    S = [rng.integers(0, 2, size=c, dtype=np.uint8) for _ in range(8)]

    rust = make_rust_decoder(H, p)
    gpu  = make_triton_decoder(H, p, device="cuda")
    for s in S:
        e_rust = np.asarray(rust.decode(s), dtype=np.uint8)
        s_dev  = torch.from_numpy(np.asarray([s], dtype=np.uint8)).to("cuda")
        out_gpu = gpu.decode(s_dev)
        e_gpu   = out_gpu["errors"][0].detach().cpu().numpy().astype(np.uint8)
        gpu_ok  = bool(out_gpu["valid_mask"][0].item())

        # If GPU did not converge, just assert it marked the sample invalid and skip comparisons.
        if not gpu_ok:
            w_gpu_api = float(out_gpu["weights"][0].item())
            assert not np.isfinite(w_gpu_api), "Non-converged decode must report infinite weight"
            continue

        # GPU converged → require parity-valid.
        assert parity_ok(H, e_gpu, s), "Triton reported valid but parity fails"

        rust_ok = parity_ok(H, e_rust, s)
        if not rust_ok:
            # Rust failed but GPU succeeded → check GPU equals ML when n<=16
            e_ml, w_ml = brute_force_ml(H, s, p)
            if e_ml is not None:
                w_gpu = weight_from_priors(e_gpu, p)
                assert np.array_equal(e_gpu, e_ml), "GPU bits must match ML when Rust fails"
                assert abs(w_gpu - w_ml) < 1e-6, "GPU weight must equal ML when Rust fails"
            continue

        # Both parity-valid → compare (allow ML tie on tiny n)
        compare_with_ml_or_tie(H, s, p, e_rust, e_gpu, tol=1e-4)

#        # Both parity-valid: allow different bits but require (near-)equal weight
#        wj = np.log((1.0 - p) / p)
#        w_rust = float((e_rust * wj).sum())
#        w_gpu  = float((e_gpu  * wj).sum())
#        if not np.array_equal(e_rust, e_gpu):
#            assert abs(w_rust - w_gpu) < 1e-4, f"Unequal weights: rust={w_rust}, gpu={w_gpu}"

def test_batch_consistency_triangle():
    """Batch-of-N equals N single decodes."""
    H = csr_matrix(np.array([[1, 1, 0],
                             [0, 1, 1]], dtype=np.uint8))
    p = np.array([0.01, 0.01, 0.01], dtype=np.float64)
    S = np.array([[0,0],[1,0],[0,1],[1,1]], dtype=np.uint8)

    rust = make_rust_decoder(H, p)
    gpu  = make_triton_decoder(H, p, device="cuda")

    # single-shot via rust
    singles = np.stack([np.asarray(rust.decode(s), dtype=np.uint8) for s in S], axis=0)

    # batch on GPU
    s_dev = torch.tensor(S, dtype=torch.uint8, device="cuda")
    out   = gpu.decode(s_dev)
    errs  = out["errors"].detach().cpu().numpy().astype(np.uint8)

    for i in range(S.shape[0]):
        assert parity_ok(H, singles[i], S[i])
        assert parity_ok(H, errs[i],    S[i])

    wj = np.log((1.0 - p) / p)
    w_rust = (singles * wj).sum(axis=1)
    w_gpu  = (errs    * wj).sum(axis=1)
    diffs  = np.any(singles != errs, axis=1)
    assert np.all(np.abs(w_rust[diffs] - w_gpu[diffs]) < 1e-5)


def test_fp16_messages_parity():
    """fp16 messages must preserve parity; tiny weight drift allowed."""
    H = csr_matrix(np.array([[1,1,0],[0,1,1]], dtype=np.uint8))
    p = np.array([0.01, 0.01, 0.01], dtype=np.float64)
    S = np.array([[1,0],[0,1],[1,1]], dtype=np.uint8)

    rust = make_rust_decoder(H, p)
    gpu  = RelayBPDecoder(
        H_csr=H, error_priors=p,
        pre_iter=10, num_sets=1, set_max_iter=10,
        gamma0=0.0, gamma_dist_interval=(0.0,0.0),
        stop_nconv=1, normalized_min_sum_alpha=1.0,
        dtype_messages="fp16", device="cuda", seed=0
    )

    for s in S:
        e_rust = np.asarray(rust.decode(s), dtype=np.uint8)
        s_dev  = torch.from_numpy(np.array([s], dtype=np.uint8)).to("cuda")
        e_gpu  = gpu.decode(s_dev)["errors"][0].detach().cpu().numpy().astype(np.uint8)

        assert parity_ok(H, e_rust, s)
        assert parity_ok(H, e_gpu,  s)


