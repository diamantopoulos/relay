import os
import sys
import time
import numpy as np
import pytest
import torch
from scipy.sparse import csr_matrix

# --- Path bootstrap: repo root + tests (for testdata helpers) ---
THIS_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
for p in [REPO_ROOT, THIS_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

# --- Imports & feature flags ---
try:
    import relay_bp  # Rust/Python frontend
    HAS_RELAY_BP = True
except Exception:
    HAS_RELAY_BP = False

try:
    from relay_bp_triton import RelayBPDecoder  # your Triton backend
    HAS_TRITON = True
except Exception:
    HAS_TRITON = False

try:
    import stim
    HAS_STIM = True
except Exception:
    HAS_STIM = False

# IMPORTANT: CheckMatrices is NOT at relay_bp.__init__!
try:
    from relay_bp.stim.sinter.check_matrices import CheckMatrices
    HAS_CHECKMATRICES = True
except Exception:
    HAS_CHECKMATRICES = False

# testdata helpers live under tests/
try:
    from testdata import get_test_circuit, filter_detectors_by_basis
    HAS_TESTDATA = True
except Exception:
    HAS_TESTDATA = False

HAS_CUDA = torch.cuda.is_available()

pytestmark = [
    pytest.mark.perf,
    pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available"),
    pytest.mark.skipif(not HAS_RELAY_BP, reason="relay_bp (Rust) not importable"),
    pytest.mark.skipif(not HAS_TRITON, reason="relay_bp_triton not importable"),
    pytest.mark.skipif(not HAS_STIM, reason="stim not importable"),
    pytest.mark.skipif(not HAS_CHECKMATRICES, reason="CheckMatrices not importable from relay_bp.stim.sinter.check_matrices"),
    pytest.mark.skipif(not HAS_TESTDATA, reason="tests/testdata helpers not importable"),
]

# ---------------- helpers ----------------

# --- real-circuit loader (basis=z to match sinter unless we want XZ) --------

def build_real_circuit_H_p(
    circuit="bicycle_bivariate_144_12_12_memory_Z",
    basis="z",
    distance=12,
    rounds=12,
    error_rate=0.003,
):
    """Load real Stim circuit and build CSR check matrix + error priors + observables."""
    circ = get_test_circuit(
        circuit=circuit,
        distance=distance,
        rounds=rounds,
        error_rate=error_rate,
    )
    if basis.lower() == "z":
        circ = filter_detectors_by_basis(circ, "Z")

    dem = circ.detector_error_model()
    cm = CheckMatrices.from_dem(dem)

    # H (checks)
    H = cm.check_matrix
    if not isinstance(H, csr_matrix):
        H = H.tocsr()
    if H.dtype != np.uint8:
        H = H.astype(np.uint8, copy=False)
    H.sort_indices()

    # p (priors)
    p = np.ascontiguousarray(np.array(cm.error_priors, dtype=np.float64)).ravel()

    # O (observables)
    O = cm.observables_matrix
    if not isinstance(O, csr_matrix):
        O = O.tocsr()
    if O.dtype != np.uint8:
        O = O.astype(np.uint8, copy=False)
    O.sort_indices()

    assert H.shape[1] == p.shape[0], f"n={H.shape[1]} != len(p)={p.shape[0]}"
    return H, p, O




def sample_batch_errors_and_syndromes(H: csr_matrix, p: np.ndarray, B: int, seed: int = 0):
    """Generate Bernoulli(p) errors and corresponding syndromes s = H @ e mod 2.

    Works whether (H @ E.T) returns ndarray or sparse; always densifies once.
    """
    rng = np.random.default_rng(seed)
    n = H.shape[1]

    # Errors: shape (B, n) uint8
    E_u8 = (rng.random((B, n)) < p).astype(np.uint8)

    # Syndrome: H (m x n) @ E^T (n x B) -> (m x B), then transpose to (B x m)
    prod = H @ E_u8.T
    prod = np.asarray(prod, dtype=np.int32)   # ensure dense integer
    S = (prod.T & 1).astype(np.uint8)         # mod-2

    return E_u8, S

def make_rust_observable_runner_sinter_like(H, p, O):
    """Rust decoder + ObservableDecoderRunner (batch + parallel)."""
    dec = make_rust_decoder_sinter_like(H, p)
    try:
        # Newer wheels may accept the kw; most expect positional.
        return relay_bp.ObservableDecoderRunner(dec, O, include_decode_result=False)
    except TypeError:
        # Very old wheels: no kwargs for include flag either.
        return relay_bp.ObservableDecoderRunner(dec, O, False)

def make_rust_decoder_sinter_like(H, p):
    """Relay-BP config matching relay_bp_sinter.py defaults."""
    base = dict(
        gamma0=0.1,
        pre_iter=80,
        num_sets=60,
        set_max_iter=60,
        gamma_dist_interval=(-0.24, 0.66),
        stop_nconv=1,
    )
    # If the wheel exposes alpha knobs, set a typical normalized-min-sum α≈0.9
    for k, v in [("alpha", 0.90), ("alpha_iteration_scaling_factor", 0.0)]:
        try:
            relay_bp.RelayDecoderF32(csr_matrix(([1],[0],[0,1]), shape=(1,1)),
                                     error_priors=[0.01], **{**base, k: v})
            base[k] = v
        except Exception:
            pass

    p64 = np.array(p, dtype=np.float64, order="C").ravel()
    return relay_bp.RelayDecoderF32(H, error_priors=p64, **base)


def make_triton_decoder_sinter_like(H, p, device="cuda"):
    """Triton config aligned with the sinter defaults."""
    return RelayBPDecoder(
        H_csr=H,
        error_priors=p,
        gamma0=0.1,
        pre_iter=80,
        num_sets=60,
        set_max_iter=60,
        gamma_dist_interval=(-0.24, 0.66),
        stop_nconv=1,
        normalized_min_sum_alpha=0.90,   # mirrors Rust α if present
        offset_min_sum_beta=None,
        dtype_messages="fp32",
        device=device,
        seed=1234,
        bitpack_output=False,
    )

def _warmup_triton(decoder, S_dev, n_warm=2):
    for _ in range(n_warm):
        _ = decoder.decode(S_dev)
    torch.cuda.synchronize()


def _time_decode(fn, repeats=5, sync_cuda=False):
    best = float("inf")
    for _ in range(repeats):
        if sync_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if sync_cuda:
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        best = min(best, dt)
    return best


# ---------------- test ----------------

def test_perf_real_circuit_speed():
    # Real circuit + sinter-like defaults
    H, p, O = build_real_circuit_H_p(
        circuit="bicycle_bivariate_144_12_12_memory_Z",
        basis="z",
        distance=12,
        rounds=12,
        error_rate=0.003,
    )
    m, n = H.shape
    print(f"[perf] source=real_circuit | H: {m}x{n} nnz={H.nnz} | O: {O.shape[0]} observables")

    rust_runner = make_rust_observable_runner_sinter_like(H, p, O)
    gpu = make_triton_decoder_sinter_like(H, p, device="cuda")  # keep for GPU timing

    for B in [64, 128, 256, 512, 1024]:
        _, S = sample_batch_errors_and_syndromes(H, p, B=B, seed=42)
        S_dev = torch.tensor(S, dtype=torch.uint8, device="cuda")

        _warmup_triton(gpu, S_dev, n_warm=2)

        # Rust batch+parallel via observable runner
        t_rust = _time_decode(
            lambda: rust_runner.decode_observables_batch(S, parallel=True),
            repeats=5, sync_cuda=False
        )

        # Triton (batch over B syndromes)
        t_gpu  = _time_decode(lambda: gpu.decode(S_dev),
                            repeats=5, sync_cuda=True)

        print(f"[perf] B={B:4d} | Rust(obs): {t_rust*1e3:8.2f} ms ({B/t_rust:8.1f}/s) | "
            f"Triton: {t_gpu*1e3:8.2f} ms ({B/t_gpu:8.1f}/s) | x{t_rust/t_gpu:0.2f}")
