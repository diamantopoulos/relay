import os
import sys
import time
import numpy as np
import pytest
import torch
from scipy.sparse import csr_matrix

# Environment controls for test configuration
INCLUDE_OBS_GPU = os.getenv("RELAY_INCLUDE_OBS_GPU", "1") != "0"
FIXED_ITERS = os.getenv("RELAY_PERF_FIXED_ITERS", "1") != "0"
GPU_MSG_DTYPE = os.getenv("RELAY_GPU_MSG_DTYPE", "fp32")  # "fp32" or "fp16"
B_LIST = [int(x) for x in os.getenv("RELAY_B_LIST", "1,64,128,256,512,1024").split(",")]

# Pin Rust threads for stability (if relay_bp uses Rayon)
os.environ.setdefault("RAYON_NUM_THREADS", str(os.cpu_count()))

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


def make_rust_decoder_sinter_like(H, p, fixed_iters=True):
    """Create Rust decoder with sinter-like defaults."""
    base = dict(
        gamma0=0.1, pre_iter=80, num_sets=60, set_max_iter=60,
        gamma_dist_interval=(-0.24, 0.66),
        stop_nconv=(10**9 if fixed_iters else 1),
    )
    # Optional alpha compatibility
    for k, v in [("alpha", 0.90), ("alpha_iteration_scaling_factor", 0.0)]:
        try:
            relay_bp.RelayDecoderF32(csr_matrix(([1],[0],[0,1]), shape=(1,1)),
                                     error_priors=[0.01], **{**base, k: v})
            base[k] = v
        except Exception:
            pass
    p64 = np.array(p, dtype=np.float64, order="C").ravel()
    return relay_bp.RelayDecoderF32(H, error_priors=p64, **base)


def make_triton_decoder_sinter_like(H, p, device="cuda", msg_dtype="fp32", fixed_iters=True):
    """Create Triton decoder with sinter-like defaults."""
    return RelayBPDecoder(
        H_csr=H,
        error_priors=p,
        gamma0=0.1,
        pre_iter=80,
        num_sets=60,
        set_max_iter=60,
        gamma_dist_interval=(-0.24, 0.66),
        stop_nconv=(10**9 if fixed_iters else 1),
        normalized_min_sum_alpha=0.90,   # mirrors Rust α if present
        offset_min_sum_beta=None,
        dtype_messages=msg_dtype,
        device=device,
        seed=1234,
        bitpack_output=False,
        mode="throughput",
    )

def make_triton_decoder_realtime(H, p, device="cuda", msg_dtype="fp16"):
    return RelayBPDecoder(
        H_csr=H,
        error_priors=p,
        gamma0=0.1,
        pre_iter=80,
        num_sets=60,
        set_max_iter=60,
        gamma_dist_interval=(-0.24, 0.66),
        stop_nconv=1_000_000_000,  # fixed iters
        normalized_min_sum_alpha=0.90,
        offset_min_sum_beta=None,
        dtype_messages=msg_dtype,
        device=device,
        seed=1234,
        bitpack_output=False,
        mode="realtime",
    )


def _cuda_time(fn, repeats=5):
    """CUDA event timing for stable GPU measurements."""
    times = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) / 1e3)  # Convert to seconds
    times = np.array(times, dtype=np.float64)
    return float(np.median(times)), float(np.percentile(times, 99.0))


def _warmup_triton(decoder, S_dev, n_warm=2):
    for _ in range(n_warm):
        _ = decoder.decode(S_dev)
    torch.cuda.synchronize()


def _build_torch_obs(O_csr):
    """Build O as torch.sparse_csr on GPU with float32 data (for spmm)."""
    indptr = torch.tensor(O_csr.indptr, dtype=torch.int32, device="cuda")
    indices = torch.tensor(O_csr.indices, dtype=torch.int32, device="cuda")
    data = torch.ones(O_csr.nnz, dtype=torch.float32, device="cuda")
    return torch.sparse_csr_tensor(indptr, indices, data, size=O_csr.shape, device="cuda")


def _gpu_decode_and_obs(gpu_decoder, S_dev, O_torch_csr=None):
    """GPU decode with optional observables computation."""
    out = gpu_decoder.decode(S_dev)  # out["errors"]: [B,V] u8
    if O_torch_csr is not None:
        # (O,V) @ (V,B) -> (O,B), then T -> (B,O), mod 2
        E_f32 = out["errors"].to(torch.float32)       # [B,V] float for spmm
        obs_sum = (O_torch_csr @ E_f32.T).T            # [B,O] float32
        obs = (obs_sum.to(torch.int32) & 1)            # reduce mod 2 if needed
    return out


def time_cpu_rust_runner(runner, S, repeats=5):
    """Time CPU Rust runner with proper median/p99 statistics."""
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        runner.decode_observables_batch(S, parallel=True)
        dt = time.perf_counter() - t0
        times.append(dt)
    return float(np.median(times)), float(np.percentile(times, 99.0))


def time_cpu_rust_decode(decoder, S, repeats=5):
    """Time CPU Rust decode only (no observables)."""
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for s in S:
            decoder.decode(s)
        dt = time.perf_counter() - t0
        times.append(dt)
    return float(np.median(times)), float(np.percentile(times, 99.0))


def _time_decode(fn, repeats=5, sync_cuda=False):
    """Legacy timing function - use _cuda_time for GPU measurements."""
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
    """Throughput test with fair comparison and CUDA event timing."""
    # Real circuit + sinter-like defaults
    H, p, O = build_real_circuit_H_p(
        circuit="bicycle_bivariate_144_12_12_memory_Z",
        basis="z",
        distance=12,
        rounds=12,
        error_rate=0.003,
    )
    m, n = H.shape
    
    # Create decoders with consistent parameters
    rust_runner = make_rust_observable_runner_sinter_like(H, p, O)
    rust_dec = make_rust_decoder_sinter_like(H, p, fixed_iters=FIXED_ITERS)
    gpu = make_triton_decoder_sinter_like(H, p, device="cuda",
                                          msg_dtype=GPU_MSG_DTYPE,
                                          fixed_iters=FIXED_ITERS)
    O_torch = _build_torch_obs(O) if INCLUDE_OBS_GPU else None

    print(f"[perf] source=real_circuit | H: {m}x{n} nnz={H.nnz} | O: {O.shape[0]} | "
          f"fixed_iters={FIXED_ITERS} | gpu_dtype={GPU_MSG_DTYPE} | obs_gpu={INCLUDE_OBS_GPU}")

    for B in B_LIST:
        _, S = sample_batch_errors_and_syndromes(H, p, B=B, seed=42)
        S_dev = torch.tensor(S, dtype=torch.uint8, device="cuda")

        _warmup_triton(gpu, S_dev, n_warm=2)

        if INCLUDE_OBS_GPU:
            # CPU: batch+parallel with observables (sinter-like)
            t_rust_med, t_rust_p99 = time_cpu_rust_runner(rust_runner, S, repeats=5)
            # GPU: decode (+ observables) with CUDA events
            t_gpu_med, t_gpu_p99 = _cuda_time(lambda: _gpu_decode_and_obs(gpu, S_dev, O_torch), repeats=5)
        else:
            # CPU: decode-only (no observables) for a fair compare
            t_rust_med, t_rust_p99 = time_cpu_rust_decode(rust_dec, S, repeats=5)
            # GPU: decode-only
            t_gpu_med, t_gpu_p99 = _cuda_time(lambda: gpu.decode(S_dev), repeats=5)

        # Sanity check
        assert t_rust_med > 0 and t_gpu_med > 0, "Non-positive timing indicates a measurement bug."

        print(f"[perf] B={B:4d} | Rust({('obs' if INCLUDE_OBS_GPU else 'dec')}): {t_rust_med*1e3:8.2f} ms ({B/t_rust_med:8.1f}/s) | "
              f"Triton: {t_gpu_med*1e3:8.2f} ms ({B/t_gpu_med:8.1f}/s) | x{t_rust_med/t_gpu_med:0.2f}")


def test_perf_real_circuit_latency():
    """Latency test for micro-batches (B=1,2,4,8,16)."""
    H, p, O = build_real_circuit_H_p(
        circuit="bicycle_bivariate_144_12_12_memory_Z",
        basis="z", distance=12, rounds=12, error_rate=0.003,
    )
    rust = make_rust_decoder_sinter_like(H, p, fixed_iters=FIXED_ITERS)
    gpu = make_triton_decoder_sinter_like(H, p, device="cuda",
                                         msg_dtype=GPU_MSG_DTYPE,
                                         fixed_iters=FIXED_ITERS)

    for B in [1, 2, 4, 8, 16]:
        _, S = sample_batch_errors_and_syndromes(H, p, B=B, seed=7)
        S_dev = torch.tensor(S, dtype=torch.uint8, device="cuda")
        _warmup_triton(gpu, S_dev, n_warm=2)

        # CPU latency: decode serially to reflect per-request latency
        med_cpu, p99_cpu = time_cpu_rust_decode(rust, S, repeats=20)

        # GPU latency (CUDA events)
        med_gpu, p99_gpu = _cuda_time(lambda: gpu.decode(S_dev), repeats=20)

        # Sanity check
        assert med_cpu > 0 and med_gpu > 0, "Non-positive timing indicates a measurement bug."

        print(f"[lat] B={B:2d} | CPU median {med_cpu*1e3:7.2f} ms  p99 {p99_cpu*1e3:7.2f} ms  || "
              f"GPU median {med_gpu*1e3:7.2f} ms  p99 {p99_gpu*1e3:7.2f} ms  | "
              f"x{med_cpu/med_gpu:0.2f}")


def test_perf_realtime_latency():
    H, p, O = build_real_circuit_H_p(
        circuit="bicycle_bivariate_144_12_12_memory_Z",
        basis="z", distance=12, rounds=12, error_rate=0.003,
    )
    rust = make_rust_decoder_sinter_like(H, p, fixed_iters=True)
    gpu  = make_triton_decoder_realtime(H, p, device="cuda", msg_dtype=os.getenv("RELAY_RT_MSG_DTYPE", "fp16"))

    # warm worker
    s_dummy = torch.zeros((1, H.shape[0]), dtype=torch.uint8, device="cuda")
    _ = gpu.decode_rt(s_dummy)

    times_cpu, times_gpu = [], []
    for _ in range(200):
        _, S = sample_batch_errors_and_syndromes(H, p, B=1, seed=np.random.randint(1<<31))
        # CPU
        t0 = time.perf_counter()
        rust.decode(S[0])
        times_cpu.append(time.perf_counter() - t0)
        # GPU
        S_dev = torch.tensor(S, dtype=torch.uint8, device="cuda")
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
        start.record(); _ = gpu.decode_rt(S_dev); end.record()
        torch.cuda.synchronize()
        times_gpu.append(start.elapsed_time(end)/1e3)

    def stats(xs):
        xs = np.array(xs)
        return float(np.median(xs)), float(np.percentile(xs, 99.0))

    med_cpu, p99_cpu = stats(times_cpu)
    med_gpu, p99_gpu = stats(times_gpu)
    print(f"[rt] p50 CPU {med_cpu*1e3:7.2f} ms  p99 {p99_cpu*1e3:7.2f} ms  || "
          f"GPU p50 {med_gpu*1e3:7.2f} ms  p99 {p99_gpu*1e3:7.2f} ms  | "
          f"x{med_cpu/med_gpu:0.2f} (p50), x{p99_cpu/p99_gpu:0.2f} (p99)")
