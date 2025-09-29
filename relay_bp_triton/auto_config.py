import json
import os
import hashlib
import platform
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import triton
import triton.testing as ttesting


DEFAULT_CACHE = Path(os.getenv("RELAY_TUNE_CACHE", "~/.cache/relay_bp_tune.json")).expanduser()


def _fingerprint() -> Dict[str, Any]:
    backend = "cpu"
    device_info: Dict[str, Any] = {}
    if torch.cuda.is_available():
        try:
            dev = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(dev)
            major = getattr(props, "major", 0)
            minor = getattr(props, "minor", 0)
            sm = major * 10 + minor
            device_info = {
                "device_index": int(dev),
                "device_name": getattr(props, "name", "unknown"),
                "capability": (major, minor),
                "sm": sm,
                "total_mem": int(getattr(props, "total_memory", 0)),
                "multi_processor_count": int(getattr(props, "multi_processor_count", 0)),
            }
            # UUID may not exist on all builds
            uuid = getattr(props, "uuid", None)
            if uuid is not None:
                device_info["uuid"] = str(uuid)
            backend = "cuda" if torch.version.cuda is not None else "rocm"
        except Exception:
            # Fallback minimal CUDA info
            backend = "cuda" if torch.version.cuda is not None else ("rocm" if getattr(torch.version, "hip", None) else "gpu")
            dev = torch.cuda.current_device()
            device_info = {
                "device_index": int(dev),
                "device_name": torch.cuda.get_device_name(dev),
            }
    else:
        # CPU-only environment
        device_info = {"device_name": "cpu"}

    return {
        "backend": backend,
        **device_info,
        "driver_cuda": torch.version.cuda,           # may be None on ROCm
        "driver_hip": getattr(torch.version, "hip", None),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "python": platform.python_version(),
        # Optional manual override
        "arch_override": os.getenv("SM", ""),
    }


def _make_cache_key(kernel_name: str, problem_key: Dict[str, Any]) -> str:
    fp = _fingerprint()
    blob = json.dumps({"kernel": kernel_name, "fp": fp, "shape": problem_key}, sort_keys=True)
    return hashlib.blake2b(blob.encode(), digest_size=24).hexdigest()


def load_cache() -> Dict[str, Dict[str, Any]]:
    if DEFAULT_CACHE.exists():
        try:
            return json.loads(DEFAULT_CACHE.read_text())
        except Exception:
            pass
    return {}


def save_cache(table: Dict[str, Dict[str, Any]]):
    DEFAULT_CACHE.parent.mkdir(parents=True, exist_ok=True)
    tmp = DEFAULT_CACHE.with_suffix(".tmp")
    tmp.write_text(json.dumps(table, indent=2, sort_keys=True))
    tmp.replace(DEFAULT_CACHE)


def try_get_saved(kernel_name: str, problem_key: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    table = load_cache()
    key = _make_cache_key(kernel_name, problem_key)
    return table.get(key)


def set_saved(kernel_name: str, problem_key: Dict[str, Any], meta: Dict[str, Any]):
    table = load_cache()
    key = _make_cache_key(kernel_name, problem_key)
    table[key] = meta
    save_cache(table)


def bench_and_select(
    kernel, grid, args: Tuple, meta_base: Dict[str, Any],
    configs: List[Dict[str, Any]],
    number: int = 20,
) -> Dict[str, Any]:
    verbose = os.getenv("RELAY_TUNE_VERBOSE", "0") == "1"
    best_t = float("inf")
    best_cfg: Optional[Dict[str, Any]] = None
    total = len(configs)
    if verbose:
        print(f"[relay-bp-triton] Tuning {getattr(kernel, '__name__', 'kernel')} with {total} configs, reps={number}")
    for idx, cfg in enumerate(configs, start=1):
        # tuned constexprs applied directly to the JIT kernel
        conf_kwargs = {k: v for k, v in cfg.items() if k.isupper()}
        num_warps = cfg.get("num_warps")
        num_stages = cfg.get("num_stages")

        # JIT compile + warmups to avoid skewing timings
        kernel[grid](*args, **meta_base, **conf_kwargs, num_warps=num_warps, num_stages=num_stages)
        torch.cuda.synchronize()
        for _ in range(2):
            kernel[grid](*args, **meta_base, **conf_kwargs, num_warps=num_warps, num_stages=num_stages)
        torch.cuda.synchronize()

        def run():
            kernel[grid](*args, **meta_base, **conf_kwargs, num_warps=num_warps, num_stages=num_stages)

        t = ttesting.do_bench(run, rep=number)
        if verbose:
            cfg_str = ", ".join([f"{k}={v}" for k, v in conf_kwargs.items()])
            print(f"  [{idx}/{total}] {cfg_str}, warps={num_warps}, stages={num_stages} -> {t:.3e} s")
        if t < best_t:
            best_t = t
            best_cfg = {"num_warps": num_warps, "num_stages": num_stages, **conf_kwargs}
            if verbose:
                print(f"    -> new best")
    if best_cfg is None:
        raise RuntimeError("No config benchmarked")
    if verbose:
        best_str = ", ".join([f"{k}={v}" for k, v in best_cfg.items()])
        print(f"[relay-bp-triton] Selected: {best_str}")
    return best_cfg


