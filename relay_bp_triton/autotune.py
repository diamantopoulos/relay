import os
import json
import hashlib
import platform
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Callable
import itertools, random
import triton
import triton.testing as ttesting
import torch
import csv

DEFAULT_CACHE = Path(os.getenv("RELAY_TUNE_CACHE", "~/.cache/relay_bp_tune.json")).expanduser()

# Optional CSV logging (set RELAY_TUNE_LOG_CSV to enable)
RUN_ID = os.getenv("RELAY_TUNE_RUN_ID", datetime.now(timezone.utc).isoformat())
LOG_CSV_PATH = os.getenv("RELAY_TUNE_LOG_CSV", "./autotuner_"+RUN_ID+".csv").strip()

def _csv_log_enabled() -> bool:
    return bool(LOG_CSV_PATH)

def _safe_str(x):
    try:
        return json.dumps(x, sort_keys=True)
    except Exception:
        return str(x)

def _append_csv_row(row: dict):
    if not _csv_log_enabled():
        return
    path = Path(LOG_CSV_PATH).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = (not path.exists()) or (path.exists() and path.stat().st_size == 0)
    fieldnames = [
        "timestamp", "run_id",
        "kernel", "backend", "device_name", "device_index", "sm",
        "torch", "triton",
        "rep", "trial_idx", "trial_total",
        "grid", "BLOCK_SIZE", "ROWS_PER_CHK", "ROWS_PER_VAR", "BTILE",
        "num_warps", "num_stages",
        "time_s", "best_so_far",
        "problem_key", "meta_base",
    ]
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fieldnames})


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
            uuid = getattr(props, "uuid", None)
            if uuid is not None:
                device_info["uuid"] = str(uuid)
            backend = "cuda" if torch.version.cuda is not None else "rocm"
        except Exception:
            backend = "cuda" if torch.version.cuda is not None else ("rocm" if getattr(torch.version, "hip", None) else "gpu")
            dev = torch.cuda.current_device()
            device_info = {
                "device_index": int(dev),
                "device_name": torch.cuda.get_device_name(dev),
            }
    else:
        device_info = {"device_name": "cpu"}

    return {
        "backend": backend,
        **device_info,
        "driver_cuda": torch.version.cuda,
        "driver_hip": getattr(torch.version, "hip", None),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "python": platform.python_version(),
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
    entry = table.get(key)
    if entry is None:
        return None
    # Backward-compat: old format stored the tuned meta directly
    if isinstance(entry, dict) and 'selected' in entry:
        return entry['selected']
    return entry


def set_saved(kernel_name: str, problem_key: Dict[str, Any], meta: Dict[str, Any]):
    table = load_cache()
    key = _make_cache_key(kernel_name, problem_key)
    # Rich record with context for human readability
    record = {
        "kernel": str(kernel_name),
        "problem": problem_key,
        "selected": meta,
        "fingerprint": _fingerprint(),
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "format": 2,
    }
    table[key] = record
    save_cache(table)


def bench_and_select(
    kernel, grid, args: Tuple, meta_base: Dict[str, Any],
    configs: List[Dict[str, Any]],
    number: int = 20,
    grid_fn: Optional[Callable[[Dict[str, Any]], Tuple[int, ...]]] = None,
) -> Dict[str, Any]:
    verbose = os.getenv("RELAY_TUNE_VERBOSE", "0") == "1"
    best_t = float("inf")
    best_cfg: Optional[Dict[str, Any]] = None
    total = len(configs)
    kname = getattr(kernel, "__name__", "kernel")
    fp = _fingerprint()
    if verbose:
        print(f"[relay-bp-triton] Tuning {kname} with {total} configs, reps={number}")
    for idx, cfg in enumerate(configs, start=1):
        conf_kwargs = {k: v for k, v in cfg.items() if k.isupper()}
        num_warps = cfg.get("num_warps")
        num_stages = cfg.get("num_stages")
        grid_local = grid_fn(cfg) if grid_fn is not None else grid
        kernel[grid_local](*args, **meta_base, **conf_kwargs, num_warps=num_warps, num_stages=num_stages)
        torch.cuda.synchronize()
        for _ in range(2):
            kernel[grid_local](*args, **meta_base, **conf_kwargs, num_warps=num_warps, num_stages=num_stages)
        torch.cuda.synchronize()

        def run():
            kernel[grid_local](*args, **meta_base, **conf_kwargs, num_warps=num_warps, num_stages=num_stages)

        t = ttesting.do_bench(run, rep=number)
        if verbose:
            cfg_str = ", ".join([f"{k}={v}" for k, v in conf_kwargs.items()])
            print(f"  [{idx}/{total}] {cfg_str}, warps={num_warps}, stages={num_stages} -> {t:.3e} s")
        _append_csv_row({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_id": RUN_ID,
            "kernel": kname,
            "backend": fp.get("backend"),
            "device_name": fp.get("device_name"),
            "device_index": fp.get("device_index"),
            "sm": fp.get("sm"),
            "torch": fp.get("torch"),
            "triton": fp.get("triton"),
            "rep": number,
            "trial_idx": idx,
            "trial_total": total,
            "grid": _safe_str(grid_local),
            "BLOCK_SIZE": conf_kwargs.get("BLOCK_SIZE"),
            "ROWS_PER_CHK": conf_kwargs.get("ROWS_PER_CHK"),
            "ROWS_PER_VAR": conf_kwargs.get("ROWS_PER_VAR"),
            "BTILE": conf_kwargs.get("BTILE"),
            "num_warps": num_warps,
            "num_stages": num_stages,
            "time_s": t,
            "best_so_far": (t < best_t),
            "problem_key": _safe_str({}),
            "meta_base": _safe_str(meta_base),
        })
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


# ---- Dynamic config builders for bench path ----
def _parse_list(env: str, default: list[int]) -> list[int]:
    v = os.getenv(env)
    if v is None:
        return default
    try:
        return [int(x) for x in v.replace(" ", "").split(",") if x]
    except Exception:
        return default


def _cap(n: int | None = None, env: str = "RELAY_TUNE_MAX_CFG", default: int = 64) -> int:
    try:
        return int(os.getenv(env, default if n is None else n))
    except Exception:
        return default if n is None else n


def _sample(cfgs: list[triton.Config], env: str = "RELAY_TUNE_SAMPLE", default: int | None = None, seed: int = 0) -> list[triton.Config]:
    k = os.getenv(env)
    if k is None and default is None:
        return cfgs
    try:
        kint = int(k if k is not None else default)
    except Exception:
        return cfgs
    if kint and kint < len(cfgs):
        rng = random.Random(int(os.getenv("RELAY_TUNE_SEED", seed)))
        return rng.sample(cfgs, kint)
    return cfgs


def _make_configs(space: dict[str, list[int]], cap: int | None = None) -> list[triton.Config]:
    keys = list(space.keys())
    vals = [space[k] for k in keys]
    out: list[triton.Config] = []
    for combo in itertools.product(*vals):
        kw = dict(zip(keys, combo))
        bs = kw.get("BLOCK_SIZE", 0)
        warps = kw.get("num_warps", 0)
        out.append(triton.Config(kw, num_warps=kw.get("num_warps", 2), num_stages=kw.get("num_stages", 2)))
    out = out[:_cap(cap)]
    out = _sample(out)
    return out


def build_c2v_configs() -> list[triton.Config]:
    bs = _parse_list("RELAY_SWEEP_C2V_BLOCK",  [1, 2, 4, 8, 16, 32, 64, 128, 256])
    r = _parse_list("RELAY_SWEEP_ROWS_PER_CHK", [1, 2, 4, 8])
    wp = _parse_list("RELAY_SWEEP_C2V_WARPS",  [1, 2, 4])
    st = _parse_list("RELAY_SWEEP_C2V_STAGES", [1, 2])
    space = {"BLOCK_SIZE": bs, "ROWS_PER_CHK": r, "num_warps": wp, "num_stages": st}
    return _make_configs(space)


def build_v2c_configs() -> list[triton.Config]:
    bs = _parse_list("RELAY_SWEEP_V2C_BLOCK",  [1, 2, 4, 8, 16, 32, 64, 128, 256])
    r = _parse_list("RELAY_SWEEP_ROWS_PER_VAR", [1, 2, 4, 8])
    wp = _parse_list("RELAY_SWEEP_V2C_WARPS",  [1, 2, 4])
    st = _parse_list("RELAY_SWEEP_V2C_STAGES", [1, 2])
    space = {"BLOCK_SIZE": bs, "ROWS_PER_VAR": r, "num_warps": wp, "num_stages": st}
    return _make_configs(space)


def build_btile_compute_configs() -> list[triton.Config]:
    bs  = _parse_list("RELAY_SWEEP_BT_BLOCK",  [1, 2, 4, 8, 16, 32, 64, 128, 256])
    wp  = _parse_list("RELAY_SWEEP_BT_WARPS",  [1, 2, 4])
    stg = _parse_list("RELAY_SWEEP_BT_STAGES", [1, 2])
    bt  = _parse_list("RELAY_SWEEP_BTILE",     [16, 32, 64, 128])
    space = {"BLOCK_SIZE": bs, "BTILE": bt, "num_warps": wp, "num_stages": stg}
    return _make_configs(space)


def build_btile_transpose_configs() -> list[triton.Config]:
    btile = _parse_list("RELAY_SWEEP_TR_BTILE",  [8, 16, 32, 64, 128, 256])
    wp    = _parse_list("RELAY_SWEEP_TR_WARPS",  [1, 2, 4])
    stg   = _parse_list("RELAY_SWEEP_TR_STAGES", [1, 2])
    space = {"BTILE": btile, "num_warps": wp, "num_stages": stg}
    return _make_configs(space)


def build_parity_configs() -> list[triton.Config]:
    bs  = _parse_list("RELAY_SWEEP_PAR_BLOCK",  [1, 2, 4, 16, 32, 64, 128, 256])
    wp  = _parse_list("RELAY_SWEEP_PAR_WARPS",  [1, 2, 4])
    stg = _parse_list("RELAY_SWEEP_PAR_STAGES", [1, 2])
    space = {"BLOCK_SIZE": bs, "num_warps": wp, "num_stages": stg}
    return _make_configs(space)



