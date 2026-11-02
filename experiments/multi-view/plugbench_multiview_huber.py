"""Frozen n-view prepared-bank adapter; contains no GT or pose-score reader."""
from pathlib import Path
import hashlib
import json


ROOT = Path(__file__).resolve().parents[2]
PREPARED = ROOT / "research/multiview-consensus-20261020/prepared"
METADATA = {"name": "multiview_huber", "requires": ["stock", "descriptors"],
    "provenance_files": ["multiview_consensus_core.py", "plugbench_multiview_transport.py",
    "../../research/multiview-consensus-20261020/prepared/manifest.json",
    "../../research/multiview-consensus-20261020/prepared/packed/bank-0000-0500.zip",
    "../../research/multiview-consensus-20261020/prepared/packed/bank-0500-1000.zip",
    "../../research/multiview-consensus-20261020/prepared/packed/bank-1000-1500.zip"]}
_CACHE = {}


def input_signature(raw, stock, descriptors):
    """Bind all six ordered input arrays, including dtype and shape."""
    digest = hashlib.sha256()
    for name, values in (("raw", raw), ("stock", stock), ("descriptors", descriptors)):
        if values is None or len(values) != 2:
            raise ValueError("Prepared bank requires raw, stock and descriptors")
        for side, array in enumerate(values):
            header = json.dumps([name, side, array.dtype.str, list(array.shape)],
                                ensure_ascii=True, separators=(",", ":")).encode("ascii")
            digest.update(len(header).to_bytes(4, "little")); digest.update(header)
            digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


class PreparedAdapter:
    def __init__(self, kind):
        if kind not in ("huber", "mean", "shrink", "shuffle"):
            raise ValueError("Unknown fixed consensus arm")
        path = PREPARED / "manifest.json"
        if not path.is_file() or path.is_symlink() or path.stat().st_size > 16*1048576:
            raise ValueError("Prepared manifest is absent or unsafe")
        value = json.loads(path.read_bytes())
        if value["schema"] != "multiview-preparation-v1" or value["coverage_pass"] is not True:
            raise ValueError("A complete coverage-passing prepared bank is required")
        self.kind = kind
        self.records = {r["pair_index"]: r for r in value["records"]}
        if len(value["records"]) != 1500 or set(self.records) != set(range(1500)):
            raise ValueError("Prepared pair coverage")

    def refine(self, pair):
        import numpy as np
        from plugbench_contract import Prediction
        from plugbench_multiview_transport import load_npz
        row = self.records.get(pair.pair_index)
        if row is None or tuple(row["image_names"]) != tuple(pair.image_names):
            raise ValueError("Input outside the frozen n-view panel")
        if row["input_signature"] != input_signature(pair.raw, pair.stock, pair.descriptors):
            raise ValueError("Current raw/stock/descriptor input differs from preparation")
        key = json.dumps(row["bank"], sort_keys=True, separators=(",", ":"))
        if _CACHE.get("key") != key:
            arrays = load_npz(np, row["bank"])
            expected = {name+str(side) for name in ("huber", "mean", "shrink", "shuffle") for side in (0, 1)}
            if set(arrays) != expected | {"eligible", "support_counts"}:
                raise ValueError("Prepared member schema")
            n = len(pair.raw[0])
            for name in expected:
                a = arrays[name]
                if a.dtype != np.dtype("float32") or a.shape != (n, 2) or not np.isfinite(a).all():
                    raise ValueError("Prepared coordinate schema")
            if arrays["eligible"].dtype != np.bool_ or arrays["eligible"].shape != (n,) or \
                    arrays["support_counts"].dtype != np.dtype("int64") or arrays["support_counts"].shape != (n, 2):
                raise ValueError("Prepared support schema")
            for a in arrays.values():
                a.setflags(write=False)
            _CACHE.clear(); _CACHE.update(key=key, arrays=arrays)
        a = _CACHE["arrays"]
        return Prediction(a[self.kind+"0"], a[self.kind+"1"])


def build(config):
    if config:
        raise ValueError("The frozen consensus experiment has no search parameters")
    return PreparedAdapter("huber")
