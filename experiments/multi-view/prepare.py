"""ROOT-only prediction-cache census and fixed n-view bank preparation.

No outcome CSV, GT, essential matrix or pose score is read. Source may be read
before admission; numerical imports occur only in the admitted main entry.
"""
from pathlib import Path
import argparse
import hashlib
import io
import json
import os
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
NS = ROOT / "research/multiview-consensus-20261020"
BENCH = ROOT / "benchmarks/xrefine"
INPUT_PLAN = ROOT / "results/xrefine/plugbench-20261020-full/plan.json"
KINDS = ("huber", "mean", "shrink", "shuffle")
SCENES = ("0015", "0022")
MIB = 1048576


def need(value, message):
    if not value:
        raise ValueError(message)


def pin(path):
    path = Path(path)
    s = path.stat()
    need(path.is_file() and not path.is_symlink() and s.st_nlink == 1, "Regular input required")
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(MIB), b""):
            h.update(b)
    after = path.stat()
    need((s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns) ==
         (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns), "Input changed while hashing")
    return dict(path=str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
                bytes=s.st_size, sha256=h.hexdigest())


def create(path, value):
    from plugbench_multiview_transport import account_growth
    body = (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)+"\n").encode("ascii")
    need(len(body) <= 16*MIB, "Metadata file cap")
    account_growth(path, len(body))
    with path.open("xb") as f:
        f.write(body); f.flush(); os.fsync(f.fileno())
    need(path.read_bytes() == body, "Metadata readback")
    return pin(path)


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(NS/"prepared"))
    args = parser.parse_args()
    output = Path(args.output).absolute()
    need(output == NS/"prepared" and not output.exists() and not output.is_symlink(), "One fixed new output directory")
    input_pin = pin(INPUT_PLAN)
    need(input_pin["bytes"] <= 16*MIB, "Input metadata cap")
    plan = json.loads(INPUT_PLAN.read_bytes())
    records = plan["records"]
    need(plan["pairs"] == list(range(1500)) and plan["normalization"] == "corrected" and len(records) == 1500,
         "Complete fixed corrected input panel")
    incidence, image_metadata = {}, {}
    for p, r in enumerate(records):
        need(r["pair_index"] == p and r["scene"] in SCENES and len(r["image_names"]) == 2 and
             r["image_names"][0] != r["image_names"][1], "Canonical directed record")
        for side, image in enumerate(r["image_names"]):
            other = r["image_names"][1-side]
            need(image.split("/")[0] == r["scene"] == other.split("/")[0], "Scene-name join")
            meta = dict(image_pin=r["image_pins"][side], K=r["K"+str(side)], hw=r["hw"+str(side)])
            need(image not in image_metadata or image_metadata[image] == meta, "Same image has inconsistent K/dimensions/digest")
            image_metadata[image] = meta
            incidence.setdefault(image, []).append((other, p, side))
    need(len(incidence) == 806, "Fixed image inventory")
    sys.path.insert(0, str(BENCH))
    import numpy as np
    from multiview_consensus_core import build_image_context, transform_pair
    from plugbench_data import load_pair
    from plugbench_multiview_huber import input_signature
    from plugbench_multiview_transport import PayloadWriter
    source_paths = [Path(__file__).resolve(), BENCH/"multiview_consensus_core.py", BENCH/"plugbench_data.py",
                    BENCH/"plugbench_multiview_huber.py", BENCH/"plugbench_multiview_transport.py"]
    source_pins = [pin(p) for p in source_paths]
    contexts, image_reports, context_bytes = {}, [], 0
    for image in sorted(incidence):
        entries = sorted(incidence[image])
        def observations():
            for partner, p, side in entries:
                values = load_pair(records[p])
                yield dict(partner=partner, pair_index=p, side=side, raw=values["raw"][side],
                           stock=values["stock"][side], descriptors=values["descriptors"][side])
        context = build_image_context(image, observations())
        arrays = context.to_arrays()
        size = sum(a.nbytes for a in arrays.values())
        context_bytes += size
        need(context_bytes <= 2*1024**3, "Finite resident context-array byte cap")
        contexts[image] = context
        image_reports.append(dict(image=image, incident_records=len(entries), nodes=len(arrays["node_keys"]),
            ambiguous_keys=len(arrays["ambiguity_keys"]), partner_votes=len(arrays["means"]), array_bytes=size,
            statistics=dict(context.statistics)))
    output.mkdir()
    reports, banks, displacement, dispersion = [], [], {k:[] for k in KINDS}, []
    coverage = {s:dict(pairs=0, pairs_with_eight_supported=0, supported_rows=0,
                      changed_pairs={k:0 for k in KINDS}, changed_rows={k:0 for k in KINDS}) for s in SCENES}
    support_histogram = {s:[0]*9 for s in ("source", "target")}
    for start in range(0, 1500, 500):
        writer = PayloadWriter(output, "bank", start, start+500, list(range(start, start+500)))
        logical = []
        try:
            for p in range(start, start+500):
                r = records[p]; values = load_pair(r)
                result = transform_pair(p, tuple(r["image_names"]), values["raw"], values["stock"],
                    values["descriptors"], tuple(contexts[n] for n in r["image_names"]))
                eligible, support = result["eligible"], result["support_counts"]
                n = r["matches"]
                need(eligible.dtype == np.bool_ and eligible.shape == (n,) and support.dtype == np.dtype("int64") and
                     support.shape == (n, 2) and np.all((support >= 0) & (support <= 8)) and
                     np.array_equal(eligible, np.all(support > 0, axis=1)), "Support/eligibility contract")
                arrays = dict(eligible=eligible, support_counts=support)
                count = int(eligible.sum())
                scene = coverage[r["scene"]]; scene["pairs"] += 1
                scene["pairs_with_eight_supported"] += count >= 8; scene["supported_rows"] += count
                changed = {}
                for kind in KINDS:
                    points = result["predictions"][kind]
                    row_change = np.zeros(n, dtype=bool)
                    distances = []
                    for side, a in enumerate(points):
                        need(a.dtype == np.dtype("float32") and a.shape == (n, 2) and np.isfinite(a).all(), "Prepared output schema")
                        need(a[~eligible].tobytes() == values["stock"][side][~eligible].tobytes(), "Unsupported rows changed")
                        arrays[kind+str(side)] = a
                        row_change |= np.any(a.view(np.uint32) != values["stock"][side].view(np.uint32), axis=1)
                        distances.append(np.linalg.norm(a.astype(np.float64)-values["stock"][side].astype(np.float64), axis=1))
                    displacement[kind].append(np.concatenate(distances))
                    changed[kind] = int(row_change.sum())
                    scene["changed_rows"][kind] += changed[kind]; scene["changed_pairs"][kind] += changed[kind] > 0
                for side, label in enumerate(("source", "target")):
                    support_histogram[label] = [a+int(b) for a,b in zip(support_histogram[label], np.bincount(support[:,side], minlength=9))]
                spread = result["context_dispersion"]
                need(spread.dtype == np.dtype("float64") and spread.shape == (n,2) and np.isfinite(spread).all(), "Context dispersion schema")
                dispersion.append(spread[support>0])
                buffer = io.BytesIO(); np.savez_compressed(buffer, **arrays)
                payload = buffer.getvalue(); need(0 < len(payload) <= 32*MIB, "Prepared payload cap")
                logical.append(writer.add_bytes(output/"bank"/f"{p:04d}.npz", payload))
                reports.append(dict(pair_index=p, image_names=r["image_names"], scene=r["scene"], matches=n,
                    input_signature=input_signature(values["raw"], values["stock"], values["descriptors"]),
                    shared_pin=r["shared_pin"], stock_pin=r["stock_pin"], supported_rows=count,
                    changed_rows=changed, stats=result["stats"]))
            packed = writer.finish(logical)
        finally:
            writer.close()
        for row, record in zip(reports[start:start+500], packed):
            row["bank"] = record
        need(len(packed) == 500, "Prepared chunk coverage")
        banks.append(packed[0]["packed"]["container"])
    def summary(values):
        a = np.concatenate(values) if values else np.zeros(0, dtype=np.float64)
        return dict(samples=len(a), mean=float(a.mean()) if len(a) else 0., maximum=float(a.max()) if len(a) else 0.,
                    median=float(np.percentile(a,50,method="linear")) if len(a) else 0.,
                    p95=float(np.percentile(a,95,method="linear")) if len(a) else 0.)
    passed = all(r["pairs_with_eight_supported"] >= 100 and r["supported_rows"] >= 5000 for r in coverage.values())
    need(pin(INPUT_PLAN) == input_pin and [pin(p) for p in source_paths] == source_pins, "Source/input drift")
    manifest = dict(schema="multiview-preparation-v1", status="PREPARED_NO_NEW_POSE_SCORES", coverage_pass=passed,
        input_plan=input_pin, sources=source_pins, banks=banks, records=reports, coverage=coverage,
        support_histogram=support_histogram, images=image_reports, resident_context_array_bytes=context_bytes,
        displacement={k:summary(v) for k,v in displacement.items()}, context_dispersion=summary(dispersion),
        fixed=dict(delta=.5, steps=32, learning_rate=.5, max_other_partners=8, seed=42),
        elapsed_seconds=time.monotonic()-started, new_model_calls=0, native_calls=0, gt_or_outcome_reads=0,
        scope="Additional cached partner-image information; n-view/transductive development, no two-view/fresh-inference/generalization claim")
    manifest_pin = create(output/"manifest.json", manifest)
    create(output/"receipt.json", dict(schema="multiview-preparation-receipt-v1", manifest=manifest_pin,
        coverage_pass=passed, input_plan=input_pin, sources=source_pins, banks=banks,
        elapsed_seconds=time.monotonic()-started, status="PREPARED_NO_NEW_POSE_SCORES"))
    print(json.dumps(dict(status=manifest["status"], coverage_pass=passed, coverage=coverage,
                          elapsed_seconds=manifest["elapsed_seconds"])))


if __name__ == "__main__":
    main()
