"""ROOT-only fixed cross-scene utility training from separate label projections.

No CSV/pose/GT reader, model forward or native solver is imported. Raw cache
loading is explicit and belongs only to the admitted numerical training stage.
Each fold's fit receives only the other scene's projected labels and features.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
NS = ROOT / "research/joint-utility-20261020"
BENCH = ROOT / "benchmarks/xrefine"
MAX_FILE = 16 * 1048576


def require(condition, message):
    if not condition:
        raise ValueError(message)


def pin(path):
    path = Path(path)
    before = path.stat()
    require(path.is_file() and not path.is_symlink() and before.st_nlink == 1, "Regular unaliased input required")
    body = path.read_bytes()
    after = path.stat()
    require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) ==
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns), "File changed during read")
    return dict(path=str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
                bytes=len(body), sha256=hashlib.sha256(body).hexdigest())


def verify(record):
    path = ROOT / record["path"]
    actual = pin(path)
    require(set(record) == {"path", "bytes", "sha256"} and actual["bytes"] == record["bytes"] and
            actual["sha256"] == record["sha256"], "Input pin drift: " + str(path))
    return path


def read(path):
    require(Path(path).stat().st_size <= MAX_FILE, "Metadata cap")
    return json.loads(Path(path).read_bytes(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def create(path, value):
    body = (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")
    require(len(body) <= MAX_FILE, "Training metadata file cap")
    with Path(path).open("xb") as stream:
        stream.write(body)
        stream.flush()
        os.fsync(stream.fileno())
    return pin(path)


def summaries(np, values):
    return dict(samples=len(values), median_ms=float(np.percentile(values, 50, method="linear")),
                p95_ms=float(np.percentile(values, 95, method="linear")), total_ms=float(sum(values)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    analysis, output = Path(args.analysis).resolve(), Path(args.output).resolve()
    require(analysis.is_relative_to(NS) and output.is_relative_to(NS) and not output.exists(), "Use declared new training directory")
    started = time.monotonic()
    projection_paths = {scene: analysis / ("labels-for-" + scene + ".json") for scene in ("0015", "0022")}
    projection_pins = {scene:pin(path) for scene,path in projection_paths.items()}
    projections = {scene:read(path) for scene,path in projection_paths.items()}
    input_plan = projections["0015"]["input_plan"]
    require(projections["0022"]["input_plan"] == input_plan, "Projections must bind the same plan")
    plan = read(verify(input_plan))
    require(plan["pairs"] == list(range(1500)) and len(plan["records"]) == 1500 and
            plan["normalization"] == "corrected", "Fixed full input plan")
    # First scientific imports occur only in the ROOT-admitted training entry.
    sys.path.insert(0, str(BENCH))
    import numpy as np
    from plugbench_data import load_pair
    from joint_utility_core import (FEATURE_NAMES, SCENES, compute_features, fit_cross_scene,
        mean_confidence_feature, predict_ridge, matched_shuffle, validate_checkpoint)

    records = plan["records"]
    features, feature_ms, raw_equal_stock, confidence_inputs, confidence_feature_ms = [], [], [], [], []
    for record in records:
        values = load_pair(record)
        t = time.perf_counter()
        x = compute_features(values["raw"], values["descriptors"], (values["K0"], values["K1"]))
        feature_ms.append((time.perf_counter() - t) * 1000)
        features.append(x)
        t = time.perf_counter()
        confidence_inputs.append(mean_confidence_feature(values["raw"], values["descriptors"]))
        confidence_feature_ms.append((time.perf_counter() - t) * 1000)
        raw_equal_stock.append(all(a.shape == b.shape and a.dtype == b.dtype and a.tobytes(order="C") == b.tobytes(order="C")
                                   for a,b in zip(values["raw"], values["stock"])))
    features = np.asarray(features, dtype=np.float64)
    confidence_inputs = np.asarray(confidence_inputs, dtype=np.float64)
    require(np.array_equal(features[:, 4], confidence_inputs[:, 4]), "Mean-confidence feature parity")
    t = time.perf_counter()
    models, details = fit_cross_scene(features, records, projections)
    fitting_seconds = time.perf_counter() - t
    checkpoint = validate_checkpoint(dict(schema="joint-utility-checkpoint-v1", feature_names=list(FEATURE_NAMES), models=models))
    primary, confidence, decisions = [], [], []
    primary_ms, confidence_ms = [], []
    for record, x, cx, base_ms, conf_ms, same in zip(records, features, confidence_inputs, feature_ms, confidence_feature_ms, raw_equal_stock):
        scene = record["scene"]
        t = time.perf_counter()
        utility = float(predict_ridge(x, models[scene]["utility_gate"]))
        primary_ms.append(base_ms + (time.perf_counter() - t) * 1000)
        t = time.perf_counter()
        conf_utility = float(predict_ridge(cx, models[scene]["confidence_gate"]))
        confidence_ms.append(conf_ms + (time.perf_counter() - t) * 1000)
        primary.append(utility >= 0)
        confidence.append(conf_utility >= 0)
        decisions.append(dict(pair_index=record["pair_index"], scene=scene,
                              utility_gate_raw=utility < 0, confidence_gate_raw=conf_utility < 0,
                              utility_gate_prediction=utility, confidence_gate_prediction=conf_utility,
                              raw_equals_stock=same))
    shuffled = dict(schema="joint-utility-shuffled-v1", seed=42,
                    entries=matched_shuffle(records, np.asarray(primary, dtype=bool)))
    for row, shuffle in zip(decisions, shuffled["entries"]):
        row["shuffled_gate_raw"] = shuffle["raw"]
    allocations = {}
    for method in ("utility_gate", "confidence_gate", "shuffled_gate"):
        allocations[method] = {}
        for scene in ("all", *SCENES):
            selected = [row for row in decisions if scene == "all" or row["scene"] == scene]
            nraw = sum(row[method + "_raw"] for row in selected)
            allocations[method][scene] = dict(pairs=len(selected), raw=nraw, stock=len(selected)-nraw,
                output_identical_to_stock=sum(not row[method+"_raw"] or row["raw_equals_stock"] for row in selected),
                output_identical_to_raw=sum(row[method+"_raw"] or row["raw_equals_stock"] for row in selected))
    for scene in SCENES:
        require(allocations["utility_gate"][scene]["raw"] == allocations["shuffled_gate"][scene]["raw"], "Count-matched shuffle")
    # Recheck immutable sources after all cache reads and fitting, before output.
    verify(input_plan)
    for item in projection_pins.values():
        verify(item)
    output.mkdir()
    outputs = []
    outputs.append(create(output / "raw-features.json", dict(schema="joint-utility-raw-features-v1", feature_names=list(FEATURE_NAMES),
        records=[dict(pair_index=r["pair_index"], scene=r["scene"], features=x.tolist()) for r,x in zip(records, features)],
        scope="Raw coordinates/descriptors/intrinsics only; no stock movement, labels, pose score or identity feature")))
    identities = {}
    for scene, info in details.items():
        identities[scene] = {key:value for key,value in info.items() if key != "fit_proofs"}
        identities[scene]["label_projection"] = projection_pins[scene]
    outputs.append(create(output / "training-identities.json", dict(schema="joint-utility-training-identities-v1", input_plan=input_plan, folds=identities)))
    outputs.append(create(output / "checkpoint.json", checkpoint))
    outputs.append(create(output / "shuffled.json", shuffled))
    outputs.append(create(output / "training.json", dict(schema="joint-utility-training-v1", records=decisions, allocations=allocations,
        fit_proofs={scene:info["fit_proofs"] for scene,info in details.items()}, fitting_seconds=fitting_seconds,
        overhead=dict(feature=summaries(np, feature_ms), utility_gate=summaries(np, primary_ms), confidence_gate=summaries(np, confidence_ms)),
        overhead_scope="One conditional cached-input measurement per pair; excludes cache I/O and neural refinement; no inference or end-to-end speed claim",
        source_scope="Two exposed development scenes; OOF identity routing only; no held-out label contributes to its model or scaler")))
    source_paths = [Path(__file__).resolve(), BENCH/"joint_utility_core.py", BENCH/"plugbench_data.py", BENCH/"plugbench_contract.py"]
    receipt = dict(schema="joint-utility-training-receipt-v1", status="TRAINED_NOT_POSE_SCORED", input_plan=input_plan,
        label_projections=projection_pins, sources=[pin(path) for path in source_paths], outputs=outputs,
        pairs=1500, linear_models_fitted=4, features=12, new_neural_model_calls=0, native_calls=0,
        old_csv_or_pose_file_reads=0, elapsed_seconds=time.monotonic()-started,
        limitations="Known utility-regression adaptation on two exposed OOF scenes; no generalization or speed gain established")
    outputs.append(create(output / "receipt.json", receipt))
    actual = [p for p in output.iterdir() if p.is_file()]
    require(len(actual) == 6 and sum(p.stat().st_size for p in actual) <= 64 * 1048576, "Training namespace cap")
    print(json.dumps(dict(status="TRAINED_NOT_POSE_SCORED", output=str(output), pairs=1500,
                          linear_models=4, allocations=allocations, elapsed_seconds=time.monotonic()-started)))


if __name__ == "__main__":
    main()
