"""Fixed pre-refinement pair-utility model; no pose/model/native calls.

This research OOF adapter routes scene identity to the model trained on the
other scene. Identity is never a numeric feature. The output action retains
all matches and chooses either both raw endpoints or both stock endpoints.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


FEATURE_NAMES = (
    "log1p_matches", "cosine_q10", "cosine_q50", "cosine_q90", "cosine_mean", "cosine_sd",
    "source_grid_occupancy", "target_grid_occupancy", "source_cov_ratio", "target_cov_ratio",
    "affine_residual_q50", "affine_residual_q90",
)
SCENES = ("0015", "0022")
RIDGE = 10.0


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical_names(names):
    require(isinstance(names, (tuple, list)) and len(names) == 2, "Two ordered image names required")
    result = tuple(names)
    require(all(type(name) is str and name for name in result), "Invalid image names")
    for name in result:
        parts = name.split("/")
        require(len(parts) == 3 and parts[0] in SCENES and parts[1] == "images" and
                parts[2] not in (".", "..") and "\\" not in name, "Canonical OOF image identity required")
    require(result[0].split("/")[0] == result[1].split("/")[0], "Cross-scene image pair")
    return result


def scene_of(names):
    return canonical_names(names)[0].split("/")[0]


def _matrix(value, columns, name):
    result = np.asarray(value, dtype=np.float64)
    require(result.ndim == 2 and result.shape[1] == columns and np.isfinite(result).all(), name)
    return result


def descriptor_cosines(descriptors, count):
    require(isinstance(descriptors, (tuple, list)) and len(descriptors) == 2, "Two descriptor arrays required")
    a, b = (np.asarray(value, dtype=np.float64) for value in descriptors)
    require(a.ndim == 2 and a.shape == b.shape and a.shape[0] == count and a.shape[1] > 0 and
            np.isfinite(a).all() and np.isfinite(b).all(), "Descriptor shape/count/finite values")
    a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
    return np.clip(np.sum(a * b, axis=1), -1.0, 1.0)


def _occupancy(points):
    if not len(points):
        return 0.0
    lower = np.min(points, axis=0)
    extent = np.max(points, axis=0) - lower
    extent = np.where(extent > 0, extent, 1.0)
    bins = np.clip(np.floor((points - lower) / extent * 4), 0, 3).astype(np.int64)
    return float(len(np.unique(bins[:, 0] + 4 * bins[:, 1])) / 16)


def _cov_ratio(points):
    if not len(points):
        return 0.0
    centered = points - np.mean(points, axis=0)
    eigenvalues = np.linalg.eigvalsh(centered.T @ centered / len(points))
    return 0.0 if eigenvalues[-1] <= 0 else float(np.clip(eigenvalues[0] / eigenvalues[-1], 0, 1))


def _rays(points, camera):
    camera = np.asarray(camera, dtype=np.float64)
    require(camera.shape == (3, 3) and np.isfinite(camera).all(), "Invalid intrinsics")
    try:
        rays = np.linalg.solve(camera, np.column_stack((points, np.ones(len(points)))).T).T
    except np.linalg.LinAlgError as error:
        raise ValueError("Singular intrinsics") from error
    require(np.isfinite(rays).all() and np.all(np.abs(rays[:, 2]) >= 1e-15), "Invalid homogeneous rays")
    return rays[:, :2] / rays[:, 2:]


def compute_features(raw, descriptors, intrinsics):
    """Twelve fixed raw-only features in FEATURE_NAMES order, FP64.

    Percentiles interpolate linearly. Covariance is population in raw pixels.
    Affine target RMS is sqrt(mean squared Euclidean centered-ray norm).
    There is no stock-coordinate, GT, error, image-name or pair-index input.
    """
    require(isinstance(raw, (tuple, list)) and len(raw) == 2, "Two raw coordinate arrays required")
    a, b = (_matrix(value, 2, "Invalid raw coordinates") for value in raw)
    require(a.shape == b.shape, "Raw correspondence alignment")
    count = len(a)
    cosines = descriptor_cosines(descriptors, count)
    cosine_features = [0.0] * 5 if not count else [
        *np.percentile(cosines, [10, 50, 90], method="linear").tolist(),
        float(np.mean(cosines)), float(np.std(cosines, ddof=0)),
    ]
    require(isinstance(intrinsics, (tuple, list)) and len(intrinsics) == 2, "Two processed-image intrinsics required")
    x, y = (_rays(points, camera) for points, camera in zip((a, b), intrinsics))
    residual_features = [0.0, 0.0]
    if count >= 8:
        design = np.column_stack((x, np.ones(count)))
        coefficients, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
        if rank == 3:
            spread = float(np.sqrt(np.mean(np.sum((y - np.mean(y, axis=0)) ** 2, axis=1))))
            residual = np.linalg.norm(y - design @ coefficients, axis=1) / max(spread, 1e-8)
            residual_features = np.percentile(residual, [50, 90], method="linear").tolist()
    features = np.asarray([np.log1p(count), *cosine_features, _occupancy(a), _occupancy(b),
                           _cov_ratio(a), _cov_ratio(b), *residual_features], dtype=np.float64)
    require(features.shape == (12,) and np.isfinite(features).all(), "Non-finite raw features")
    return features


def mean_confidence_feature(raw, descriptors):
    """Compute only feature index4 for the cheap confidence control."""
    count = len(raw[0])
    cosines = descriptor_cosines(descriptors, count)
    result = np.zeros(12, dtype=np.float64)
    result[4] = 0.0 if not count else float(np.mean(cosines))
    return result


def fit_ridge(features, targets, indices):
    """Fixed summed-loss ridge with a truly unpenalized intercept.

    Every statistic is fitted on the supplied training arrays only. Clip Z,
    then recenter it using its own training mean before fitting coefficients.
    """
    features = _matrix(features, 12, "Invalid training features")
    targets = np.asarray(targets, dtype=np.float64)
    require(len(features) > 0 and targets.shape == (len(features),) and np.isfinite(targets).all() and
            np.all(np.abs(targets) <= 1), "Invalid capped-utility labels")
    require(list(indices) in (list(range(12)), [4]), "Only locked primary/control feature sets")
    selected = features[:, list(indices)]
    mean = np.mean(selected, axis=0)
    scale = np.maximum(np.std(selected, axis=0, ddof=0), 1e-6)
    z = np.clip((selected - mean) / scale, -10, 10)
    clipped_mean = np.mean(z, axis=0)
    c = z - clipped_mean
    intercept = float(np.mean(targets))
    coefficients = np.linalg.solve(c.T @ c + RIDGE * np.eye(len(indices)), c.T @ (targets - intercept))
    return dict(feature_indices=list(indices), training_mean=mean.tolist(), training_scale=scale.tolist(),
                clipped_training_mean=clipped_mean.tolist(), coefficients=coefficients.tolist(), intercept=intercept)


def _model_arrays(model):
    keys = {"feature_indices", "training_mean", "training_scale", "clipped_training_mean", "coefficients", "intercept"}
    require(isinstance(model, dict) and set(model) == keys, "Model schema")
    indices = model["feature_indices"]
    require(indices in (list(range(12)), [4]), "Model feature indices")
    arrays = [np.asarray(model[key], dtype=np.float64) for key in
              ("training_mean", "training_scale", "clipped_training_mean", "coefficients")]
    require(all(a.shape == (len(indices),) and np.isfinite(a).all() for a in arrays) and
            np.all(arrays[1] >= 1e-6), "Model scaler/coefficient schema")
    require(type(model["intercept"]) in (int, float) and np.isfinite(model["intercept"]), "Model intercept")
    return indices, arrays


def predict_ridge(features, model):
    features = np.asarray(features, dtype=np.float64)
    require(features.ndim in (1, 2) and features.shape[-1] == 12 and np.isfinite(features).all(), "Prediction feature schema")
    indices, (mean, scale, clipped_mean, coefficients) = _model_arrays(model)
    c = np.clip((features[..., indices] - mean) / scale, -10, 10) - clipped_mean
    result = model["intercept"] + c @ coefficients
    require(np.isfinite(result).all(), "Non-finite utility prediction")
    return result


def fit_cross_scene(features, records, projections):
    """Fit each held-out-scene model only from its opposite-scene projection."""
    features = _matrix(features, 12, "Invalid cross-scene features")
    require(len(records) == len(features) and [r["pair_index"] for r in records] == list(range(len(records))), "Canonical record order")
    require(set(projections) == set(SCENES), "Two label projections required")
    names = [canonical_names(r["image_names"]) for r in records]
    require(len(set(names)) == len(names), "Duplicate ordered pair identity")
    require(all(scene_of(n) == r["scene"] for n, r in zip(names, records)), "Record scene join")
    models, details = {}, {}
    for held_out in SCENES:
        training_scene = next(scene for scene in SCENES if scene != held_out)
        projection = projections[held_out]
        require(projection["eval_scene"] == held_out and projection["train_scene"] == training_scene, "Fold label authority")
        train = [r["pair_index"] for r in records if r["scene"] == training_scene]
        evaluation = [r["pair_index"] for r in records if r["scene"] == held_out]
        require(train and evaluation, "Both scenes require pairs")
        projected = {}
        for row in projection["records"]:
            p, target = row["pair_index"], row["target"]
            require(type(p) is int and p in train and p not in projected and type(target) in (int, float) and
                    np.isfinite(target) and abs(target) <= 1, "Invalid or held-out projected label")
            projected[p] = float(target)
        require(set(projected) == set(train), "Complete opposite-scene label inventory")
        train_images = {name for p in train for name in names[p]}
        eval_images = {name for p in evaluation for name in names[p]}
        require(not train_images.intersection(eval_images), "Image leakage between folds")
        target = np.asarray([projected[p] for p in train], dtype=np.float64)
        primary = fit_ridge(features[train], target, list(range(12)))
        confidence = fit_ridge(features[train], target, [4])
        models[held_out] = {"utility_gate": primary, "confidence_gate": confidence}
        proofs = {}
        for kind, model in models[held_out].items():
            indices, (mean, scale, clipped_mean, coefficients) = _model_arrays(model)
            c = np.clip((features[train][:, indices] - mean) / scale, -10, 10) - clipped_mean
            residual = predict_ridge(features[train], model) - target
            gradient = c.T @ residual + RIDGE * coefficients
            proofs[kind] = dict(training_pairs=len(train), objective=float(np.sum(residual ** 2) + RIDGE * np.sum(coefficients ** 2)),
                                normal_equation_maxabs=float(np.max(np.abs(gradient))),
                                intercept_gradient_abs=float(abs(np.sum(residual))))
        details[held_out] = dict(train_scene=training_scene, eval_scene=held_out, train_indices=train, eval_indices=evaluation,
                                train_images=sorted(train_images), eval_images=sorted(eval_images), fit_proofs=proofs)
    return models, details


def shuffle_key(names):
    wire = (json.dumps([42, *canonical_names(names)], ensure_ascii=True, separators=(",", ":"), allow_nan=False) + "\n").encode("ascii")
    return hashlib.sha256(wire).hexdigest()


def matched_shuffle(records, primary_stock):
    require([r["pair_index"] for r in records] == list(range(len(records))) and
            all(scene_of(r["image_names"]) == r["scene"] for r in records), "Canonical shuffle records")
    primary_stock = np.asarray(primary_stock)
    require(primary_stock.dtype == np.bool_ and primary_stock.shape == (len(records),), "Primary allocation schema")
    raw = np.zeros(len(records), dtype=bool)
    for scene in SCENES:
        selected = [r["pair_index"] for r in records if r["scene"] == scene]
        count = sum(not bool(primary_stock[p]) for p in selected)
        ordered = sorted(selected, key=lambda p: (shuffle_key(records[p]["image_names"]), p))
        raw[ordered[:count]] = True
    return [dict(image_names=list(canonical_names(r["image_names"])), raw=bool(raw[r["pair_index"]])) for r in records]


def validate_checkpoint(checkpoint):
    require(isinstance(checkpoint, dict) and set(checkpoint) == {"schema", "feature_names", "models"} and
            checkpoint["schema"] == "joint-utility-checkpoint-v1" and
            checkpoint["feature_names"] == list(FEATURE_NAMES) and set(checkpoint["models"]) == set(SCENES), "Inference checkpoint schema")
    for models in checkpoint["models"].values():
        require(set(models) == {"utility_gate", "confidence_gate"}, "Checkpoint action models")
        for kind, model in models.items():
            indices, _ = _model_arrays(model)
            require(indices == (list(range(12)) if kind == "utility_gate" else [4]), "Checkpoint feature/action mismatch")
    return checkpoint


class UtilityAdapter:
    """Trusted OOF adapter: no label-file or benchmark-outcome access."""
    def __init__(self, checkpoint_path, kind, shuffled_path=None):
        require(kind in ("utility_gate", "confidence_gate", "shuffled_gate"), "Unknown utility adapter")
        self.kind = kind
        self.checkpoint = validate_checkpoint(json.loads(Path(checkpoint_path).read_bytes()))
        self.shuffled = None
        if kind == "shuffled_gate":
            value = json.loads(Path(shuffled_path).read_bytes())
            require(set(value) == {"schema", "seed", "entries"} and value["schema"] == "joint-utility-shuffled-v1" and
                    value["seed"] == 42, "Shuffle checkpoint schema")
            self.shuffled = {}
            for row in value["entries"]:
                names = canonical_names(row["image_names"])
                require(set(row) == {"image_names", "raw"} and type(row["raw"]) is bool and names not in self.shuffled, "Shuffle entry schema")
                self.shuffled[names] = row["raw"]

    def choose_stock(self, pair):
        scene = scene_of(pair.image_names)
        if self.kind == "shuffled_gate":
            names = canonical_names(pair.image_names)
            require(names in self.shuffled, "Shuffled control pair outside frozen panel")
            return not self.shuffled[names]
        features = (compute_features(pair.raw, pair.descriptors, pair.intrinsics) if self.kind == "utility_gate"
                    else mean_confidence_feature(pair.raw, pair.descriptors))
        return bool(predict_ridge(features, self.checkpoint["models"][scene][self.kind]) >= 0)

    def refine(self, pair):
        from plugbench_contract import Prediction
        points = pair.stock if self.choose_stock(pair) else pair.raw
        return Prediction(points[0], points[1])
