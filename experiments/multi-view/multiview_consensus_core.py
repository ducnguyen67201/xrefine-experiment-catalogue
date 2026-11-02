"""Fixed, array-only other-view consensus for the 2026-10-20 protocol.

No files, images, poses, scene labels, models or evaluation scores are read.
The producer must establish image pin/raster/intrinsics consistency before
calling this module. Exact RAW coordinate/descriptor identities are required;
ambiguous RAW coordinates always fall back. This is known multi-view adaptation,
not the separately trained XRefine n-view architecture.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Iterable, Mapping

import numpy as np


SCHEMA = "xrefine-multiview-context-v1"
METHODS = ("huber", "mean", "shrink", "shuffle")
_FIELDS = {"schema", "image_name", "node_keys", "raw", "ambiguity_keys",
           "offsets", "partner_names", "partner_ids", "means",
           "observation_counts", "statistics_json"}
_STATISTICS = {"observations", "unique_raw_coordinates",
               "ambiguous_raw_coordinates", "ambiguous_observations",
               "unambiguous_nodes", "distinct_partner_votes"}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _name(value):
    _require(type(value) is str and bool(value), "Nonempty image name required")
    try:
        value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError("Image names must be ASCII") from exc
    return value


def canonical_digest(values):
    """Protocol wire: compact ASCII JSON, no trailing LF."""
    wire = json.dumps(values, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode("ascii")
    return hashlib.sha256(wire).hexdigest()


def _f32(value, width, label):
    _require(type(value) is np.ndarray and value.ndim == 2 and
             value.shape[1:] == (width,) and value.dtype.kind == "f" and
             value.dtype.itemsize == 4 and np.isfinite(value).all(),
             label + " must be finite float32 N x " + str(width))
    return np.ascontiguousarray(value, dtype="<f4")


def _arrays(raw, stock, descriptors):
    raw = _f32(raw, 2, "RAW")
    stock = _f32(stock, 2, "Stock")
    descriptors = _f32(descriptors, 64, "Descriptors")
    _require(len(raw) == len(stock) == len(descriptors), "Unaligned inputs")
    return raw, stock, descriptors


def _frozen(value, dtype=None):
    result = np.array(value, dtype=dtype, copy=True, order="C")
    result.flags.writeable = False
    return result


def _scalar_text(value, label):
    _require(type(value) is np.ndarray and value.shape == () and
             value.dtype.kind == "U", label + " must be scalar Unicode")
    return str(value.item())


@dataclass(frozen=True, slots=True)
class ImageContext:
    """One image's exact nodes and per-partner votes, NPZ-safe without pickle.

    node_keys are RAW xy (8 bytes) followed by descriptor (256 bytes), both
    little-endian FP32. RAW ambiguities are separate 8-byte keys. Partner names
    occur once; offsets and partner_ids encode each node's distinct votes.
    Arrays and the internal lookup are owned copies, never input mutations.
    """
    image_name: str
    node_keys: np.ndarray
    raw: np.ndarray
    ambiguity_keys: np.ndarray
    offsets: np.ndarray
    partner_names: np.ndarray
    partner_ids: np.ndarray
    means: np.ndarray
    observation_counts: np.ndarray
    statistics: Mapping
    _node_rows: dict = field(init=False, repr=False, compare=False)
    _ambiguous: frozenset = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        _name(self.image_name)
        specs = {"node_keys": ("V264", 1), "raw": ("<f4", 2),
                 "ambiguity_keys": ("V8", 1), "offsets": ("<i8", 1),
                 "partner_ids": ("<i8", 1), "means": ("<f8", 2),
                 "observation_counts": ("<i8", 1)}
        for name, (dtype, ndim) in specs.items():
            value = getattr(self, name)
            _require(type(value) is np.ndarray and value.dtype == np.dtype(dtype)
                     and value.ndim == ndim, "Context field dtype/shape: " + name)
            object.__setattr__(self, name, _frozen(value))
        _require(type(self.partner_names) is np.ndarray and
                 self.partner_names.ndim == 1 and self.partner_names.dtype.kind == "U",
                 "Context partner_names must be Unicode vector")
        object.__setattr__(self, "partner_names", _frozen(self.partner_names))
        nodes, votes = len(self.node_keys), len(self.partner_ids)
        _require(self.raw.shape == (nodes, 2) and np.isfinite(self.raw).all() and
                 self.means.shape == (votes, 2) and np.isfinite(self.means).all(),
                 "Context coordinate dimensions/finite values")
        _require(self.offsets.shape == (nodes + 1,) and self.offsets[0] == 0 and
                 self.offsets[-1] == votes and np.all(np.diff(self.offsets) > 0),
                 "Each context node must have one or more votes")
        _require(self.observation_counts.shape == (votes,) and
                 np.all(self.observation_counts > 0), "Context vote counts")
        names = [str(value) for value in self.partner_names]
        _require(names == sorted(set(names)) and all(_name(value) != self.image_name
                 for value in names), "Distinct ordered auxiliary image names")
        _require(not votes or (self.partner_ids.min() >= 0 and
                 self.partner_ids.max() < len(names)), "Context partner ID range")
        keys = [value.tobytes() for value in self.node_keys]
        ambiguity = [value.tobytes() for value in self.ambiguity_keys]
        _require(keys == sorted(set(keys)) and ambiguity == sorted(set(ambiguity)),
                 "Context keys must be unique and byte sorted")
        ambiguity_set = frozenset(ambiguity)
        for row, key in enumerate(keys):
            _require(key[:8] == self.raw[row].tobytes() and key[:8] not in ambiguity_set,
                     "Node RAW witness or ambiguity mismatch")
            start, stop = map(int, self.offsets[row:row + 2])
            _require(np.all(np.diff(self.partner_ids[start:stop]) > 0),
                     "Each node needs distinct ordered partner IDs")
        stats = dict(self.statistics)
        _require(set(stats) == _STATISTICS and all(type(value) is int and value >= 0
                 for value in stats.values()), "Context statistics schema")
        _require(stats["unambiguous_nodes"] == nodes and
                 stats["distinct_partner_votes"] == votes and
                 stats["ambiguous_raw_coordinates"] == len(ambiguity) and
                 stats["unique_raw_coordinates"] == nodes + len(ambiguity) and
                 stats["ambiguous_observations"] <= stats["observations"],
                 "Context statistics alignment")
        object.__setattr__(self, "statistics", stats)
        object.__setattr__(self, "_node_rows", dict(zip(keys, range(nodes))))
        object.__setattr__(self, "_ambiguous", ambiguity_set)

    def to_arrays(self):
        """Owned readonly arrays suitable for np.savez, with no object dtype."""
        result = {name: getattr(self, name) for name in
                  ("node_keys", "raw", "ambiguity_keys", "offsets", "partner_names",
                   "partner_ids", "means", "observation_counts")}
        result.update(schema=np.asarray(SCHEMA), image_name=np.asarray(self.image_name),
                      statistics_json=np.asarray(json.dumps(dict(self.statistics),
                          sort_keys=True, separators=(",", ":"), allow_nan=False)))
        return result

    @classmethod
    def from_arrays(cls, arrays):
        _require(set(arrays) == _FIELDS, "Context NPZ fields differ")
        _require(_scalar_text(arrays["schema"], "schema") == SCHEMA,
                 "Context schema differs")
        stats = json.loads(_scalar_text(arrays["statistics_json"], "statistics_json"))
        _require(type(stats) is dict, "Context statistics must be an object")
        return cls(_scalar_text(arrays["image_name"], "image_name"),
                   *(arrays[name] for name in ("node_keys", "raw", "ambiguity_keys",
                     "offsets", "partner_names", "partner_ids", "means",
                     "observation_counts")), stats)


def build_image_context(image_name: str, observations: Iterable[Mapping]) -> ImageContext:
    """Aggregate one image partition in fixed partner/pair/side/row order.

    Each mapping has exactly partner,pair_index,side,raw,stock,descriptors.
    Same-partner repeats/orientations are averaged before partner weighting.
    If any descriptor conflicts at one RAW coordinate, all its observations
    are ambiguous, including earlier ones. No approximate matching occurs.
    """
    _name(image_name)
    items = list(observations)
    seen = set()
    for observation in items:
        _require(isinstance(observation, Mapping) and set(observation) ==
                 {"partner", "pair_index", "side", "raw", "stock", "descriptors"},
                 "Observation fields differ")
        _require(_name(observation["partner"]) != image_name, "Self-image observation")
        index, side = observation["pair_index"], observation["side"]
        _require(type(index) is int and index >= 0 and type(side) is int and side in (0, 1),
                 "Observation pair index/side")
        _require((index, side) not in seen, "Duplicate observation partition")
        seen.add((index, side))
    items.sort(key=lambda item: (item["partner"], item["pair_index"], item["side"]))
    nodes, observations_total = {}, 0
    for item in items:
        raw, stock, descriptors = _arrays(item["raw"], item["stock"], item["descriptors"])
        observations_total += len(raw)
        for point, refined, descriptor in zip(raw, stock, descriptors):
            raw_key, descriptor_key = point.tobytes(), descriptor.tobytes()
            state = nodes.get(raw_key)
            if state is None:
                state = [descriptor_key, {}, 0]
                nodes[raw_key] = state
            state[2] += 1
            if state[0] is None:
                continue
            if state[0] != descriptor_key:
                state[0], state[1] = None, {}
                continue
            vote = state[1].get(item["partner"])
            if vote is None:
                vote = [np.zeros(2, dtype=np.float64), 0]
                state[1][item["partner"]] = vote
            vote[0] += refined.astype(np.float64)
            vote[1] += 1
    ambiguity = sorted(key for key, state in nodes.items() if state[0] is None)
    ordered = sorted((raw_key + state[0], state) for raw_key, state in nodes.items()
                     if state[0] is not None)
    names = sorted({partner for _, state in ordered for partner in state[1]})
    name_ids = {name: index for index, name in enumerate(names)}
    keys, coordinates, offsets, partners, means, counts = [], [], [0], [], [], []
    for key, state in ordered:
        keys.append(key)
        coordinates.append(np.frombuffer(key[:8], dtype="<f4"))
        for partner in sorted(state[1]):
            total, count = state[1][partner]
            partners.append(name_ids[partner])
            means.append(total / count)
            counts.append(count)
        offsets.append(len(partners))
    stats = {"observations": observations_total, "unique_raw_coordinates": len(nodes),
             "ambiguous_raw_coordinates": len(ambiguity),
             "ambiguous_observations": sum(state[2] for state in nodes.values()
                                            if state[0] is None),
             "unambiguous_nodes": len(keys), "distinct_partner_votes": len(partners)}
    return ImageContext(image_name, np.asarray(keys, dtype="V264"),
                        np.asarray(coordinates, dtype="<f4").reshape(-1, 2),
                        np.asarray(ambiguity, dtype="V8"), np.asarray(offsets, dtype="<i8"),
                        np.asarray(names, dtype=str), np.asarray(partners, dtype="<i8"),
                        np.asarray(means, dtype="<f8").reshape(-1, 2),
                        np.asarray(counts, dtype="<i8"), stats)


def _selected(context, raw, descriptors, excluded_partner):
    """Gather at most eight distinct other partners; no target-pair values."""
    votes = np.zeros((len(raw), 8, 2), dtype=np.float64)
    counts = np.zeros(len(raw), dtype=np.int64)
    ambiguous = np.zeros(len(raw), dtype=bool)
    for row, (point, descriptor) in enumerate(zip(raw, descriptors)):
        raw_key = point.tobytes()
        if raw_key in context._ambiguous:
            ambiguous[row] = True
            continue
        node = context._node_rows.get(raw_key + descriptor.tobytes())
        if node is None:
            continue
        start, stop = map(int, context.offsets[node:node + 2])
        candidates = []
        for index in range(start, stop):
            partner = str(context.partner_names[context.partner_ids[index]])
            if partner != excluded_partner:
                key = canonical_digest([42, context.image_name, raw_key.hex(), partner])
                candidates.append((key, partner, index))
        candidates.sort()
        chosen = candidates[:8]
        counts[row] = len(chosen)
        for slot, (_, _, index) in enumerate(chosen):
            votes[row, slot] = context.means[index]
    return votes, counts, ambiguous


def radial_huber_32(anchors, votes, counts):
    """Locked 32-step FP64 optimizer; no convergence/threshold search."""
    _require(type(anchors) is np.ndarray and anchors.dtype == np.float64 and
             anchors.ndim == 2 and anchors.shape[1:] == (2,) and
             type(votes) is np.ndarray and votes.dtype == np.float64 and
             votes.shape == (len(anchors), 8, 2) and
             type(counts) is np.ndarray and counts.dtype == np.int64 and
             counts.shape == (len(anchors),) and np.all((counts >= 1) & (counts <= 8))
             and np.isfinite(anchors).all() and np.isfinite(votes).all(),
             "Huber inputs must be finite aligned FP64 votes/counts")
    current = anchors.copy()
    active = np.arange(8)[None, :] < counts[:, None]
    for _ in range(32):
        difference = current[:, None, :] - votes
        distance = np.sqrt(np.sum(difference * difference, axis=2))
        scale = np.minimum(1.0, 0.5 / np.maximum(distance, 1e-12))
        gradient = (current - anchors) + np.sum(
            difference * (scale * active)[:, :, None], axis=1) / counts[:, None]
        current -= 0.5 * gradient
    _require(np.isfinite(current).all(), "Non-finite Huber result")
    return current


def transform_pair(pair_index, image_names, raw, stock, descriptors, contexts):
    """Return predictions, shared eligibility/support counts and JSON stats.

    Both endpoints must have another partner. Unsupported rows are exact stock
    copies in every arm. Shuffle rotates *joint* final-FP32 Huber deltas +1 in
    canonical hash order; target[i] receives source[i-1]. Singleton is reported.
    """
    _require(type(pair_index) is int and pair_index >= 0, "Invalid pair index")
    _require(isinstance(image_names, (tuple, list)) and len(image_names) == 2 and
             _name(image_names[0]) != _name(image_names[1]), "Distinct pair image names")
    for values in (raw, stock, descriptors, contexts):
        _require(isinstance(values, (tuple, list)) and len(values) == 2,
                 "Two endpoint inputs required")
    validated = [_arrays(raw[side], stock[side], descriptors[side]) for side in (0, 1)]
    _require(len(validated[0][0]) == len(validated[1][0]), "Pair count mismatch")
    count = len(validated[0][0])
    selected, support, ambiguity = [], [], []
    dispersions = []
    for side in (0, 1):
        _require(type(contexts[side]) is ImageContext and
                 contexts[side].image_name == image_names[side], "Context image mismatch")
        votes, counts, ambiguous = _selected(contexts[side], validated[side][0],
                                             validated[side][2], image_names[1 - side])
        selected.append(votes)
        support.append(counts)
        ambiguity.append(ambiguous)
        dispersion = np.zeros(count, dtype=np.float64)
        supported = counts > 0
        if np.any(supported):
            active = np.arange(8)[None, :] < counts[supported, None]
            mean = np.sum(votes[supported] * active[:, :, None], axis=1) / counts[supported, None]
            squared = np.sum((votes[supported] - mean[:, None, :]) ** 2, axis=2)
            dispersion[supported] = np.sqrt(np.sum(squared * active, axis=1) / counts[supported])
        dispersions.append(dispersion)
    support_counts = np.column_stack(support)
    eligible = (support_counts[:, 0] >= 1) & (support_counts[:, 1] >= 1)
    outputs = {method: [validated[side][1].copy() for side in (0, 1)] for method in METHODS}
    for side in (0, 1):
        source_raw, source_stock, _ = validated[side]
        anchors = source_stock[eligible].astype(np.float64)
        votes, counts = selected[side][eligible], support[side][eligible]
        if len(anchors):
            active = np.arange(8)[None, :] < counts[:, None]
            mean = np.sum(votes * active[:, :, None], axis=1) / counts[:, None]
            outputs["huber"][side][eligible] = radial_huber_32(anchors, votes, counts).astype("<f4")
            outputs["mean"][side][eligible] = ((anchors + mean) * 0.5).astype("<f4")
            center = np.floor(source_raw[eligible].astype(np.float64)) + 0.5
            outputs["shrink"][side][eligible] = ((anchors + center) * 0.5).astype("<f4")
    order = sorted((canonical_digest([42, pair_index, int(row)]), int(row))
                   for row in np.flatnonzero(eligible))
    target = np.asarray([row for _, row in order], dtype=np.int64)
    source = np.roll(target, 1)
    for side in (0, 1):
        base = validated[side][1].astype(np.float64)
        delta = outputs["huber"][side].astype(np.float64) - base
        outputs["shuffle"][side][target] = (base[target] + delta[source]).astype("<f4")
    changed_rows = {}
    changed_endpoints = {}
    for method, points in outputs.items():
        changes = [np.any(points[side].view("<u4") != validated[side][1].view("<u4"), axis=1)
                   for side in (0, 1)]
        changed_rows[method] = int(np.count_nonzero(changes[0] | changes[1]))
        changed_endpoints[method] = sum(int(np.count_nonzero(value)) for value in changes)
        _require(all(np.isfinite(value).all() for value in points), "Non-finite prediction")
    stats = {"matches": count, "eligible_matches": int(eligible.sum()),
             "supported_endpoints": int(np.count_nonzero(support_counts)),
             "ambiguous_endpoints": sum(int(value.sum()) for value in ambiguity),
             "shuffled_nonself_rows": len(target) if len(target) > 1 else 0,
             "shuffled_self_rows": len(target) if len(target) == 1 else 0,
             "changed_rows": changed_rows, "changed_endpoints": changed_endpoints}
    return {"predictions": {method: tuple(points) for method, points in outputs.items()},
            "eligible": eligible, "support_counts": support_counts,
            "context_dispersion": np.column_stack(dispersions), "stats": stats}
