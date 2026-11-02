"""RM042 pre-head cross-match adapter; no file I/O or execution admission.

Caller authenticates source/runtime/checkpoint/inputs, qualifies resources and
ownership, seals ALL300 GT-free feature records before reading FIT targets,
validates original pose bottom rows, freezes protocol, and retains evidence.
Synthetic decoder support is not actual model/checkpoint qualification.

Pooling and FIT population moments use FP64. Standardized pooled features are
cast to FP32; neighbor means accumulate in FP64 and return FP32. Actual decoder
operations remain CPU FP32. Training retains g, C-context and S-context once per
FIT pair; independent context aliases g. Graphs are built once per FIT pair,
outside all epoch/arm loops. Returned metadata counts those prepared tensors
and provenance arrays, excluding raw caches and transient working memory.
"""
from __future__ import annotations

import copy
import hashlib
import json

import numpy as np
import torch
from torch import nn

SEED = 20261027
ARMS = ("context", "independent", "shuffled")
PARAMETERS = 12448
MAX_ROWS = 10000
BATCH = 256
QUERY_CHUNK = 128
CACHE_ARRAYS = (
    "x0", "x1", "p0", "p1", "q0", "q1", "geometry", "row_ids"
)


def _need(ok, message):
    if not ok:
        raise ValueError(message)


def _uid(value):
    _need(type(value) is str and value and "\0" not in value,
          "Nonempty NUL-free pair UID")
    value.encode("utf-8")
    return value


def _array(value, dtype, shape, name):
    _need(type(value) is np.ndarray
          and value.dtype == np.dtype(dtype) and value.dtype.isnative
          and value.shape == shape and value.flags.c_contiguous,
          "Native contiguous array: " + name)
    _need(bool(np.isfinite(value).all()), "Finite array: " + name)
    return value


def _ids(value, n):
    _array(value, np.int64, (n,), "immutable row IDs")
    _need(bool((value >= 0).all()) and len(np.unique(value)) == n,
          "Unique nonnegative immutable row IDs")
    return value


def _tensor(value, shape, name):
    _need(torch.is_tensor(value) and value.device.type == "cpu"
          and value.dtype == torch.float32 and tuple(value.shape) == shape,
          "CPU FP32 tensor: " + name)
    _need(bool(torch.isfinite(value).all()), "Finite tensor: " + name)
    return value


def _decoder(net):
    """Structural fixture support; does not authenticate the checkpoint."""
    _need(isinstance(net, nn.Module)
          and callable(getattr(net, "score_map_head", None))
          and callable(getattr(net, "spatial_argmax", None)),
          "Score-head and spatial-argmax decoder")
    _need(all(not m.training for m in net.modules()), "Frozen eval decoder")
    _need(all(p.device.type == "cpu" and p.dtype == torch.float32
              and not p.requires_grad and p.grad is None
              for p in net.parameters()), "Frozen CPU FP32 decoder parameters")
    _need(all(b.device.type == "cpu" and b.dtype == torch.float32
              for b in net.buffers()), "CPU FP32 decoder buffers")
    head = net.score_map_head
    _need(not head._forward_pre_hooks and not head._forward_hooks,
          "Exclusive non-reentrant score head")
    return head


def _small(model):
    _need(type(model).__name__ == "XRefine"
          and model.patch_radius == 5
          and model.adjust_only_second_keypoint is False,
          "Original two-sided radius-five XRefine")
    net = model.net
    _need(type(net).__name__ == "SimpleJointAttnTuner", "Original Small net")
    expected = dict(
        desc_dim=64, color_normalization_strategy="orig",
        spatial_argmax_type="soft", attn_with_patch=True,
        num_attention_blocks=1, adjust_only_second_keypoint=False,
        image_values_are_normalized=True,
    )
    _need(all(getattr(net, k, None) == v for k, v in expected.items()),
          "Unchanged Small configuration")
    encoder = net.encoder
    _need(type(encoder).__name__ == "Encoder"
          and encoder.variant == "small"
          and encoder.patch_size_reduction == 8, "Original Small encoder")
    _need(len(net.cross_attn_patch_blocks) == 1, "One attention block")
    block = net.cross_attn_patch_blocks[0]
    _need(block.positional_encoding_type == "learnable"
          and block.skip_connection is True
          and block.apply_layer_norm is False, "Original attention settings")
    _need(tuple(block.learned_positional_encoding_query.shape) == (1, 9, 64)
          and tuple(block.learned_positional_encoding_kv.shape) == (1, 9, 64),
          "Original nine-token positional tensors")
    attn = block.multihead_attn
    _need(attn.embed_dim == 64 and attn.num_heads == 4 and attn.batch_first,
          "Original four-head attention")
    _need(type(net.score_map_head).__name__ == "ScoreMapHead"
          and net.score_map_head.in_channels == 64
          and not hasattr(net, "match_score_head"), "Original spatial head")
    spatial = net.spatial_argmax
    _need(type(spatial).__name__ == "SpatialSoftArgmax2d"
          and spatial.temperature == 1.0 and spatial.eps == 1e-6,
          "Original softargmax and epsilon")
    _need(all(not m.training for m in model.modules()), "CPU eval wrapper")
    _decoder(net)
    return net.score_map_head


def _layout_spec(spec, m):
    """Validate captured dense, nonoverlapping head layout as metadata."""
    _need(type(spec) is dict and set(spec) == {
        "shape", "stride", "storage_offset", "contiguous", "channels_last"
    }, "Exact original head-layout keys")
    shape, stride = spec["shape"], spec["stride"]
    _need(type(m) is int and 1 <= m <= BATCH
          and type(shape) is list and len(shape) == 4
          and all(type(v) is int for v in shape)
          and shape == [m, 64, 3, 3], "Original head-input shape")
    span = m * 576
    _need(type(stride) is list and len(stride) == 4
          and all(type(v) is int and 0 < v <= span for v in stride),
          "Positive bounded original strides")
    offset = spec["storage_offset"]
    _need(type(offset) is int and 0 <= offset <= BATCH * 576
          and offset + span <= 2 * BATCH * 576,
          "Bounded original storage offset and replay allocation")
    _need(type(spec["contiguous"]) is bool
          and type(spec["channels_last"]) is bool,
          "Literal original layout flags")
    running = 1
    for axis in sorted(
            (axis for axis in range(4) if shape[axis] > 1),
            key=lambda axis: stride[axis]):
        _need(stride[axis] == running,
              "Original head layout is a dense nonoverlapping permutation")
        running *= shape[axis]
    _need(running == span
          and 1 + sum((size - 1) * step
                      for size, step in zip(shape, stride)) == span,
          "Exact dense original storage span")

    def matches_order(physical_order):
        required = 1
        for axis in physical_order:
            if shape[axis] > 1:
                if stride[axis] != required:
                    return False
                required *= shape[axis]
        return True

    _need(spec["contiguous"] == matches_order((3, 2, 1, 0))
          and spec["channels_last"] == matches_order((1, 3, 2, 0)),
          "Captured flags agree with actual shape and strides")
    return offset + span


def _validate_chunks(chunks, n):
    _need(type(chunks) is list
          and len(chunks) == (n + BATCH - 1) // BATCH,
          "Complete original chunk metadata")
    for index, chunk in enumerate(chunks):
        start = index * BATCH
        stop = min(start + BATCH, n)
        _need(type(chunk) is dict and set(chunk) == {
            "start", "stop", "endpoint0", "endpoint1"
        }, "Exact original chunk keys")
        _need(type(chunk["start"]) is int and chunk["start"] == start
              and type(chunk["stop"]) is int and chunk["stop"] == stop,
              "Original 256-row chunks and exact tail")
        _layout_spec(chunk["endpoint0"], stop - start)
        _layout_spec(chunk["endpoint1"], stop - start)


def _capture_head_layout(value, m):
    _tensor(value, (m, 64, 3, 3), "original pre-score tensor")
    _need(value.layout == torch.strided, "Original strided head input")
    spec = dict(
        shape=list(value.shape), stride=list(value.stride()),
        storage_offset=int(value.storage_offset()),
        contiguous=bool(value.is_contiguous()),
        channels_last=bool(value.is_contiguous(
            memory_format=torch.channels_last)),
    )
    _layout_spec(spec, m)
    return spec


def _restore_head_layout(scaled, spec):
    """Copy into exact original layout without detaching the gradient path."""
    m = len(scaled)
    _tensor(scaled, (m, 64, 3, 3), "scaled pre-score values")
    total = _layout_spec(spec, m)
    # Fresh storage starts without requires_grad. copy_ propagates gradients
    # from scaled; setting the storage leaf's requires_grad would be wrong.
    storage = torch.empty(total, dtype=torch.float32, device="cpu")
    replay = storage.as_strided(
        tuple(spec["shape"]), tuple(spec["stride"]),
        storage_offset=spec["storage_offset"])
    replay.copy_(scaled.contiguous())
    _need(_capture_head_layout(replay, m) == spec,
          "Exact original shape/stride/offset/layout replay")
    return replay


def _chunks_digest(chunks):
    raw = json.dumps(
        chunks, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def _cache(cache):
    _need(type(cache) is dict
          and set(cache) == set(CACHE_ARRAYS) | {"pair_uid", "chunks"},
          "Exact cache keys")
    _uid(cache["pair_uid"])
    p0 = cache["p0"]
    _need(type(p0) is np.ndarray and p0.ndim == 2
          and p0.shape[1:] == (2,) and len(p0) <= MAX_ROWS, "Bounded Nx2 rows")
    n = len(p0)
    for key in ("p0", "p1", "q0", "q1"):
        _array(cache[key], np.float32, (n, 2), key)
    for key in ("x0", "x1"):
        _array(cache[key], np.float32, (n, 64, 3, 3), key)
    _array(cache["geometry"], np.float64, (n, 4), "directed geometry")
    _ids(cache["row_ids"], n)
    _validate_chunks(cache["chunks"], n)
    return n


def collect_pair(model, p0, p1, image0, image1, row_ids, pair_uid):
    """Capture detached normal tensors through unchanged 256-row forward calls.

    The model must be exclusive. Hook order is endpoint1 then endpoint0 using
    zero-based endpoint names. Empty input invokes no model. Captures precede
    the head's in-place ReLU. Outer inference_mode is rejected.
    """
    _need(not torch.is_inference_mode_enabled(),
          "Collect under no_grad, not outer inference_mode")
    head = _small(model)
    _uid(pair_uid)
    _need(type(p0) is np.ndarray and p0.ndim == 2
          and p0.shape[1:] == (2,) and len(p0) <= MAX_ROWS, "Bounded p0")
    n = len(p0)
    _array(p0, np.float32, (n, 2), "p0")
    _array(p1, np.float32, (n, 2), "p1")
    _ids(row_ids, n)
    for image in (image0, image1):
        _need(torch.is_tensor(image) and image.device.type == "cpu"
              and image.dtype == torch.float32 and image.ndim == 3
              and image.shape[0] in (1, 3) and image.is_contiguous()
              and min(image.shape[1:]) > 0
              and image.shape[1] * image.shape[2] <= 16777216
              and bool(torch.isfinite(image).all()), "Finite CPU FP32 CHW image")
    captured, chunks0, chunks1, outputs0, outputs1 = [], [], [], [], []
    captured_layouts, chunk_records = [], []
    current = 0
    before = (p0.tobytes(), p1.tobytes(), row_ids.tobytes())

    def capture(module, inputs):
        _need(len(inputs) == 1, "One score-head input")
        captured_layouts.append(_capture_head_layout(inputs[0], current))
        captured.append(inputs[0].detach().clone().contiguous())
        return None

    handle = head.register_forward_pre_hook(capture)
    try:
        for start in range(0, n, BATCH):
            stop = min(start + BATCH, n)
            current = stop - start
            captured.clear()
            captured_layouts.clear()
            with torch.no_grad():
                output = model(
                    torch.from_numpy(p0[start:stop].copy()),
                    torch.from_numpy(p1[start:stop].copy()), image0, image1)
            _need(type(output) is tuple and len(output) == 2,
                  "Original endpoint tuple")
            _need(len(captured) == len(captured_layouts) == 2,
                  "Exactly endpoint1 then endpoint0")
            chunk_records.append(dict(
                start=start, stop=stop,
                endpoint0=captured_layouts[1], endpoint1=captured_layouts[0]))
            for side, parts in ((0, outputs0), (1, outputs1)):
                value = _tensor(output[side], (current, 2), "original q")
                parts.append(value.detach().contiguous().numpy().copy())
            chunks0.append(captured[1].numpy().copy())
            chunks1.append(captured[0].numpy().copy())
    finally:
        handle.remove()
    _need(before == (p0.tobytes(), p1.tobytes(), row_ids.tobytes()),
          "Input rows and IDs unchanged")
    geometry = np.column_stack((
        p0[:, 0].astype(np.float64) / image0.shape[2],
        p0[:, 1].astype(np.float64) / image0.shape[1],
        p1[:, 0].astype(np.float64) / image1.shape[2],
        p1[:, 1].astype(np.float64) / image1.shape[1],
    ))
    cache = dict(
        pair_uid=pair_uid, chunks=chunk_records, p0=p0.copy(), p1=p1.copy(),
        q0=np.concatenate(outputs0) if outputs0 else np.empty((0, 2), np.float32),
        q1=np.concatenate(outputs1) if outputs1 else np.empty((0, 2), np.float32),
        x0=np.concatenate(chunks0) if chunks0 else np.empty((0, 64, 3, 3), np.float32),
        x1=np.concatenate(chunks1) if chunks1 else np.empty((0, 64, 3, 3), np.float32),
        geometry=np.ascontiguousarray(geometry, dtype=np.float64),
        row_ids=row_ids.copy(),
    )
    _cache(cache)
    return cache


def _pooled(cache):
    return np.concatenate((
        cache["x0"].mean(axis=(2, 3), dtype=np.float64),
        cache["x1"].mean(axis=(2, 3), dtype=np.float64),
    ), axis=1)


def fit_normalizer(caches):
    """Caller supplies FIT-only caches; every row contributes, including zeroT."""
    count = 0
    mean = np.zeros(128, np.float64)
    m2 = np.zeros(128, np.float64)
    seen = set()
    for cache in caches:
        n = _cache(cache)
        uid = cache["pair_uid"]
        _need(uid not in seen, "Unique FIT pair UIDs")
        seen.add(uid)
        if n == 0:
            continue
        g = _pooled(cache)
        local_mean = g.mean(axis=0, dtype=np.float64)
        local_m2 = ((g - local_mean) ** 2).sum(axis=0, dtype=np.float64)
        total = count + n
        delta = local_mean - mean
        m2 += local_m2 + delta ** 2 * (count * n / total)
        mean += delta * (n / total)
        count = total
    _need(count > 0, "Nonempty FIT row population")
    std = np.maximum(np.sqrt(np.maximum(m2 / count, 0.0)), 1e-6)
    _need(bool(np.isfinite(mean).all() and np.isfinite(std).all()),
          "Finite FIT population moments")
    mean.setflags(write=False)
    std.setflags(write=False)
    return {"mean": mean, "std": std}


def _standardized(cache, normalizer):
    _need(type(normalizer) is dict and set(normalizer) == {"mean", "std"},
          "Exact frozen normalizer")
    mean = _array(normalizer["mean"], np.float64, (128,), "FIT mean")
    std = _array(normalizer["std"], np.float64, (128,), "FIT std")
    _need(bool((std >= 1e-6).all()), "Frozen population std floor")
    result = np.ascontiguousarray((_pooled(cache) - mean) / std, dtype=np.float32)
    _need(bool(np.isfinite(result).all()), "Finite standardized pooled evidence")
    return torch.from_numpy(result)


def build_neighbors(geometry, row_ids, query_chunk=QUERY_CHUNK):
    n = len(row_ids)
    _need(n <= MAX_ROWS, "Bounded whole-pair graph")
    _array(geometry, np.float64, (n, 4), "whole-pair directed geometry")
    _ids(row_ids, n)
    _need(type(query_chunk) is int and 1 <= query_chunk <= QUERY_CHUNK,
          "Bounded graph query chunk")
    k = min(8, max(n - 1, 0))
    result = np.empty((n, k), np.int64)
    if k == 0:
        return result
    for start in range(0, n, query_chunk):
        stop = min(start + query_chunk, n)
        with np.errstate(over="ignore", invalid="ignore"):
            difference = geometry[start:stop, None, :] - geometry[None, :, :]
            distance = (difference * difference).sum(axis=2, dtype=np.float64)
        _need(bool(np.isfinite(distance).all()),
              "Finite squared graph distances before self exclusion")
        for local, query in enumerate(range(start, stop)):
            distance[local, query] = np.inf
            result[query] = np.lexsort((row_ids, distance[local]))[:k]
    _neighbors(result, row_ids)
    return result


def _neighbors(neighbors, row_ids):
    n = len(row_ids)
    _ids(row_ids, n)
    k = min(8, max(n - 1, 0))
    _array(neighbors, np.int64, (n, k), "neighbor indices")
    if k:
        _need(bool(((neighbors >= 0) & (neighbors < n)).all()),
              "Neighbor bounds")
        _need(bool((neighbors != np.arange(n)[:, None]).all()),
              "No self neighbors")
        _need(all(len(np.unique(row)) == k for row in neighbors),
              "Unique neighbors and unchanged degree")
    return k


def _offset(n, pair_uid, query_id, seed):
    if n <= 2:
        return 0
    wire = "RM042\0" + str(seed) + "\0" + pair_uid + "\0" + str(int(query_id))
    digest = int.from_bytes(hashlib.sha256(wire.encode("utf-8")).digest(), "big")
    return 1 + digest % (n - 2)


def shuffled_neighbors(neighbors, row_ids, pair_uid, seed=SEED):
    _uid(pair_uid)
    _need(type(seed) is int, "Integer shuffle seed")
    k = _neighbors(neighbors, row_ids)
    n = len(row_ids)
    if n <= 2:
        return neighbors.copy()
    order = np.argsort(row_ids, kind="stable")
    rank = np.empty(n, np.int64)
    rank[order] = np.arange(n, dtype=np.int64)
    result = np.empty_like(neighbors)
    for i in range(n):
        true_rank = rank[neighbors[i]]
        other_rank = true_rank - (true_rank > rank[i])
        shifted = (other_rank + _offset(n, pair_uid, row_ids[i], seed)) % (n - 1)
        result[i] = order[shifted + (shifted >= rank[i])]
    _neighbors(result, row_ids)
    _need(k == 0 or bool((result != neighbors).all()),
          "Nonidentity cyclic mapping of each other-ID set")
    return result


def _message_mean(g, indices):
    if indices.shape[1] == 0:
        return torch.zeros_like(g)
    return g[torch.from_numpy(indices)].mean(dim=1, dtype=torch.float64).float()


def context(g, neighbors, arm, row_ids, pair_uid):
    n = len(row_ids)
    _tensor(g, (n, 128), "standardized pooled evidence")
    _ids(row_ids, n)
    _uid(pair_uid)
    _need(arm in ARMS, "Declared arm")
    if arm == "independent":
        return g
    _neighbors(neighbors, row_ids)
    indices = (shuffled_neighbors(neighbors, row_ids, pair_uid)
               if arm == "shuffled" else neighbors)
    return _message_mean(g, indices)


def _provenance(cache, arm, neighbors, messages):
    n = len(cache["row_ids"])
    return dict(
        pair_uid=cache["pair_uid"], arm=arm, seed=SEED,
        algorithm="RM042-query-specific-other-ID-cyclic-SHA256-big-endian-v1",
        row_ids=cache["row_ids"].copy(), neighbors=neighbors.copy(),
        message_indices=messages.copy(),
        offsets=np.asarray([
            _offset(n, cache["pair_uid"], rid, SEED) if arm == "shuffled" else 0
            for rid in cache["row_ids"]], dtype=np.int64),
        nominal_degree=min(8, max(n - 1, 0)),
        uses_neighbors=arm != "independent",
    )


def graph_provenance(cache, arm):
    n = _cache(cache)
    _need(arm in ARMS, "Declared arm")
    neighbors = (np.empty((n, 0), np.int64) if arm == "independent"
                 else build_neighbors(cache["geometry"], cache["row_ids"]))
    messages = (shuffled_neighbors(neighbors, cache["row_ids"], cache["pair_uid"])
                if arm == "shuffled" else neighbors)
    return _provenance(cache, arm, neighbors, messages)


class Adapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(256, 32), nn.ReLU(),
                                    nn.Linear(32, 128))
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)
        _need(sum(p.numel() for p in self.parameters()) == PARAMETERS,
              "Equal-capacity 12448-parameter adapter")

    def forward(self, value):
        return 0.1 * torch.tanh(self.layers(value))


def _decode_with_context(net, cache, adapter, g, c, provenance):
    """Pure internal kernel; inputs were structurally checked by its caller."""
    n = len(cache["row_ids"])
    head = _decoder(net)
    _need(isinstance(adapter, Adapter)
          and all(p.device.type == "cpu" and p.dtype == torch.float32
                  for p in adapter.parameters()), "CPU FP32 adapter")
    _tensor(g, (n, 128), "standardized pooled evidence")
    _tensor(c, (n, 128), "prepared context")
    gamma = adapter(torch.cat((g, c), dim=1))
    _tensor(gamma, (n, 128), "bounded channel scales")
    _need(bool((gamma.abs() <= 0.1).all()), "Bounded residual scales")
    out0, out1 = [], []
    _validate_chunks(cache["chunks"], n)
    for chunk in cache["chunks"]:
        start, stop = chunk["start"], chunk["stop"]
        m = stop - start
        x0 = torch.from_numpy(cache["x0"][start:stop]).clone()
        x1 = torch.from_numpy(cache["x1"][start:stop]).clone()
        scaled0 = x0 * (1.0 + gamma[start:stop, :64, None, None])
        scaled1 = x1 * (1.0 + gamma[start:stop, 64:, None, None])
        y1 = _restore_head_layout(scaled1, chunk["endpoint1"])
        score1 = _tensor(head(y1), (m, 1, 3, 3), "endpoint1 scores")
        y0 = _restore_head_layout(scaled0, chunk["endpoint0"])
        score0 = _tensor(head(y0), (m, 1, 3, 3), "endpoint0 scores")
        delta1 = (net.spatial_argmax(score1.view(1, m, 3, 3)) - 1.0) * 5.0
        delta0 = (net.spatial_argmax(score0.view(1, m, 3, 3)) - 1.0) * 5.0
        p0 = torch.from_numpy(cache["p0"][start:stop].copy())
        p1 = torch.from_numpy(cache["p1"][start:stop].copy())
        out0.append(torch.floor(p0) + 0.5 + delta0.squeeze(0))
        out1.append(torch.floor(p1) + 0.5 + delta1.squeeze(0))
    return dict(
        q0=torch.cat(out0) if out0 else torch.empty((0, 2), dtype=torch.float32),
        q1=torch.cat(out1) if out1 else torch.empty((0, 2), dtype=torch.float32),
        gamma=gamma, provenance=provenance,
    )


def decode_pair(net, cache, adapter, normalizer, arm):
    """No backbone rerun. Caller controls grad mode; training enables gradients."""
    _cache(cache)
    g = _standardized(cache, normalizer)
    provenance = graph_provenance(cache, arm)
    c = context(g, provenance["neighbors"], arm,
                cache["row_ids"], cache["pair_uid"])
    return _decode_with_context(net, cache, adapter, g, c, provenance)


def _constant64(value, shape, name):
    result = torch.as_tensor(value, dtype=torch.float64, device="cpu").detach().clone()
    _need(tuple(result.shape) == shape and bool(torch.isfinite(result).all()),
          "Finite constant geometry: " + name)
    return result


def _fundamental(K0, K1, R, t):
    K0 = _constant64(K0, (3, 3), "K0")
    K1 = _constant64(K1, (3, 3), "K1")
    R = _constant64(R, (3, 3), "R")
    t = _constant64(t, (3,), "t")
    _need(bool(torch.max(torch.abs(R.T @ R - torch.eye(3, dtype=torch.float64)))
               <= 1e-5)
          and abs(float(torch.linalg.det(R)) - 1.0) <= 1e-5,
          "Proper directed rotation without projection")
    try:
        inverse0, inverse1 = torch.linalg.inv(K0), torch.linalg.inv(K1)
    except RuntimeError as exc:
        raise ValueError("Nonsingular calibrated cameras required") from exc
    norm = torch.linalg.vector_norm(t)
    if float(norm) == 0.0:
        return None
    tx, ty, tz = (t / norm).unbind()
    zero = torch.zeros((), dtype=torch.float64)
    cross = torch.stack((zero, -tz, ty, tz, zero, -tx, -ty, tx, zero)).view(3, 3)
    F = inverse1.T @ cross @ R @ inverse0
    scale = torch.linalg.vector_norm(F)
    _need(bool(torch.isfinite(F).all()) and float(scale) > 0.0,
          "Finite nonzero pixel fundamental matrix")
    return F / scale


def _loss_with_F(q0, q1, F, qbase0, qbase1, gamma):
    n = len(q0)
    _tensor(q0, (n, 2), "q0")
    _tensor(q1, (n, 2), "q1")
    _tensor(gamma, (n, 128), "gamma")
    if n == 0 or F is None:
        return None
    base0 = _constant64(qbase0, (n, 2), "baseline q0")
    base1 = _constant64(qbase1, (n, 2), "baseline q1")
    ones = torch.ones((n, 1), dtype=torch.float64)
    point0 = torch.cat((q0.double(), ones), dim=1)
    point1 = torch.cat((q1.double(), ones), dim=1)
    line1, line0 = point0 @ F.T, point1 @ F
    residual = (point1 * line1).sum(dim=1)
    denominator = line1[:, :2].square().sum(dim=1) + line0[:, :2].square().sum(dim=1)
    sampson_px2 = residual.square() / denominator.clamp_min(1e-12)
    displacement = torch.cat((q0.double() - base0, q1.double() - base1), dim=1)
    loss = (torch.log1p(sampson_px2).mean()
            + 0.01 * displacement.square().mean()
            + 0.001 * gamma.double().square().mean())
    _need(bool(torch.isfinite(loss)), "Finite weak-surrogate loss")
    return loss


def calibrated_pixel_sampson(q0, q1, K0, K1, R, t, qbase0, qbase1, gamma):
    """Weak objective in processed pixels, with exactly 1px² log1p scale.

    F is FP64 and Frobenius-normalized. Denominator floor is 1e-12.
    Empty and zero-translation pairs return None and receive no optimizer step.
    This is not a camera-pose accuracy metric.
    """
    return _loss_with_F(q0, q1, _fundamental(K0, K1, R, t),
                        qbase0, qbase1, gamma)


def _module_digest(module):
    digest = hashlib.sha256()
    for key, value in module.state_dict().items():
        digest.update(key.encode("utf-8") + b"\0")
        digest.update(str(tuple(value.shape)).encode("ascii") + b"\0")
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _cache_digests(caches):
    return tuple((
        cache["pair_uid"],
        tuple(hashlib.sha256(
            b"" if cache[key].size == 0 else memoryview(cache[key]).cast("B")
        ).hexdigest() for key in CACHE_ARRAYS),
        _chunks_digest(cache["chunks"]),
    ) for cache in caches)


def _prepare_training(cache, normalizer):
    """Exactly one graph build and one shuffle binding for this FIT pair."""
    n = len(cache["row_ids"])
    g = _standardized(cache, normalizer)
    neighbors = build_neighbors(cache["geometry"], cache["row_ids"])
    messages = shuffled_neighbors(neighbors, cache["row_ids"], cache["pair_uid"])
    empty = np.empty((n, 0), np.int64)
    contexts = dict(
        context=_message_mean(g, neighbors),
        independent=g,
        shuffled=_message_mean(g, messages),
    )
    provenance = dict(
        context=_provenance(cache, "context", neighbors, neighbors),
        independent=_provenance(cache, "independent", empty, empty),
        shuffled=_provenance(cache, "shuffled", neighbors, messages),
    )
    return dict(g=g, contexts=contexts, provenance=provenance)


def _prepared_bytes(prepared):
    tensor_bytes = 0
    provenance_bytes = 0
    for entry in prepared.values():
        tensors = (entry["g"], entry["contexts"]["context"],
                   entry["contexts"]["shuffled"])
        tensor_bytes += sum(t.numel() * t.element_size() for t in tensors)
        for provenance in entry["provenance"].values():
            provenance_bytes += sum(
                value.nbytes for value in provenance.values()
                if type(value) is np.ndarray)
    return int(tensor_bytes), int(provenance_bytes)


def train_heads(net, fit_caches, targets):
    """Train exact three LAST adapters from admitted in-memory FIT operands.

    This function supplies no source/resource/GT admission. The actual caller
    verifies ALL300 GT-free source/feature/ID seals before loading any target,
    and validates original pose bottom rows before supplying directed R,t.
    DEV/MD/FINAL data must not be passed here. No epoch/model selection occurs.
    """
    _need(torch.is_grad_enabled() and not torch.is_inference_mode_enabled(),
          "Enabled training graph outside inference_mode")
    _decoder(net)
    caches = list(fit_caches)
    _need(len(caches) == 150, "Exact FIT150 population")
    for cache in caches:
        _cache(cache)
    uids = [cache["pair_uid"] for cache in caches]
    _need(len(set(uids)) == 150 and type(targets) is dict
          and set(targets) == set(uids), "Exact unique FIT target join")
    fundamentals = {}
    for uid in uids:
        target = targets[uid]
        _need(type(target) is dict and set(target) == {"K0", "K1", "R", "t"},
              "Exact in-memory FIT target fields")
        fundamentals[uid] = _fundamental(**target)

    cache_before = _cache_digests(caches)
    decoder_before = _module_digest(net)
    normalizer = fit_normalizer(caches)
    prepared = {
        cache["pair_uid"]: _prepare_training(cache, normalizer)
        for cache in caches
    }
    prepared_tensor_bytes, prepared_provenance_bytes = _prepared_bytes(prepared)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED)
        initial = Adapter()
    initial_digest = _module_digest(initial)
    heads = {arm: copy.deepcopy(initial) for arm in ARMS}
    _need(len({id(head) for head in heads.values()}) == 3
          and all(_module_digest(head) == initial_digest for head in heads.values()),
          "Separate equal-capacity heads with identical initialization")
    optimizers = {
        arm: torch.optim.AdamW(
            head.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8,
            weight_decay=1e-4)
        for arm, head in heads.items()
    }
    empty_uids = [c["pair_uid"] for c in caches if len(c["row_ids"]) == 0]
    zero_t_uids = [uid for uid in uids if fundamentals[uid] is None]
    skipped = set(empty_uids) | set(zero_t_uids)
    epoch_records = []
    updates = {arm: 0 for arm in ARMS}
    nonzero_gradient_updates = {arm: 0 for arm in ARMS}

    for epoch in range(20):
        order = np.random.default_rng(SEED + epoch).permutation(150)
        summaries = {arm: {"loss_sum": 0.0, "updates": 0} for arm in ARMS}
        for index in order:
            cache = caches[int(index)]
            uid = cache["pair_uid"]
            if uid in skipped:
                continue
            entry = prepared[uid]
            for arm in ARMS:
                optimizer = optimizers[arm]
                optimizer.zero_grad(set_to_none=True)
                result = _decode_with_context(
                    net, cache, heads[arm], entry["g"],
                    entry["contexts"][arm], entry["provenance"][arm])
                loss = _loss_with_F(
                    result["q0"], result["q1"], fundamentals[uid],
                    cache["q0"], cache["q1"], result["gamma"])
                _need(loss is not None, "Nonempty nonzeroT pair loss")
                loss.backward()
                gradients = [p.grad for p in heads[arm].parameters()]
                _need(all(g is not None and bool(torch.isfinite(g).all())
                          for g in gradients), "Finite adapter gradients")
                if any(bool((g != 0).any()) for g in gradients):
                    nonzero_gradient_updates[arm] += 1
                norm = nn.utils.clip_grad_norm_(heads[arm].parameters(), 1.0)
                _need(bool(torch.isfinite(norm)), "Finite preclip gradient norm")
                optimizer.step()
                summaries[arm]["loss_sum"] += float(loss.detach())
                summaries[arm]["updates"] += 1
                updates[arm] += 1
        epoch_records.append(dict(
            epoch=epoch, pair_order_uids=[uids[int(i)] for i in order],
            arms=summaries, skipped_pairs=len(skipped)))

    _need(_cache_digests(caches) == cache_before, "All cached inputs immutable")
    _need(_module_digest(net) == decoder_before
          and all(p.grad is None and not p.requires_grad for p in net.parameters()),
          "Backbone/decoder state and gradients unchanged")
    for head in heads.values():
        head.eval()
        for parameter in head.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
    metadata = dict(
        status="LAST_EPOCH_STATES_RETURNED",
        seed=SEED, epochs=20, checkpoint_selection="LAST_ONLY",
        parameters_per_head=PARAMETERS, initial_state_sha256=initial_digest,
        identical_initialization=True, separate_optimizers=True,
        fit_pairs=150, fit_rows=sum(len(c["row_ids"]) for c in caches),
        normalizer_scope="ALL_FIT_ROWS_INCLUDING_ZERO_T",
        normalizer_population_std=True, normalizer_std_floor=1e-6,
        graph_builds=len(caches), graph_build_scope="ONCE_PER_FIT_PAIR",
        contexts_prepared_once=True, independent_context_aliases_g=True,
        prepared_tensor_bytes=prepared_tensor_bytes,
        prepared_provenance_array_bytes=prepared_provenance_bytes,
        prepared_memory_scope=(
            "During training: g+C-context+S-context FP32 tensors and every "
            "retained provenance ndarray; excludes raw caches, models, "
            "optimizers, transient distance arrays, decoder activations, "
            "Python/container overhead, and caller-owned objects. "
            "Prepared tensors/provenance are not returned as retained outputs."
        ),
        empty_pair_uids=empty_uids, zero_translation_pair_uids=zero_t_uids,
        skipped_unique_pairs=len(skipped), updates=updates,
        nonzero_gradient_updates=nonzero_gradient_updates,
        optimizer=dict(name="AdamW", lr=0.001, betas=[0.9, 0.999],
                       eps=1e-8, weight_decay=1e-4, clip_norm=1.0),
        objective="WEAK_PIXEL_SAMPSON_NOT_POSE_ACCURACY",
        epochs_actual=epoch_records, cache_immutable=True, decoder_immutable=True,
        decoder_state_sha256=decoder_before, final_heads_frozen=True,
    )
    return dict(
        heads=heads, normalizer=normalizer, metadata=metadata,
        optimizer_states={arm: opt.state_dict() for arm, opt in optimizers.items()},
    )


def real_parity(model, p0, p1, image0, image1, row_ids, pair_uid,
                expected_q=None):
    """Actual caller qualifies model/inputs/HEAVY resources; this grants none.

    Two original whole-pair model passes preserve exact 256-row/tail shapes.
    Each zero adapter replays under no_grad and enable_grad with the original
    captured head layouts. No fitting, backward, optimizer, GT, or file I/O.
    Returned cache and layout metadata remain actual retained memory operands.
    """
    _need(not torch.is_inference_mode_enabled(),
          "Call real_parity outside outer inference_mode")
    _small(model)
    _uid(pair_uid)
    _need(type(p0) is np.ndarray and p0.ndim == 2
          and p0.shape[1:] == (2,) and len(p0) <= MAX_ROWS,
          "Bounded actual p0 population")
    n = len(p0)
    _array(p0, np.float32, (n, 2), "actual p0")
    _array(p1, np.float32, (n, 2), "actual p1")
    _ids(row_ids, n)
    for image in (image0, image1):
        _need(torch.is_tensor(image) and image.device.type == "cpu"
              and image.dtype == torch.float32 and image.ndim == 3
              and image.shape[0] in (1, 3) and image.is_contiguous()
              and min(image.shape[1:]) > 0
              and image.shape[1] * image.shape[2] <= 16777216
              and bool(torch.isfinite(image).all()),
              "Finite actual CPU FP32 CHW image")
    if expected_q is not None:
        _need(type(expected_q) is tuple and len(expected_q) == 2,
              "Saved baseline endpoint tuple")
        for side in (0, 1):
            _array(expected_q[side], np.float32, (n, 2),
                   "saved baseline q" + str(side))

    def array_digest(value):
        payload = b"" if value.size == 0 else memoryview(value).cast("B")
        return hashlib.sha256(payload).hexdigest()

    def snapshot():
        return dict(
            model=_module_digest(model),
            p0=array_digest(p0), p1=array_digest(p1),
            row_ids=array_digest(row_ids),
            image0=array_digest(image0.detach().numpy()),
            image1=array_digest(image1.detach().numpy()),
            training=tuple(module.training for module in model.modules()),
            requires_grad=tuple(p.requires_grad for p in model.parameters()),
            parameter_grad_absent=tuple(p.grad is None
                                       for p in model.parameters()),
        )

    before = snapshot()
    untapped_parts = [[], []]
    chunks = (n + BATCH - 1) // BATCH
    for start in range(0, n, BATCH):
        stop = min(start + BATCH, n)
        with torch.inference_mode():
            output = model(
                torch.from_numpy(p0[start:stop].copy()),
                torch.from_numpy(p1[start:stop].copy()), image0, image1)
        _need(type(output) is tuple and len(output) == 2,
              "Untapped actual endpoint tuple")
        for side in (0, 1):
            value = _tensor(output[side], (stop - start, 2),
                            "untapped actual q")
            untapped_parts[side].append(
                value.detach().contiguous().numpy().copy())
    untapped = tuple(
        np.concatenate(parts) if parts else np.empty((0, 2), np.float32)
        for parts in untapped_parts
    )
    cache = collect_pair(
        model, p0, p1, image0, image1, row_ids, pair_uid)
    cache_before = _cache_digests([cache])
    layout_digest = _chunks_digest(cache["chunks"])
    normalizer = dict(
        mean=np.zeros(128, np.float64),
        std=np.ones(128, np.float64),
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED)
        initial = Adapter()
    heads = {arm: copy.deepcopy(initial) for arm in ARMS}
    decoded = {}
    for arm in ARMS:
        head = heads[arm]
        head.eval()
        _need(all(p.requires_grad for p in head.parameters()),
              "Trainable zero adapter for gradient-mode parity")
        with torch.no_grad():
            result = decode_pair(model.net, cache, head, normalizer, arm)
        _need(bool((result["gamma"] == 0).all()),
              "Exactly zero no-grad residual scales: " + arm)
        _need(not result["gamma"].requires_grad
              and not result["q0"].requires_grad
              and not result["q1"].requires_grad,
              "No-grad replay mode bound: " + arm)
        decoded[(arm, "no_grad")] = tuple(
            _tensor(result["q" + str(side)], (n, 2),
                    "no-grad actual q").detach().contiguous().numpy().copy()
            for side in (0, 1)
        )
        with torch.enable_grad():
            result = decode_pair(model.net, cache, head, normalizer, arm)
            _need(bool((result["gamma"] == 0).all()),
                  "Exactly zero gradient-enabled residual scales: " + arm)
            if n > 0:
                _need(result["gamma"].requires_grad
                      and result["q0"].requires_grad
                      and result["q1"].requires_grad,
                      "Differentiable actual decoder replay: " + arm)
        decoded[(arm, "grad_enabled")] = tuple(
            _tensor(result["q" + str(side)], (n, 2),
                    "gradient-enabled actual q").detach().contiguous().numpy().copy()
            for side in (0, 1)
        )
        _need(all(p.grad is None for p in head.parameters()),
              "Parity helper performed no backward: " + arm)

    _small(model)
    _need(snapshot() == before,
          "Actual images/rows/IDs/model values/modes/gradient state unchanged")
    _need(_cache_digests([cache]) == cache_before
          and _chunks_digest(cache["chunks"]) == layout_digest,
          "Captured arrays and original chunk layouts unchanged")
    _need(cache["pair_uid"] == pair_uid
          and cache["row_ids"].tobytes() == row_ids.tobytes()
          and cache["p0"].tobytes() == p0.tobytes()
          and cache["p1"].tobytes() == p1.tobytes(),
          "Complete captured row identity and endpoint bytes")
    for side in (0, 1):
        reference = untapped[side].tobytes()
        _need(cache["q" + str(side)].tobytes() == reference,
              "Actual inference-mode untapped versus no-grad capture parity")
        if expected_q is not None:
            _need(expected_q[side].tobytes() == reference,
                  "Actual untapped versus saved baseline parity")
        for arm in ARMS:
            for mode in ("no_grad", "grad_enabled"):
                _need(decoded[(arm, mode)][side].tobytes() == reference,
                      "Actual zero-adapter byte parity: " + arm + "/" + mode)
    return dict(
        status="BYTE_PARITY_PASS",
        pair_uid=pair_uid, rows=n, batch_size=BATCH,
        original_chunk_count=chunks,
        same_original_call_shapes=True,
        arbitrary_changed_batch_shape_parity_claim=False,
        saved_q_checked=expected_q is not None,
        zero_adapter_arms=list(ARMS),
        replay_execution_modes=["no_grad", "grad_enabled"],
        zero_adapter_parameters_trainable=True,
        grad_enabled_replay_checked=n > 0,
        nonempty_gradient_graph_checked=n > 0,
        backward_performed=False, optimizer_steps=0,
        original_head_layouts_bound=True,
        head_layout_chunks=cache["chunks"],
        head_layout_chunks_sha256=layout_digest,
        untapped_model_forward_calls=chunks,
        collection_model_forward_calls=chunks,
        total_model_forward_calls=2 * chunks,
        whole_pair_model_passes=2 if n else 0,
        frozen_head_replay_calls=12 * chunks,
        frozen_softargmax_replay_calls=12 * chunks,
        total_score_head_calls_including_original_passes=16 * chunks,
        total_softargmax_calls_including_original_passes=16 * chunks,
        replay_encoder_calls=0,
        actual_input_and_model_immutability=True,
        captured_arrays_and_layouts_immutable=True,
        fitting_performed=False,
        method_effect_or_accuracy_claim=False,
        cache=cache,
    )
