"""RM032 known virtual-line photometric adaptation with joint endpoint states.

The new input is image support between matches, outside every unary footprint.
Discrete min-sum BP and table centering are known primitives, not novelty claims.
No model, training, file IO, reference, pose, runner or learned compatibility.
"""
import hashlib
import itertools
import json

import numpy as np

MAX_ROWS = 64
EPSILON = 1e-6
MODES = ('local', 'independent', 'mean', 'map', 'bp', 'shuffled')


def need(condition, message):
    if not condition:
        raise ValueError(message)


def state_grid():
    values = list(itertools.product((-1., 0., 1.), repeat=4))
    values.sort(key=lambda x: (sum(v*v for v in x), x))
    return np.array(values, dtype=np.float64)


def _table_inputs(unary, edges, factors):
    need(type(unary) is np.ndarray and unary.dtype == np.dtype('float64') and unary.dtype.isnative
         and unary.ndim == 2 and 0 <= len(unary) <= MAX_ROWS and 1 <= unary.shape[1] <= 81
         and bool(np.isfinite(unary).all()), 'Finite native FP64 unary[N,K], N<=64,K<=81')
    n, k = unary.shape
    need(type(edges) is np.ndarray and edges.dtype == np.dtype('int64') and edges.dtype.isnative
         and edges.ndim == 2 and edges.shape[1:] == (2,) and len(edges) <= 4*n
         and edges.tolist() == [list(e) for e in sorted(set(map(tuple, edges.tolist())))]
         and all(0 <= i < j < n for i, j in edges.tolist()), 'Canonical unique undirected edges')
    need(type(factors) is np.ndarray and factors.dtype == np.dtype('float64') and factors.dtype.isnative
         and factors.shape == (len(edges), k, k) and bool(np.isfinite(factors).all()),
         'Finite native FP64 factor tables[E,K,K]')
    return n, k


def decompose(unary, edges, factors):
    """Exact U+alpha*sum(W) = sum(A)+alpha*sum(R)+constant.

    alpha=N/E (zero for E=0). Uniform row/column means become node unaries;
    R has zero row/column means. Clip appearance BEFORE this reparameterization.
    """
    n, _ = _table_inputs(unary, edges, factors)
    a = unary.copy()
    if not len(edges):
        return a, factors.copy(), 0.
    alpha = n/len(edges)
    rows, cols = factors.mean(2), factors.mean(1)
    grand = factors.mean((1, 2))
    residual = factors-rows[:, :, None]-cols[:, None, :]+grand[:, None, None]
    for e, (i, j) in enumerate(edges):
        a[i] += alpha*rows[e]
        a[j] += alpha*cols[e]
    return a, residual, -float(alpha*grand.sum())


def _bp(a, edges, residual, valid, alpha):
    n, k = a.shape
    messages = np.zeros((2*len(edges), k), dtype=np.float64)
    directions = [(int(i), int(j), e, False) for e, (i, j) in enumerate(edges)]
    directions = [d for pair in zip(directions, [(j, i, e, True) for i, j, e, _ in directions]) for d in pair]
    masked = np.where(valid, a, np.inf)
    for _ in range(4):
        incoming = np.zeros_like(a)
        for d, (_, target, _, _) in enumerate(directions):
            incoming[target] += messages[d]
        updated = np.zeros_like(messages)
        for d, (source, target, e, transpose) in enumerate(directions):
            table = residual[e].T if transpose else residual[e]
            cavity = masked[source]+incoming[source]-messages[d ^ 1]
            raw = np.min(cavity[:, None]+alpha*table, axis=0)
            origin = float(raw[valid[target]].min())
            updated[d] = np.where(valid[target], raw-origin, 0.)
        messages = updated
    belief = masked.copy()
    for d, (_, target, _, _) in enumerate(directions):
        belief[target] += messages[d]
    return belief, messages


def _permutation(k, ids, endpoint):
    strata = np.sum(state_grid()**2, axis=1) if k == 81 else np.zeros(k)
    order = np.arange(k, dtype=np.int64)
    for movement in sorted(set(strata.tolist())):
        members = [i for i in range(1, k) if strata[i] == movement]
        def key(i):
            body = json.dumps([42, *ids, int(endpoint), i], separators=(',', ':')).encode('ascii')
            return hashlib.sha256(body).digest(), i
        for old, new in zip(members, sorted(members, key=key)):
            order[old] = new
    return order


def infer(unary, edges, factors, valid=None, *, mode, row_ids=None):
    """Same-data modes return lower-is-better costs and one discrete choice.

    independent runs the same four dense BP sweeps with zero interaction.
    mean/map are one fixed readout, not recursive mean-field iterations.
    shuffled changes only centered interactions, fixing zero/movement strata.
    """
    n, k = _table_inputs(unary, edges, factors)
    need(mode in MODES, 'Declared inference mode')
    if valid is None:
        valid = np.ones((n, k), dtype=np.bool_)
    need(type(valid) is np.ndarray and valid.dtype == np.dtype('bool') and valid.shape == (n, k)
         and bool(valid.any(1).all()), 'Each row has at least one valid state')
    ids = np.arange(n, dtype=np.int64) if row_ids is None else row_ids
    need(type(ids) is np.ndarray and ids.dtype == np.dtype('int64') and ids.dtype.isnative
         and ids.shape == (n,) and bool((ids >= 0).all()) and len(set(ids.tolist())) == n,
         'Unique nonnegative stable native int64 IDs')
    a, residual, constant = decompose(unary, edges, factors)
    alpha = n/len(edges) if len(edges) else 0.
    used = residual.copy()
    permutations = []
    if mode == 'shuffled':
        for e, (i, j) in enumerate(edges):
            pair = sorted([int(ids[i]), int(ids[j])])
            first = _permutation(k, pair, int(ids[i]))
            second = _permutation(k, pair, int(ids[j]))
            used[e] = residual[e][np.ix_(first, second)]
            permutations.append(dict(edge=e, first=first.tolist(), second=second.tolist()))
    messages = None
    if mode in ('bp', 'shuffled', 'independent', 'local'):
        supplied = np.zeros_like(used) if mode in ('independent', 'local') else used
        belief, messages = _bp(unary if mode == 'local' else a, edges, supplied, valid, alpha)
    else:
        belief = np.where(valid, a, np.inf)
        logits = np.where(valid, -a, -np.inf)
        probability = np.exp(logits-logits.max(1, keepdims=True))
        probability /= probability.sum(1, keepdims=True)
        fixed = np.argmin(np.where(valid, a, np.inf), axis=1)
        for e, (i, j) in enumerate(edges):
            if mode == 'mean':
                belief[i] += alpha*(residual[e] @ probability[j])
                belief[j] += alpha*(residual[e].T @ probability[i])
            else:
                belief[i] += alpha*residual[e, :, fixed[j]]
                belief[j] += alpha*residual[e, fixed[i], :]
        belief = np.where(valid, belief, np.inf)
    choice = np.argmin(belief, axis=1).astype(np.int64)
    need(bool(np.isfinite(belief[np.arange(n), choice]).all()), 'Finite selected beliefs')
    return dict(choice=choice, belief=belief, augmented_unary=a, centered_factors=residual,
                used_centered_factors=used, constant=constant, alpha=alpha, messages=messages,
                shuffle_permutations=permutations, mode=mode, sweeps=4 if messages is not None else 0)


def _sample(gray, centers, radius):
    q = np.arange(-radius, radius+1, dtype=np.float64)
    yy, xx = np.meshgrid(q, q, indexing='ij')
    xy = centers[..., None, :]+np.stack((xx, yy), -1).reshape(-1, 2)
    h, w = gray.shape
    valid = ((xy[..., 0] >= 0) & (xy[..., 0] <= w-1) &
             (xy[..., 1] >= 0) & (xy[..., 1] <= h-1))
    safe = np.where(valid[..., None], xy, 0.)
    lo = np.floor(safe).astype(np.int64)
    hi = np.minimum(lo+1, np.array([w-1, h-1], dtype=np.int64))
    dx, dy = safe[..., 0]-lo[..., 0], safe[..., 1]-lo[..., 1]
    patch = (gray[lo[..., 1], lo[..., 0]]*(1-dx)*(1-dy)+
             gray[lo[..., 1], hi[..., 0]]*dx*(1-dy)+
             gray[hi[..., 1], lo[..., 0]]*(1-dx)*dy+
             gray[hi[..., 1], hi[..., 0]]*dx*dy)
    return np.where(valid, patch, 0.), valid.all(-1)


def _box(center, radius):
    lo, hi = np.floor(center-radius), np.floor(center+radius)+1
    return np.array([lo[0], lo[1], hi[0], hi[1]], dtype=np.int64)


def _overlap(a, b):
    return bool(a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3])


def build_evidence(rgb0, rgb1, points0, points1, *, row_ids=None):
    """Actual image-derived unary and virtual-line bridge tables; no labels.

    Complete conservative integer footprints include bilinear neighbours.
    Supports and nearest-four graph are decided before inspecting appearance.
    Raw same-grid bridge patches assume locally translation-like transport.
    """
    need(type(points0) is np.ndarray and points0.ndim == 2, 'Point array')
    n = len(points0)
    need(0 <= n <= MAX_ROWS, 'Bounded diagnostic population, not a full factory')
    for p in (points0, points1):
        need(type(p) is np.ndarray and p.dtype == np.dtype('float32') and p.dtype.isnative
             and p.shape == (n, 2) and bool(np.isfinite(p).all()) and bool((np.abs(p) <= 2**30).all()),
             'Finite representable native FP32 Nx2 points')
    for rgb in (rgb0, rgb1):
        need(type(rgb) is np.ndarray and rgb.dtype == np.dtype('float32') and rgb.dtype.isnative
             and rgb.ndim == 3 and rgb.shape[0] == 3 and min(rgb.shape[1:]) >= 2
             and max(rgb.shape[1:]) <= 1024 and bool(np.isfinite(rgb).all()) and bool((rgb >= 0).all()),
             'Finite nonnegative native FP32 CHW RGB, <=1024; overshoot retained')
    ids = np.arange(n, dtype=np.int64) if row_ids is None else row_ids
    need(type(ids) is np.ndarray and ids.dtype == np.dtype('int64') and ids.dtype.isnative
         and ids.shape == (n,) and bool((ids >= 0).all()) and len(set(ids.tolist())) == n,
         'Unique nonnegative stable native int64 IDs')
    states = state_grid()
    bases = [p.astype(np.float64) for p in (points0, points1)]
    positions = [bases[s][:, None, :]+states[None, :, 2*s:2*s+2] for s in (0, 1)]
    boxes = [[_box(p, 6.) for p in bases[s]] for s in (0, 1)]
    geometry = np.ones(n, dtype=np.bool_)
    for s, rgb in enumerate((rgb0, rgb1)):
        h, w = rgb.shape[1:]
        for i, box in enumerate(boxes[s]):
            geometry[i] &= box[0] >= 0 and box[1] >= 0 and box[2] < w and box[3] < h
    ts = np.array([.25, .5, .75], dtype=np.float64)
    admissible, bridge_boxes = {}, {}
    for i in range(n):
        for j in range(i+1, n):
            if not (geometry[i] and geometry[j]):
                continue
            if any(float(np.linalg.norm(p[i]-p[j])) > 64 for p in bases):
                continue
            candidate, okay = [], True
            for s, rgb in enumerate((rgb0, rgb1)):
                h, w = rgb.shape[1:]
                centers = (1-ts[:, None])*bases[s][i]+ts[:, None]*bases[s][j]
                edge_boxes = [_box(center, 2.) for center in centers]
                for box in edge_boxes:
                    okay &= box[0] >= 0 and box[1] >= 0 and box[2] < w and box[3] < h
                    okay &= not any(_overlap(box, unary) for unary in boxes[s])
                candidate.append(edge_boxes)
            if okay:
                distance = sum(float(np.sum(((p[i]-p[j])/np.array([rgb.shape[2], rgb.shape[1]]))**2))
                               for p, rgb in zip(bases, (rgb0, rgb1)))
                admissible[(i, j)] = distance
                bridge_boxes[(i, j)] = candidate
    chosen = set()
    for i in range(n):
        neighbors = [(j if i == a else a, distance) for (a, j), distance in admissible.items()
                     if i in (a, j)]
        for j, _ in sorted(neighbors, key=lambda value: (value[1], int(ids[value[0]])))[:4]:
            chosen.add(tuple(sorted((i, j))))
    edges = np.array(sorted(chosen), dtype=np.int64).reshape(-1, 2)
    # All graph/support decisions above depend only on coordinates and dimensions.
    gray = [(rgb.astype(np.float64)*np.array([.299, .587, .114])[:, None, None]).sum(0)
            for rgb in (rgb0, rgb1)]
    descriptors, norms = [], []
    for s in (0, 1):
        patch, _ = _sample(gray[s], positions[s], 5)
        centered = patch-patch.mean(-1, keepdims=True)
        norm = np.linalg.norm(centered, axis=-1)
        descriptors.append(centered/np.maximum(norm[..., None], EPSILON))
        norms.append(norm)
    eligible = geometry & (norms[0][:, 0] > EPSILON) & (norms[1][:, 0] > EPSILON)
    unary = np.sum((descriptors[0]-descriptors[1])**2, axis=-1)
    unary[(norms[0] <= EPSILON) | (norms[1] <= EPSILON)] = 4.
    unary += .05*np.sum(states**2, axis=1)[None, :]
    valid = np.repeat(eligible[:, None], 81, axis=1)
    valid[:, 0] = True
    unary[~eligible] = 0.
    factors = np.zeros((len(edges), 81, 81), dtype=np.float64)
    means, scales, zeroed = [], [], []
    for e, (i, j) in enumerate(edges):
        pooled = []
        for s in (0, 1):
            centers = ((1-ts[None, None, :, None])*positions[s][i, :, None, None, :]+
                       ts[None, None, :, None]*positions[s][j, None, :, None, :])
            patch, supported = _sample(gray[s], centers, 1)
            need(bool(supported.all()), 'Complete bridge support')
            pooled.append(patch.reshape(81, 81, 27))
        initial = np.concatenate([p[0, 0] for p in pooled])
        mean = float(initial.mean())
        scale = float(np.sqrt(sum(np.sum((p[0, 0]-mean)**2) for p in pooled)/2))
        means.append(mean); scales.append(scale)
        if not (eligible[i] and eligible[j]) or scale <= EPSILON:
            zeroed.append(e)
            continue
        factors[e] = np.minimum(4., np.sum(((pooled[0]-pooled[1])/scale)**2, axis=-1))
    return dict(unary=unary, valid=valid, edges=edges, factors=factors,
                positions0=positions[0], positions1=positions[1], row_ids=ids.copy(),
                diagnostics=dict(initial_eligible=np.flatnonzero(eligible).tolist(),
                    unary_support_boxes=np.array(boxes, dtype=np.int64).reshape(2, n, 4),
                    bridge_support_boxes=np.array([bridge_boxes[tuple(e)] for e in edges],
                        dtype=np.int64).reshape(len(edges), 2, 3, 4).transpose(1, 0, 2, 3),
                    bridge_mu=np.array(means), bridge_scale=np.array(scales),
                    zeroed_bridge_edges=zeroed, graph_before_appearance=True,
                    support='conservative integer bilinear footprints; disjoint from ALL unary supports',
                    classification='known K-VLD/photometric factor adaptation; novelty UNESTABLISHED'))


def refine_pair(rgb0, rgb1, points0, points1, *, mode, row_ids=None):
    evidence = build_evidence(rgb0, rgb1, points0, points1, row_ids=row_ids)
    result = infer(evidence['unary'], evidence['edges'], evidence['factors'], evidence['valid'],
                   mode=mode, row_ids=evidence['row_ids'])
    selected = result['choice']
    out = [points0.copy(), points1.copy()]
    moved = selected != 0
    for s in (0, 1):
        out[s][moved] = evidence['positions'+str(s)][np.arange(len(selected)), selected][moved].astype(np.float32)
    return dict(points0=out[0], points1=out[1], evidence=evidence, inference=result)
