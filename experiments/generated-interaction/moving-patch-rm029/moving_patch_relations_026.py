"""RM026: observable S0 derivative of preserved moving_patch_relations_023.

SOURCE ONLY / UNTESTED. Same joint science:11x11 luma patches, four steps,
1px Euclidean trust ball, unary1/relation1/prior.1, step.25*N. New controls:
literal mean and full frozen initial edge-residual Jacobian. No new image,
truth, weight, scorer, native solver, controller, file IO or benchmark factory.
Norm clamping is unchanged: initial eligibility is fixed; later flat patches
may collapse evidence. Diagnostics expose that failure, without filtering it.
"""
from types import SimpleNamespace
import random


METADATA = dict(assignment="RM-20261026-026", parent="RM-20261026-023",
    parent_sha256="bd6aac47458d6fe5554617f2cb7d618681fb4a5372a1a1046906f1bb12c193c2",
    scope="bounded premise S0; no production factory", max_rows=64, patch_side=11,
    steps=4, radius_px=1.0, step_px_squared=0.25, relation_weight=1.0,
    displacement_weight=0.1, modes=("joint", "frozen", "independent", "mean", "linearized"),
    classification="known relational featuremetric adaptation; novelty UNESTABLISHED",
    training=False, actual_tests="NONE")


def need(ok, message):
    if not ok:
        raise ValueError(message)


def _points(points0, points1, np):
    need(type(points0) is np.ndarray and points0.ndim == 2, "Point array")
    n = len(points0)
    need(0 <= n <= 64, "Bounded premise population")
    for value in (points0, points1):
        need(type(value) is np.ndarray and value.dtype == np.dtype("float32") and value.dtype.isnative and
             value.shape == (n, 2) and bool(np.isfinite(value).all()), "Finite native FP32 points")
    return n


def _edges(edges, n, np):
    need(type(edges) is np.ndarray and edges.dtype == np.dtype("int64") and edges.dtype.isnative and
         edges.ndim == 2 and edges.shape[1] == 2 and len(edges) <= 4*n and
         list(map(tuple, edges.tolist())) == sorted(set(map(tuple, edges.tolist()))) and
         all(0 <= i < j < n for i, j in edges.tolist()), "Unique ordered undirected edges")


def _positions(points0, points1, hw0, hw1, np):
    _points(points0, points1, np)
    for hw in (hw0, hw1):
        need(type(hw) is tuple and len(hw) == 2 and all(type(v) is int and v > 0 for v in hw), "Raster dimensions")
    return np.concatenate((points0/np.array((hw0[1], hw0[0]), dtype=np.float64),
                           points1/np.array((hw1[1], hw1[0]), dtype=np.float64)), axis=1)


def spatial_edges(points0, points1, hw0, hw1, *, row_ids=None):
    """Same4-neighbour union. Default ties use ARRAY INDICES, as023 actually did.

    Optional nativeint64 unique nonnegative row_ids give stable original-ID
    ties. Edges always index current arrays. Operator equivariance with fixed
    edges is separate from graph recomputation with/without stable identities.
    """
    import numpy as np

    z = _positions(points0, points1, hw0, hw1, np)
    n = len(z)
    ids = np.arange(n, dtype=np.int64) if row_ids is None else row_ids
    need(type(ids) is np.ndarray and ids.dtype == np.dtype("int64") and ids.dtype.isnative and
         ids.shape == (n,) and bool((ids >= 0).all()) and len(set(ids.tolist())) == n, "Unique stable row IDs")
    edges = set()
    for i in range(n):
        order = sorted((j for j in range(n) if j != i),
                       key=lambda j: (float(((z[i]-z[j])**2).sum()), int(ids[j])))
        for j in order[:4]:
            edges.add((min(i, j), max(i, j)))
    return np.array(sorted(edges), dtype=np.int64).reshape(-1, 2)


def shuffle_edges(edges, points0, points1, hw0, hw1, *, seed=42, attempts=None, eligible_rows=None):
    """Genuine bounded two-switch rewiring; never a mere node permutation.

    Switch only edges with four distinct eligible endpoints. Thus full AND
    eligibility-filtered degree vectors remain fixed. Caller must pass the
    actual initial eligibility from evaluate_pair for a real filtered control;
    omitted eligibility explicitly assumes all rows. This finite procedure is
    not a uniform graph sampler. Complete graphs can have no valid shuffle.
    """
    import numpy as np

    z = _positions(points0, points1, hw0, hw1, np)
    n = len(z)
    _edges(edges, n, np)
    need(type(seed) is int, "Integer shuffle seed")
    limit = 8*max(1, len(edges)) if attempts is None else attempts
    need(type(limit) is int and 0 <= limit <= 4096, "Finite switch attempts")
    rows = list(range(n)) if eligible_rows is None else eligible_rows
    need(type(rows) is list and rows == sorted(set(rows)) and
         all(type(i) is int and 0 <= i < n for i in rows), "Actual eligible row indices")
    active = set(rows)
    original = set(map(tuple, edges.tolist()))
    current = set(original)
    rng = random.Random(seed)
    accepted, rejected, switches = 0, 0, []
    for attempt in range(limit):
        choices = sorted(e for e in current if set(e) <= active)
        if len(choices) < 2:
            rejected += 1
            continue
        first, second = rng.sample(choices, 2)
        a, b = first
        c, d = second
        if len({a, b, c, d}) != 4:
            rejected += 1
            continue
        proposed = ((a, c), (b, d)) if rng.randrange(2) else ((a, d), (b, c))
        proposed = {tuple(sorted(e)) for e in proposed}
        removed = {first, second}
        rest = current-removed
        candidate = rest | proposed
        if len(proposed) != 2 or proposed & rest or candidate == current or candidate == original:
            rejected += 1
            continue
        switches.append(dict(attempt=attempt, removed=sorted(removed), added=sorted(proposed)))
        current = candidate
        accepted += 1

    def degrees(graph, filtered=False):
        values = [0]*n
        for i, j in graph:
            if not filtered or i in active and j in active:
                values[i] += 1
                values[j] += 1
        return values

    def distances(graph):
        return sorted(float(np.sqrt(((z[i]-z[j])**2).sum())) for i, j in graph)

    informative = current != original
    before, after = degrees(original), degrees(current)
    ebefore, eafter = degrees(original, True), degrees(current, True)
    need(before == after and ebefore == eafter and len(current) == len(original), "True degree/count preservation")
    result = np.array(sorted(current), dtype=np.int64).reshape(-1, 2)
    record = dict(seed=seed, attempts=limit, valid_switches=accepted, rejected_attempts=rejected,
        switches=switches, changed_edges=len(original.symmetric_difference(current)),
        degree_before=before, degree_after=after,
        effective_degree_before=ebefore, effective_degree_after=eafter,
        eligible_rows=rows, eligibility="all_rows_assumed" if eligible_rows is None else "caller_supplied_initial",
        normalized_lengths_before=distances(original), normalized_lengths_after=distances(current),
        informative=informative, reason="changed_by_real_switches" if informative else "UNINFORMATIVE_NO_CHANGED_GRAPH")
    return dict(edges=result, record=record)


def _problem(rgb0, rgb1, points0, points1, edges):
    import numpy as np
    import torch
    import torch.nn.functional as F

    n = _points(points0, points1, np)
    _edges(edges, n, np)
    for rgb in (rgb0, rgb1):
        need(type(rgb) is np.ndarray and rgb.dtype == np.dtype("float32") and rgb.dtype.isnative and
             rgb.ndim == 3 and rgb.shape[0] == 3 and 2 <= min(rgb.shape[1:]) and max(rgb.shape[1:]) <= 1024 and
             bool(np.isfinite(rgb).all()) and bool((rgb >= 0).all()), "Finite nonnegative CHW RGB, <=1024; above1 retained")
    p = SimpleNamespace(np=np, torch=torch, F=F, n=n, dtype=torch.float64, linear=None)
    p.base = tuple(torch.from_numpy(x.copy()).to(dtype=p.dtype, device="cpu") for x in (points0, points1))
    weights = torch.tensor([.299, .587, .114], dtype=p.dtype).view(3, 1, 1)
    p.images = tuple((torch.from_numpy(x.copy()).to(dtype=p.dtype, device="cpu")*weights).sum(0)[None, None]
                     for x in (rgb0, rgb1))
    q = torch.arange(-5, 6, dtype=p.dtype)
    yy, xx = torch.meshgrid(q, q, indexing="ij")
    p.offsets = torch.stack((xx, yy), dim=-1).reshape(1, 121, 2)
    p.zero = torch.zeros((n, 2), dtype=p.dtype)
    sampled = tuple(_sample(p, s, p.zero) for s in (0, 1))
    p.originals = tuple(x[0].detach() for x in sampled)
    p.valid = torch.ones(n, dtype=torch.bool)
    for s in (0, 1):
        h, w = p.images[s].shape[-2:]
        p.valid &= ((p.base[s] >= 6).all(1) & (p.base[s][:, 0] <= w-7) &
                    (p.base[s][:, 1] <= h-7) & (sampled[s][1] > 1e-6))
    e = torch.from_numpy(edges.copy())
    p.e = e[p.valid[e[:, 0]] & p.valid[e[:, 1]]]
    return p


def _sample(p, side, delta):
    if p.n == 0:
        return p.torch.empty((0, 121), dtype=p.dtype), p.torch.empty(0, dtype=p.dtype)
    image = p.images[side]
    h, w = image.shape[-2:]
    xy = p.base[side][:, None, :]+delta[:, None, :]+p.offsets
    grid = p.torch.stack((2*(xy[..., 0]+.5)/w-1, 2*(xy[..., 1]+.5)/h-1), dim=-1)
    patch = p.F.grid_sample(image, grid[None], mode="bilinear", padding_mode="zeros",
                           align_corners=False).reshape(p.n, 121)
    patch = patch-patch.mean(1, keepdim=True)
    norm = p.torch.linalg.vector_norm(patch, dim=1, keepdim=True)
    return patch/norm.clamp_min(1e-6), norm[:, 0]


def _residual(p, descriptors):
    i, j = p.e[:, 0], p.e[:, 1]
    return (descriptors[0][i]*descriptors[0][j]).sum(1)-(descriptors[1][i]*descriptors[1][j]).sum(1)


def _linear_reference(p):
    """Freeze full r(0), J(0): retain cross-match J^T J blocks.

    This drops residual-weighted second derivatives of the moving residual;
    it is not the full nonlinear Hessian or a new optimizer.
    """
    if p.linear is not None:
        return p.linear
    torch = p.torch
    if not len(p.e):
        p.linear = (torch.empty(0, dtype=p.dtype), torch.empty((0, 4*p.n), dtype=p.dtype))
        return p.linear
    delta = tuple(p.zero.clone().requires_grad_(True) for _ in (0, 1))
    r = _residual(p, tuple(_sample(p, s, delta[s])[0] for s in (0, 1)))
    rows = []
    for k in range(len(r)):
        values = torch.autograd.grad(r[k], delta, retain_graph=k+1 < len(r))
        rows.append(torch.cat(tuple(v.reshape(-1) for v in values)).detach())
    p.linear = (r.detach(), torch.stack(rows))
    return p.linear


def _loss(p, delta, mode):
    sampled = tuple(_sample(p, s, delta[s]) for s in (0, 1))
    d = tuple(v[0] for v in sampled)
    unary = (((d[0]-d[1])**2).sum(1)*p.valid).sum()/max(1, p.n)
    if len(p.e):
        i, j = p.e[:, 0], p.e[:, 1]
        moving = _residual(p, d).square().mean()
        left = (d[0][i]*p.originals[0][j]).sum(1)-(d[1][i]*p.originals[1][j]).sum(1)
        right = (d[0][j]*p.originals[0][i]).sum(1)-(d[1][j]*p.originals[1][i]).sum(1)
        initial = _residual(p, p.originals).square().mean()
        frozen = left.square().mean()+right.square().mean()-initial
    else:
        moving = frozen = unary*0
    mean = (d[0][p.valid].mean(0)-d[1][p.valid].mean(0)).square().sum() if bool(p.valid.any()) else unary*0
    linearized = None
    if mode == "linearized":
        r0, j0 = _linear_reference(p)
        stacked = p.torch.cat(tuple(v.reshape(-1) for v in delta))
        linearized = (r0+j0@stacked).square().mean() if len(p.e) else unary*0
    relation = {"joint": moving, "frozen": frozen, "independent": moving*0, "mean": mean,
                "linearized": linearized}[mode]
    prior = sum((v.square().sum(1)*p.valid).sum() for v in delta)/(2*max(1, p.n))
    terms = dict(unary=unary, moving=moving, frozen=frozen, mean=mean, linearized=linearized,
                 relational=relation, prior=prior)
    return unary+relation+.1*prior, terms, sampled


def _evaluation(p, delta, mode):
    torch = p.torch
    loss, terms, sampled = _loss(p, delta, mode)
    need(bool(torch.isfinite(loss).item()), "Finite appearance objective")
    gradients = torch.autograd.grad(loss, delta) if p.n else (p.zero.clone(), p.zero.clone())
    need(all(bool(torch.isfinite(v).all()) for v in gradients), "Finite appearance gradients")
    record = dict(loss=float(loss.detach()), terms={k: None if v is None else float(v.detach()) for k, v in terms.items()},
        gradient0=gradients[0].detach().numpy().copy(), gradient1=gradients[1].detach().numpy().copy(),
        descriptors0=sampled[0][0].detach().numpy().copy(), descriptors1=sampled[1][0].detach().numpy().copy(),
        norm0=sampled[0][1].detach().numpy().copy(), norm1=sampled[1][1].detach().numpy().copy(),
        eligible_rows=p.np.flatnonzero(p.valid.numpy()).tolist(), effective_edges=p.e.numpy().copy(),
        collapse_rows=p.np.flatnonzero((p.valid & ((sampled[0][1] <= 1e-6) | (sampled[1][1] <= 1e-6))).detach().numpy()).tolist(),
        normalization=dict(population=p.n, eligible=int(p.valid.sum()), edges=len(p.e), mean_denominator="eligible_rows"),
        weights=dict(unary=1.0, relation=1.0, prior=.1), gradient_scale="dL/dDelta; UNMULTIPLIED")
    record["linearized_reference"] = None if p.linear is None else dict(
        residual0=p.linear[0].numpy().copy(), jacobian0=p.linear[1].numpy().copy(),
        order="side0 rowmajor[x,y], then side1 rowmajor[x,y]")
    return record, gradients


def evaluate_pair(rgb0, rgb1, points0, points1, edges, delta0, delta1, *, mode):
    """Same production evaluator at explicit FP64 pixel deltas, with unscaled grads.

    This is an observation seam for an INDEPENDENT scalar/finite-difference
    oracle, not an alternative implementation used to validate itself.
    """
    need(mode in METADATA["modes"], "Declared mode")
    p = _problem(rgb0, rgb1, points0, points1, edges)
    for v in (delta0, delta1):
        need(type(v) is p.np.ndarray and v.dtype == p.np.dtype("float64") and v.dtype.isnative and
             v.shape == (p.n, 2) and bool(p.np.isfinite(v).all()) and
             bool((p.np.linalg.norm(v, axis=1) <= 1+1e-12).all()), "Finite FP64 pixel delta in trust ball")
    delta = tuple(p.torch.from_numpy(v.copy()).requires_grad_(True) for v in (delta0, delta1))
    return _evaluation(p, delta, mode)[0]


def refine_pair(rgb0, rgb1, points0, points1, edges, *, mode):
    """Same four production updates, now with initial/step/final/norm diagnostics.

    Fixed INITIAL eligibility is never recomputed as a filter. Later flatness
    is exposed as collapse_rows. Mean is a literal control; its gradient is
    generally not a scalar multiple of the joint gradient. No matching claim.
    """
    need(mode in METADATA["modes"], "Declared mode")
    p = _problem(rgb0, rgb1, points0, points1, edges)
    before = tuple(v.tobytes() for v in (rgb0, rgb1, points0, points1, edges))
    delta = [p.zero.clone(), p.zero.clone()]
    trace = []
    for _ in range(4 if p.n else 0):
        values = tuple(v.detach().requires_grad_(True) for v in delta)
        evaluation, gradients = _evaluation(p, values, mode)
        start = [v.detach().numpy().copy() for v in values]
        updated = []
        for v, gradient in zip(values, gradients):
            trial = v.detach()-.25*p.n*gradient.detach()
            trial = trial/p.torch.linalg.vector_norm(trial, dim=1, keepdim=True).clamp_min(1)
            updated.append(p.torch.where(p.valid[:, None], trial, p.torch.zeros_like(trial)))
        trace.append(dict(evaluation=evaluation, delta_before0=start[0], delta_before1=start[1],
            delta_after0=updated[0].numpy().copy(), delta_after1=updated[1].numpy().copy()))
        delta = updated
    values = tuple(v.detach().requires_grad_(True) for v in delta)
    final = _evaluation(p, values, mode)[0]
    out = [points0.copy(), points1.copy()]
    mask = p.valid.numpy()
    for s in (0, 1):
        out[s][mask] = (p.base[s]+delta[s]).detach().numpy()[mask].astype(p.np.float32)
        need(bool(p.np.isfinite(out[s]).all()), "Finite emitted coordinates")
    need(before == tuple(v.tobytes() for v in (rgb0, rgb1, points0, points1, edges)), "Input mutation")
    return dict(points0=out[0], points1=out[1], eligible_rows=p.np.flatnonzero(mask).tolist(),
        diagnostics=dict(mode=mode, steps=len(trace), effective_edges=len(p.e),
            initial=trace[0]["evaluation"] if trace else final, trace=trace, final=final,
            final_delta0=delta[0].numpy().copy(), final_delta1=delta[1].numpy().copy(),
            radius_px=1.0, applied_gradient_multiplier=p.n, step_px_squared=.25,
            mean_gradient_scalar_matched=False, eligibility="INITIAL_ONLY; clamp-norm objective unchanged",
            accuracy_evaluated=False, no_GT_or_paths_supplied=True))
