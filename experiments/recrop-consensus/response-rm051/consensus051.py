"""RM051 fixed local recrop consensus; pure arithmetic, no payload access.

Branches are (baseline, +x0, -x0, +y0, -y0, +x1, -x1, +y1, -y1).
Agreement is not calibrated uncertainty or physical correspondence truth.
"""
import hashlib

import numpy as np


BRANCHES = 9
RADIUS_SQUARED = 1.0
MAJORITY = 5


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _points(value, n=None, name="points"):
    value = np.asarray(value)
    _need(value.dtype == np.float32 and value.ndim == 2
          and value.shape[1] == 2 and (n is None or len(value) == n)
          and np.isfinite(value).all(), "finite FP32 Nx2 " + name)
    return value


def _hw(value):
    _need(type(value) in (tuple, list) and len(value) == 2
          and all(type(v) is int and v > 0 for v in value),
          "positive integer (height, width)")
    return value


def edge_mask(p0, p1, hw0, hw1):
    """True for a row needing common baseline fallback for all recrop arms.

    Preserve finite native overshoot from the qualified original population.
    Outside coordinates are edge fallback, never clipped, filtered or rejected.
    At a nonedge row all integer anchor shifts remain inside; full radius-five
    containment is not required because the original extractor zero-pads.
    """
    p0 = _points(p0, name="native endpoint0")
    p1 = _points(p1, len(p0), "native endpoint1")
    edge = np.zeros(len(p0), dtype=np.bool_)
    for p, hw in ((p0, _hw(hw0)), (p1, _hw(hw1))):
        h, w = hw
        anchor = np.floor(p.astype(np.float64))
        edge |= ((anchor[:, 0] < 1) | (anchor[:, 0] > w - 2)
                 | (anchor[:, 1] < 1) | (anchor[:, 1] > h - 2))
    return edge


def _branches(value, n=None):
    value = np.asarray(value)
    _need(value.dtype == np.float32 and value.ndim == 3
          and value.shape[0] == BRANCHES and value.shape[2] == 2
          and (n is None or value.shape[1] == n) and np.isfinite(value).all(),
          "finite FP32 nine-branch (9,N,2) array")
    return value


def _indices(row_ids, n):
    row_ids = np.asarray(row_ids)
    _need(row_ids.dtype == np.int64 and row_ids.shape == (n,)
          and (row_ids >= 0).all() and len(np.unique(row_ids)) == n,
          "unique nonnegative original int64 row IDs")
    return row_ids


def _pair_seed(pair_uid):
    _need(type(pair_uid) is str and bool(pair_uid)
          and all(32 <= ord(c) < 127 for c in pair_uid), "printable ASCII pair UID")
    body = ("RM051/20261102/" + pair_uid).encode("ascii")
    return int.from_bytes(hashlib.sha256(body).digest()[:8], "little")


def outputs(branches0, branches1, edge_mask, row_ids, pair_uid):
    """Return candidate and four fixed controls, without sorting/dropping rows.

    C/U/F/S choose paired endpoint bytes from one actual recipient-row branch.
    M takes FP64 coordinatewise medians, cast to FP32. Every edge row uses the
    exact baseline for all controls regardless of supplied shifted responses.
    shuffle_mapping is a source-position permutation, not a row-ID remapping.
    support/energy describe raw best; effective choices additionally apply the
    majority (C) and common edge fallback (every arm).
    """
    b0 = _branches(branches0)
    n = b0.shape[1]
    b1 = _branches(branches1, n)
    row_ids = _indices(row_ids, n)
    edge = np.asarray(edge_mask)
    _need(edge.dtype == np.bool_ and edge.shape == (n,), "one bool edge flag per row")
    seed = _pair_seed(pair_uid)

    a0, a1 = b0.astype(np.float64), b1.astype(np.float64)
    delta0 = a0[:, None] - a0[None, :]
    delta1 = a1[:, None] - a1[None, :]
    d2 = np.maximum((delta0 * delta0).sum(axis=-1, dtype=np.float64),
                    (delta1 * delta1).sum(axis=-1, dtype=np.float64))
    _need(np.isfinite(d2).all(), "finite FP64 paired distances")
    neighbors = d2 <= RADIUS_SQUARED
    support = neighbors.sum(axis=1, dtype=np.int64)
    energy = np.where(neighbors, d2, 0.0).sum(axis=1, dtype=np.float64)
    rows = np.arange(n, dtype=np.int64)
    best = np.zeros(n, dtype=np.int64)
    # Strict updates preserve the first branch rank on an exact support/energy tie.
    for k in range(1, BRANCHES):
        current_support, current_energy = support[best, rows], energy[best, rows]
        better = ((support[k] > current_support)
                  | ((support[k] == current_support) & (energy[k] < current_energy)))
        best[better] = k
    chosen_support, chosen_energy = support[best, rows], energy[best, rows]
    choice = np.where(chosen_support >= MAJORITY, best, 0).astype(np.int64)
    choice[edge] = 0
    ungated = best.copy()
    ungated[edge] = 0
    fixed = np.ones(n, dtype=np.int64)
    fixed[edge] = 0

    eligible = np.flatnonzero(~edge)
    mapping = rows.copy()
    mapping[eligible] = eligible[np.random.default_rng(seed).permutation(len(eligible))]
    shuffled = choice[mapping]
    result = dict(choice=choice, best=best, ungated_choice=ungated,
                  fixed_choice=fixed, shuffled_choice=shuffled,
                  support=chosen_support, energy=chosen_energy,
                  shuffle_mapping=mapping, edge=edge.copy(), row_ids=row_ids.copy())
    for prefix, selected in (("C", choice), ("U", ungated), ("F", fixed), ("S", shuffled)):
        result[prefix + "0"] = b0[selected, rows].copy()
        result[prefix + "1"] = b1[selected, rows].copy()
    for side, a in ((0, a0), (1, a1)):
        median = np.median(a, axis=0).astype(np.float32)
        median[edge] = (b0 if side == 0 else b1)[0, edge]
        result["M" + str(side)] = median
    return result
