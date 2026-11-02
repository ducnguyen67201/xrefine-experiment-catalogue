"""RM050 fixed weak-risk arithmetic. No data access or model execution.

Action columns are QQ, PP, PQ, QP, HALF. Only the first four are learned.
Ridge scores are unconstrained regression outputs, not probabilities.
"""
import numpy as np


ACTIONS = ("QQ", "PP", "PQ", "QP", "HALF")


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _real(value, shape, name):
    value = np.asarray(value)
    _need(value.dtype.kind in "fiu" and value.shape == shape, name + " shape/type")
    result = np.asarray(value, dtype=np.float64)
    _need(np.isfinite(result).all(), name + " nonfinite")
    return result


def _matrix(value, columns, name):
    value = np.asarray(value)
    _need(value.ndim == 2 and value.shape[1] == columns, name + " columns")
    return _real(value, value.shape, name)


def _coordinates(cache):
    p0 = np.asarray(cache["p0"])
    _need(p0.ndim == 2 and p0.shape[1] == 2, "p0 shape")
    result = []
    for key in ("p0", "p1", "q0", "q1"):
        value = np.asarray(cache[key])
        _need(value.dtype == np.float32 and value.shape == p0.shape
              and np.isfinite(value).all(), "finite FP32 cache " + key)
        result.append(value)
    return tuple(result)


def displacement(cache):
    """Return signed q0-p0, q1-p1 in FP64, using actual native p values."""
    p0, p1, q0, q1 = _coordinates(cache)
    return np.concatenate((q0.astype(np.float64) - p0,
                           q1.astype(np.float64) - p1), axis=1)


def features(cache):
    """Flatten own x0, x1 in C order, followed by four signed displacements."""
    d = displacement(cache)
    n = len(d)
    parts = []
    for key in ("x0", "x1"):
        x = np.asarray(cache[key])
        _need(x.dtype == np.float32 and x.shape == (n, 64, 3, 3)
              and np.isfinite(x).all(), "finite FP32 cache " + key)
        parts.append(x.reshape(n, 576).astype(np.float64))
    return np.concatenate((*parts, d), axis=1)


class FeatureMoments:
    """Streaming FP64 population moments; only training rows belong here."""
    def __init__(self, dimension):
        _need(type(dimension) is int and dimension > 0, "positive feature dimension")
        self.count = 0
        self.mean = np.zeros(dimension, np.float64)
        self.m2 = np.zeros(dimension, np.float64)

    def add(self, x):
        x = _matrix(x, len(self.mean), "moment features")
        n = len(x)
        if not n:
            return
        local_mean = x.mean(axis=0, dtype=np.float64)
        centered = x - local_mean
        local_m2 = (centered * centered).sum(axis=0, dtype=np.float64)
        delta = local_mean - self.mean
        total = self.count + n
        self.m2 += local_m2 + delta * delta * (self.count * n / total)
        self.mean += delta * (n / total)
        self.count = total
        _need(np.isfinite(self.mean).all() and np.isfinite(self.m2).all(),
              "finite accumulated moments")

    def finish(self):
        _need(self.count > 0, "nonempty training moments")
        std = np.sqrt(np.maximum(self.m2 / self.count, 0.0))
        std = np.where(std < 1e-12, 1.0, std)
        return self.mean.copy(), std


def _normalizer(mean, std):
    mean = np.asarray(mean)
    _need(mean.ndim == 1 and len(mean) > 0, "normalizer dimension")
    mean = _real(mean, mean.shape, "mean")
    std = _real(std, mean.shape, "std")
    _need((std > 0).all(), "positive standard deviations")
    return mean, std


def _design(x, mean, std):
    x = _matrix(x, len(mean), "ridge features")
    z = (x - mean) / std
    _need(np.isfinite(z).all(), "finite standardized features")
    return np.column_stack((z, np.ones(len(x), dtype=np.float64)))


class RidgeAccumulator:
    """SUM squared loss + ||W_nonintercept||², fixed alpha=1, four outputs."""
    def __init__(self, mean, std):
        mean, std = _normalizer(mean, std)
        self.mean, self.std = mean.copy(), std.copy()
        d = len(mean) + 1
        self.normal = np.zeros((d, d), np.float64)
        self.cross = np.zeros((d, 4), np.float64)
        self.count = 0

    def add(self, x, y):
        z = _design(x, self.mean, self.std)
        y = _real(y, (len(z), 4), "four training risks")
        self.normal += z.T @ z
        self.cross += z.T @ y
        self.count += len(z)
        _need(np.isfinite(self.normal).all() and np.isfinite(self.cross).all(),
              "finite ridge sufficient statistics")

    def solve(self):
        _need(self.count > 0, "nonempty ridge training")
        lhs = self.normal.copy()
        lhs[np.diag_indices(len(lhs) - 1)] += 1.0
        try:
            weights = np.linalg.solve(lhs, self.cross)
        except np.linalg.LinAlgError as exc:
            raise ValueError("nonsingular ridge system") from exc
        _need(np.isfinite(weights).all(), "finite fitted ridge weights")
        return weights


def predict(x, mean, std, weights):
    mean, std = _normalizer(mean, std)
    weights = _real(weights, (len(mean) + 1, 4), "ridge weights")
    result = _design(x, mean, std) @ weights
    _need(np.isfinite(result).all(), "finite unconstrained risk scores")
    return result


def fundamental(K0, K1, R, t):
    """Directed FP64 K1^-T [t/||t||]x R K0^-1, then Frobenius normalization.

    Same geometry as RM042's qualified helper. Zero translation is rejected
    here, rather than silently removing rows or returning an absent target.
    """
    K0 = _real(K0, (3, 3), "K0")
    K1 = _real(K1, (3, 3), "K1")
    R = _real(R, (3, 3), "R")
    t = _real(t, (3,), "t")
    _need(np.max(np.abs(R.T @ R - np.eye(3))) <= 1e-5
          and abs(np.linalg.det(R) - 1.0) <= 1e-5,
          "proper directed rotation without projection")
    try:
        inverse0, inverse1 = np.linalg.inv(K0), np.linalg.inv(K1)
    except np.linalg.LinAlgError as exc:
        raise ValueError("nonsingular calibrated cameras") from exc
    norm = np.linalg.norm(t)
    _need(np.isfinite(norm) and norm > 0.0, "nonzero finite translation norm")
    tx, ty, tz = t / norm
    cross = np.array(((0.0, -tz, ty), (tz, 0.0, -tx), (-ty, tx, 0.0)))
    F = inverse1.T @ cross @ R @ inverse0
    scale = np.linalg.norm(F)
    _need(np.isfinite(F).all() and np.isfinite(scale) and scale > 0.0,
          "finite nonzero pixel fundamental matrix")
    return F / scale


def risk(p0, p1, F):
    """Per-row log1p squared pixel Sampson; caller supplies normalized F."""
    p0 = _matrix(p0, 2, "risk endpoint0")
    p1 = _real(p1, p0.shape, "risk endpoint1")
    F = _real(F, (3, 3), "risk F")
    _need(np.linalg.norm(F) > 0.0, "nonzero risk F")
    a = np.column_stack((p0, np.ones(len(p0), dtype=np.float64)))
    b = np.column_stack((p1, np.ones(len(p1), dtype=np.float64)))
    line1, line0 = a @ F.T, b @ F
    residual = (b * line1).sum(axis=1, dtype=np.float64)
    denominator = ((line1[:, :2] ** 2).sum(axis=1, dtype=np.float64)
                   + (line0[:, :2] ** 2).sum(axis=1, dtype=np.float64))
    result = np.log1p(residual ** 2 / np.maximum(denominator, 1e-12))
    _need(np.isfinite(result).all(), "finite all-row weak risk")
    return result


def endpoints(cache, choices):
    """Return FP32 endpoints, with every p/q choice preserving its exact bytes."""
    p0, p1, q0, q1 = _coordinates(cache)
    choices = np.asarray(choices)
    _need(choices.dtype == np.int64 and choices.shape == (len(p0),)
          and ((choices >= 0) & (choices < 5)).all(), "one declared action per row")
    out0 = np.where(((choices == 0) | (choices == 3))[:, None], q0, p0)
    out1 = np.where(((choices == 0) | (choices == 2))[:, None], q1, p1)
    half = choices == 4
    out0[half] = ((p0[half].astype(np.float64) + q0[half]) / 2.0).astype(np.float32)
    out1[half] = ((p1[half].astype(np.float64) + q1[half]) / 2.0).astype(np.float32)
    return out0, out1


def action_risks(cache, F):
    """All five fixed action risks, with learned actions first in tie order."""
    n = len(_coordinates(cache)[0])
    return np.column_stack([risk(*endpoints(cache, np.full(n, a, np.int64)), F)
                            for a in range(5)])


def choose(predicted):
    predicted = _matrix(predicted, 4, "four predicted risk scores")
    return np.argmin(predicted, axis=1).astype(np.int64)


def policy_loss(cache, F, choices):
    """Weak risk of exact row-preserving selected endpoints; no filtering."""
    return risk(*endpoints(cache, choices), F)
