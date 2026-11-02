"""RM026 independent fixture SOURCE ONLY; no tests executed by its author.

NumPy, Torch and the production module are imported only inside owner-run
fixtures. The energy/sampling oracle below uses Python math and scalar access,
never a production helper, grid_sample, autograd or NumPy numerical reduction.
Finite differences are centred at fractional coordinates away from pixel knots.
"""
from __future__ import annotations

import importlib
import math

import pytest


MODES = ("joint", "frozen", "independent", "mean", "linearized")
EPS = 1e-6
FD = 1e-5


def _dot(a, b):
    return math.fsum(x * y for x, y in zip(a, b))


def _luma(rgb, x, y):
    """Scalar zero-padded bilinear interpolation at array-index coordinates."""
    h, w = rgb.shape[1:]
    ix, iy = math.floor(x), math.floor(y)
    fx, fy = x - ix, y - iy
    total = []
    for c, weight in enumerate((0.299, 0.587, 0.114)):
        values = []
        for dx, wx in ((0, 1 - fx), (1, fx)):
            for dy, wy in ((0, 1 - fy), (1, fy)):
                xx, yy = ix + dx, iy + dy
                value = float(rgb[c, yy, xx]) if 0 <= xx < w and 0 <= yy < h else 0.0
                values.append(wx * wy * value)
        total.append(weight * math.fsum(values))
    return math.fsum(total)


def _describe(rgb, points, delta):
    descriptors, norms = [], []
    for point, shift in zip(points, delta):
        x, y = float(point[0]) + shift[0], float(point[1]) + shift[1]
        raw = [_luma(rgb, x + dx, y + dy) for dy in range(-5, 6) for dx in range(-5, 6)]
        center = math.fsum(raw) / 121
        centered = [value - center for value in raw]
        norm = math.sqrt(_dot(centered, centered))
        descriptors.append([value / max(EPS, norm) for value in centered])
        norms.append(norm)
    return descriptors, norms


def _residuals(d0, d1, edges):
    return [_dot(d0[i], d0[j]) - _dot(d1[i], d1[j]) for i, j in edges]


def _context(rgb0, rgb1, p0, p1, edges, *, linear=False):
    n = len(p0)
    zero = [[0.0, 0.0] for _ in range(n)]
    originals, norms = [], []
    for rgb, points in ((rgb0, p0), (rgb1, p1)):
        descriptor, norm = _describe(rgb, points, zero)
        originals.append(descriptor); norms.append(norm)
    valid = []
    for i in range(n):
        ok = True
        for s, (rgb, points) in enumerate(((rgb0, p0), (rgb1, p1))):
            h, w = rgb.shape[1:]
            x, y = map(float, points[i])
            ok = ok and 6 <= x <= w - 7 and 6 <= y <= h - 7 and norms[s][i] > EPS
        if ok:
            valid.append(i)
    active = set(valid)
    effective = [(int(i), int(j)) for i, j in edges if int(i) in active and int(j) in active]
    context = dict(rgb=(rgb0, rgb1), points=(p0, p1), n=n, valid=valid,
                   edges=effective, original=originals,
                   r0=_residuals(*originals, effective), jacobian=None)
    if linear:
        # Independent scalar FD of the edge residual, not FD of production loss.
        jacobian = [[0.0] * (4 * n) for _ in effective]
        for column in range(4 * n):
            side, local = divmod(column, 2 * n)
            row, axis = divmod(local, 2)
            residual = []
            for direction in (-1, 1):
                shifts = [[[0.0, 0.0] for _ in range(n)] for _ in range(2)]
                shifts[side][row][axis] = direction * FD
                ds = [_describe(rgb, points, shift)[0]
                      for rgb, points, shift in zip(context["rgb"], context["points"], shifts)]
                residual.append(_residuals(*ds, effective))
            for edge in range(len(effective)):
                jacobian[edge][column] = (residual[1][edge] - residual[0][edge]) / (2 * FD)
        context["jacobian"] = jacobian
    return context


def _energy(context, delta0, delta1, mode):
    shifts = (delta0, delta1)
    sampled = [_describe(rgb, points, delta)
               for rgb, points, delta in zip(context["rgb"], context["points"], shifts)]
    d0, d1 = sampled[0][0], sampled[1][0]
    n, valid, edges = context["n"], context["valid"], context["edges"]
    original0, original1 = context["original"]
    denominator = max(1, n)
    unary = math.fsum(math.fsum((a - b) ** 2 for a, b in zip(d0[i], d1[i]))
                      for i in valid) / denominator
    residual = _residuals(d0, d1, edges)
    moving = math.fsum(value * value for value in residual) / max(1, len(edges))
    frozen_terms = []
    for edge, (i, j) in enumerate(edges):
        left = _dot(d0[i], original0[j]) - _dot(d1[i], original1[j])
        right = _dot(d0[j], original0[i]) - _dot(d1[j], original1[i])
        frozen_terms.append(left * left + right * right - context["r0"][edge] ** 2)
    frozen = math.fsum(frozen_terms) / max(1, len(edges))
    mean = 0.0
    if valid:
        mean = math.fsum((math.fsum(d0[i][j] for i in valid) / len(valid)
                          - math.fsum(d1[i][j] for i in valid) / len(valid)) ** 2
                         for j in range(121))
    prior = math.fsum(delta[i][axis] ** 2 for delta in shifts for i in valid for axis in (0, 1)) / (2 * denominator)
    linearized = None
    if mode == "linearized":
        flat = [float(value) for delta in shifts for row in delta for value in row]
        linearized = math.fsum((r0 + _dot(jac, flat)) ** 2
                              for r0, jac in zip(context["r0"], context["jacobian"])) / max(1, len(edges))
    relation = {"joint": moving, "frozen": frozen, "independent": 0.0,
                "mean": mean, "linearized": linearized}[mode]
    terms = dict(unary=unary, moving=moving, frozen=frozen, mean=mean,
                 linearized=linearized, relational=relation, prior=prior)
    return dict(loss=unary + relation + 0.1 * prior, terms=terms,
                descriptors=(d0, d1), norms=(sampled[0][1], sampled[1][1]))


def _finite_gradient(context, delta0, delta1, mode):
    derivatives = []
    for side in (0, 1):
        matrix = []
        for row in range(context["n"]):
            pair = []
            for axis in (0, 1):
                losses = []
                for direction in (-1, 1):
                    shifts = [[[float(x) for x in r] for r in delta] for delta in (delta0, delta1)]
                    shifts[side][row][axis] += direction * FD
                    losses.append(_energy(context, *shifts, mode)["loss"])
                pair.append((losses[1] - losses[0]) / (2 * FD))
            matrix.append(pair)
        derivatives.append(matrix)
    return derivatives


def _texture(np, side):
    rgb = np.empty((3, 32, 32), dtype=np.float32)
    for c in range(3):
        for y in range(32):
            for x in range(32):
                if side == 0:
                    value = (2 + 0.0017 * x * y + 0.0003 * x * x
                             + (0.17 + 0.02 * c) * math.sin(0.31 * x + (0.43 + 0.03 * c) * y)
                             + 0.09 * math.cos((0.19 + 0.02 * c) * x - 0.23 * y))
                else:
                    value = (1.8 + 0.0011 * x * y + 0.0005 * y * y
                             + (0.19 + 0.01 * c) * math.sin(0.37 * x + (0.29 + 0.05 * c) * y + 0.4)
                             + 0.12 * math.cos(0.17 * x - (0.31 + 0.02 * c) * y + 0.2))
                rgb[c, y, x] = value
    return rgb


@pytest.fixture
def case():
    np = importlib.import_module("numpy")
    importlib.import_module("torch")
    production = importlib.import_module("moving_patch_relations_026")
    p0 = np.asarray([[7.25, 8.375], [11.4, 9.3], [17.2, 12.45],
                     [22.35, 19.2], [10.7, 22.4], [20.1, 23.6]], dtype=np.float32)
    p1 = np.asarray([[7.42, 8.11], [11.08, 9.62], [17.65, 12.10],
                     [21.94, 19.51], [11.05, 22.08], [19.74, 23.21]], dtype=np.float32)
    edges = np.asarray([(0, 1), (0, 3), (1, 2), (1, 4), (2, 5), (3, 4), (4, 5)], dtype=np.int64)
    rgb0, rgb1 = _texture(np, 0), _texture(np, 1)
    return dict(np=np, production=production, rgb0=rgb0, rgb1=rgb1,
                p0=p0, p1=p1, edges=edges)


def _inputs(case):
    return tuple(case[key] for key in ("rgb0", "rgb1", "p0", "p1", "edges"))


def _deltas(case, nonzero=False):
    np = case["np"]
    a = np.zeros((len(case["p0"]), 2), dtype=np.float64)
    b = a.copy()
    if nonzero:
        a[:] = [0.031, -0.017]; b[:] = [-0.023, 0.019]
    return a, b


@pytest.mark.parametrize("mode", MODES)
def test_scalar_bilinear_descriptor_energy_and_unscaled_finite_difference(case, mode):
    np, production = case["np"], case["production"]
    inputs = _inputs(case)
    context = _context(*inputs, linear=mode == "linearized")
    delta0, delta1 = _deltas(case, nonzero=True)
    expected = _energy(context, delta0.tolist(), delta1.tolist(), mode)
    actual = production.evaluate_pair(*inputs, delta0, delta1, mode=mode)
    assert actual["eligible_rows"] == list(range(6)) and len(actual["effective_edges"]) == 7
    assert actual["loss"] == pytest.approx(expected["loss"], abs=3e-9, rel=3e-9)
    for term, value in expected["terms"].items():
        if value is None:
            assert actual["terms"][term] is None
        else:
            assert actual["terms"][term] == pytest.approx(value, abs=3e-9, rel=3e-9)
    for side in (0, 1):
        descriptor = actual["descriptors" + str(side)]
        norm = actual["norm" + str(side)]
        assert descriptor.dtype == np.dtype("float64") and descriptor.shape == (6, 121)
        assert norm.dtype == np.dtype("float64") and norm.shape == (6,)
        np.testing.assert_allclose(descriptor, expected["descriptors"][side], atol=3e-9, rtol=3e-9)
        np.testing.assert_allclose(norm, expected["norms"][side], atol=3e-9, rtol=3e-9)
    fd = _finite_gradient(context, delta0.tolist(), delta1.tolist(), mode)
    norm_squared = 0.0
    for side in (0, 1):
        gradient = actual["gradient" + str(side)]
        assert gradient.dtype == np.dtype("float64") and gradient.shape == (6, 2)
        np.testing.assert_allclose(gradient, fd[side], atol=4e-7, rtol=4e-4)
        norm_squared += math.fsum(float(v) ** 2 for row in gradient for v in row)
    assert norm_squared > 1e-8  # Real gradient-bearing textures, not a zero-gradient ramp PASS.
    assert actual["gradient_scale"] == "dL/dDelta; UNMULTIPLIED"
    if mode == "linearized":
        reference = actual["linearized_reference"]
        assert reference["jacobian0"].shape == (7, 24)
        np.testing.assert_allclose(reference["residual0"], context["r0"], atol=3e-9, rtol=3e-9)
        np.testing.assert_allclose(reference["jacobian0"], context["jacobian"], atol=3e-7, rtol=3e-4)
    else:
        assert actual["linearized_reference"] is None


def test_scalar_sampler_at_pixel_centers_and_fractional_rgb_luma(case):
    np, production = case["np"], case["production"]
    # Descriptor comparison covers all121 ordered samples. This explicit scalar
    # check additionally witnesses RGB weights and bilinear array-index meaning.
    x, y = 7.25, 8.375
    rgb = case["rgb0"]
    manual = math.fsum(weight * (0.75 * 0.625 * float(rgb[c, 8, 7])
                                  + 0.25 * 0.625 * float(rgb[c, 8, 8])
                                  + 0.75 * 0.375 * float(rgb[c, 9, 7])
                                  + 0.25 * 0.375 * float(rgb[c, 9, 8]))
                       for c, weight in enumerate((0.299, 0.587, 0.114)))
    assert _luma(rgb, x, y) == pytest.approx(manual, abs=1e-14)
    zero = _deltas(case)
    actual = production.evaluate_pair(*_inputs(case), *zero, mode="joint")
    desc, norms = _describe(rgb, case["p0"], zero[0].tolist())
    np.testing.assert_allclose(actual["descriptors0"], desc, atol=3e-9, rtol=3e-9)
    np.testing.assert_allclose(actual["norm0"], norms, atol=3e-9, rtol=3e-9)


def test_joint_frozen_linearized_initial_match_first_step_then_real_divergence(case):
    np, production = case["np"], case["production"]
    results = {mode: production.refine_pair(*_inputs(case), mode=mode)
               for mode in ("joint", "frozen", "linearized")}
    initial = results["joint"]["diagnostics"]["initial"]
    for mode, result in results.items():
        diag = result["diagnostics"]
        assert diag["steps"] == 4 and len(diag["trace"]) == 4
        assert len(diag["initial"]["effective_edges"]) == 7
        assert diag["initial"]["loss"] == pytest.approx(initial["loss"], abs=1e-11)
        for side in (0, 1):
            key = "gradient" + str(side)
            np.testing.assert_allclose(diag["initial"][key], initial[key], atol=1e-10, rtol=1e-9)
            np.testing.assert_allclose(diag["trace"][0]["delta_after" + str(side)],
                                       results["joint"]["diagnostics"]["trace"][0]["delta_after" + str(side)],
                                       atol=1e-10, rtol=1e-9)
    for mode in ("frozen", "linearized"):
        distance = math.fsum(float(v) ** 2 for side in (0, 1)
                            for row in (results[mode]["diagnostics"]["final_delta" + str(side)]
                                        - results["joint"]["diagnostics"]["final_delta" + str(side)])
                            for v in row)
        assert distance > 1e-18  # Later operator coupling must not be a no-op.
    assert math.fsum(float(v) ** 2 for row in results["joint"]["diagnostics"]["final_delta0"] for v in row) > 1e-8


def test_rectangular_integer_fractional_ramp_axes_and_raw_norm(case):
    np, production = case["np"], case["production"]
    slopes = ((0.125, 0.25), (0.25, 0.0625), (0.0625, 0.125))
    images = []
    for h, w in ((24, 40), (28, 36)):
        rgb = np.empty((3, h, w), dtype=np.float32)
        for c, (ax, by) in enumerate(slopes):
            for y in range(h):
                for x in range(w):
                    rgb[c, y, x] = 2 + c + ax * x + by * y
        images.append(rgb)
    points = np.asarray([[10, 10], [20.25, 13.375]], dtype=np.float32)
    edges = np.asarray([[0, 1]], dtype=np.int64)
    zero = np.zeros((2, 2), dtype=np.float64)
    actual = production.evaluate_pair(images[0], images[1], points, points, edges, zero, zero, mode="joint")
    assert actual["eligible_rows"] == [0, 1] and actual["effective_edges"].tolist() == [[0, 1]]
    weights = (0.299, 0.587, 0.114)
    a = _dot(weights, [s[0] for s in slopes]); b = _dot(weights, [s[1] for s in slopes])
    closed_norm = math.sqrt(1210 * (a * a + b * b))
    for side, rgb in enumerate(images):
        descriptors, norms = _describe(rgb, points, zero.tolist())
        np.testing.assert_allclose(actual["descriptors" + str(side)], descriptors, atol=3e-9, rtol=3e-9)
        np.testing.assert_allclose(actual["norm" + str(side)], norms, atol=3e-9, rtol=3e-9)
        np.testing.assert_allclose(actual["norm" + str(side)], [closed_norm] * 2, atol=3e-9, rtol=3e-9)


def test_trace_uses_minus_quarter_times_N_gradient_trust_ball_and_wire_cast(case):
    np, production = case["np"], case["production"]
    inputs = _inputs(case)
    result = production.refine_pair(*inputs, mode="joint")
    n = len(case["p0"])
    valid = set(result["eligible_rows"])
    for step in result["diagnostics"]["trace"]:
        for side in (0, 1):
            before = step["delta_before" + str(side)]
            after = step["delta_after" + str(side)]
            gradient = step["evaluation"]["gradient" + str(side)]
            expected = []
            for row in range(n):
                trial = [float(before[row, axis]) - 0.25 * n * float(gradient[row, axis]) for axis in (0, 1)]
                radius = max(1.0, math.hypot(*trial))
                expected.append([x / radius for x in trial] if row in valid else [0.0, 0.0])
            np.testing.assert_allclose(after, expected, atol=3e-12, rtol=3e-12)
            assert all(math.hypot(*map(float, row)) <= 1 + 1e-12 for row in after)
    for side in (0, 1):
        delta = result["diagnostics"]["final_delta" + str(side)]
        base = case["p" + str(side)]
        wire = result["points" + str(side)]
        expected = (base.astype(np.float64) + delta).astype(np.float32)
        assert wire.dtype == np.dtype("float32")
        assert wire.tobytes() == expected.tobytes()
        # Trust is enforced before FP32 rounding; emitted distances allow one ULP.
        for row in valid:
            error = math.hypot(*(float(wire[row, axis]) - float(base[row, axis]) for axis in (0, 1)))
            rounding = math.hypot(*(abs(float(np.spacing(expected[row, axis]))) for axis in (0, 1)))
            assert error <= 1 + rounding
        np.testing.assert_allclose(result["diagnostics"]["trace"][-1]["delta_after" + str(side)], delta, atol=0, rtol=0)


@pytest.mark.parametrize("mode", MODES)
def test_view_exchange_and_fixed_edge_permutation_equivariance(case, mode):
    np, production = case["np"], case["production"]
    d0, d1 = _deltas(case, nonzero=True)
    original = production.evaluate_pair(*_inputs(case), d0, d1, mode=mode)
    swapped = production.evaluate_pair(case["rgb1"], case["rgb0"], case["p1"], case["p0"],
                                       case["edges"], d1, d0, mode=mode)
    assert swapped["loss"] == pytest.approx(original["loss"], abs=3e-11)
    np.testing.assert_allclose(swapped["gradient0"], original["gradient1"], atol=3e-10, rtol=3e-9)
    np.testing.assert_allclose(swapped["gradient1"], original["gradient0"], atol=3e-10, rtol=3e-9)
    permutation = np.asarray([4, 0, 5, 2, 1, 3], dtype=np.int64)
    inverse = {int(old): new for new, old in enumerate(permutation)}
    edges = np.asarray(sorted(tuple(sorted((inverse[int(i)], inverse[int(j)])))
                              for i, j in case["edges"]), dtype=np.int64)
    permuted = production.evaluate_pair(case["rgb0"], case["rgb1"], case["p0"][permutation],
                                        case["p1"][permutation], edges, d0[permutation], d1[permutation], mode=mode)
    assert permuted["loss"] == pytest.approx(original["loss"], abs=3e-11)
    for side in (0, 1):
        np.testing.assert_allclose(permuted["gradient" + str(side)], original["gradient" + str(side)][permutation],
                                   atol=3e-10, rtol=3e-9)


def test_mean_is_literal_valid_descriptor_mean_and_not_scalar_matched_gradient(case):
    production = case["production"]
    d0, d1 = _deltas(case, nonzero=True)
    values = {mode: production.evaluate_pair(*_inputs(case), d0, d1, mode=mode)
              for mode in ("joint", "independent", "mean")}
    mean = values["mean"]
    rows = mean["eligible_rows"]
    literal = math.fsum((math.fsum(float(mean["descriptors0"][i, j]) for i in rows) / len(rows)
                         - math.fsum(float(mean["descriptors1"][i, j]) for i in rows) / len(rows)) ** 2
                        for j in range(121))
    assert mean["terms"]["mean"] == pytest.approx(literal, abs=3e-12)
    assert mean["terms"]["relational"] == mean["terms"]["mean"]
    vectors = []
    for mode in ("joint", "mean"):
        vectors.append([float(values[mode]["gradient" + str(s)][i, j]
                              - values["independent"]["gradient" + str(s)][i, j])
                        for s in (0, 1) for i in rows for j in (0, 1)])
    a, b = vectors
    assert _dot(a, a) > 1e-12 and _dot(b, b) > 1e-12
    alpha = _dot(a, b) / _dot(a, a)
    assert math.sqrt(math.fsum((y - alpha * x) ** 2 for x, y in zip(a, b))) > 1e-5 * math.sqrt(_dot(b, b))


def test_relational_residual_identity_bound_and_identical_descriptor_null(case):
    np, production = case["np"], case["production"]
    d0, d1 = _deltas(case, nonzero=True)
    result = production.evaluate_pair(*_inputs(case), d0, d1, mode="joint")
    a, b = result["descriptors0"].tolist(), result["descriptors1"].tolist()
    e = [[x - y for x, y in zip(u, v)] for u, v in zip(a, b)]
    s = [[x + y for x, y in zip(u, v)] for u, v in zip(a, b)]
    for i, j in result["effective_edges"]:
        residual = _dot(a[i], a[j]) - _dot(b[i], b[j])
        assert residual == pytest.approx(0.5 * (_dot(e[i], s[j]) + _dot(s[i], e[j])), abs=3e-13)
        assert residual * residual <= 2 * (_dot(e[i], e[i]) + _dot(e[j], e[j])) + 1e-12
    zeros = np.zeros_like(d0)
    null = production.evaluate_pair(case["rgb0"], case["rgb0"], case["p0"], case["p0"],
                                    case["edges"], zeros, zeros, mode="joint")
    assert null["terms"]["unary"] == pytest.approx(0, abs=1e-25)
    assert null["terms"]["moving"] == pytest.approx(0, abs=1e-25)
    unchanged = production.refine_pair(case["rgb0"], case["rgb0"], case["p0"], case["p0"], case["edges"], mode="joint")
    assert unchanged["points0"].tobytes() == case["p0"].tobytes()
    assert unchanged["points1"].tobytes() == case["p0"].tobytes()


@pytest.mark.parametrize("kind", ["empty", "all_flat", "all_invalid"])
def test_whole_population_empty_flat_invalid_preserves_signed_zero_bytes(case, kind):
    np, production = case["np"], case["production"]
    rgb0, rgb1 = case["rgb0"].copy(), case["rgb1"].copy()
    if kind == "empty":
        p0 = np.empty((0, 2), dtype=np.float32)
    else:
        p0 = np.asarray([[-0.0, 8.25], [5.75, 12.125], [26.25, 20.5]], dtype=np.float32)
    if kind == "all_flat":
        rgb0.fill(2); rgb1.fill(2)
        p0[1:] = [[10.25, 12.125], [20.25, 20.5]]
    p1 = p0.copy()
    edges = np.asarray([(0, 1), (1, 2)] if len(p0) else [], dtype=np.int64).reshape((-1, 2))
    before = (p0.tobytes(), p1.tobytes())
    result = production.refine_pair(rgb0, rgb1, p0, p1, edges, mode="joint")
    assert result["eligible_rows"] == []
    assert result["points0"].shape == p0.shape and result["points1"].shape == p1.shape
    assert (result["points0"].tobytes(), result["points1"].tobytes()) == before
    assert result["diagnostics"]["final"]["loss"] == pytest.approx(0, abs=1e-25)


def test_mixed_invalid_rows_stay_in_denominator_and_wire_and_inputs_immutable(case):
    np, production = case["np"], case["production"]
    invalid = np.asarray([[-0.0, 10.25], [26.25, 12.5]], dtype=np.float32)
    p0 = np.concatenate((case["p0"], invalid)); p1 = np.concatenate((case["p1"], invalid))
    edges = np.asarray(sorted(list(map(tuple, case["edges"].tolist())) + [(0, 6), (5, 7), (6, 7)]), dtype=np.int64)
    values = (case["rgb0"].copy(), case["rgb1"].copy(), p0, p1, edges)
    before = [value.tobytes() for value in values]
    for value in values:
        value.flags.writeable = False
    delta = np.zeros((8, 2), dtype=np.float64)
    evaluated = production.evaluate_pair(*values, delta, delta, mode="joint")
    context = _context(*values)
    expected = _energy(context, delta.tolist(), delta.tolist(), "joint")
    assert evaluated["loss"] == pytest.approx(expected["loss"], abs=3e-9)
    assert evaluated["normalization"]["population"] == 8 and evaluated["eligible_rows"] == list(range(6))
    assert evaluated["effective_edges"].tolist() == case["edges"].tolist()
    result = production.refine_pair(*values, mode="joint")
    assert result["points0"][6:].tobytes() == invalid.tobytes()
    assert result["points1"][6:].tobytes() == invalid.tobytes()
    for entry in result["diagnostics"]["trace"]:
        assert entry["delta_after0"][6:].tobytes() == np.zeros((2, 2), dtype=np.float64).tobytes()
        assert entry["delta_after1"][6:].tobytes() == np.zeros((2, 2), dtype=np.float64).tobytes()
    assert [value.tobytes() for value in values] == before


def test_feasible_later_norm_collapse_does_not_refilter_and_records_actual_path(case, record_property):
    np, production = case["np"], case["production"]
    rgb0 = np.full((3, 32, 32), 2, dtype=np.float32); rgb1 = rgb0.copy()
    rgb0[:, :, 5] = 3; rgb1[:, 5, :] = 3
    p = np.asarray([[10.25, 10.25], [10.5, 10.5]], dtype=np.float32)
    edges = np.asarray([[0, 1]], dtype=np.int64)
    zero = np.zeros((2, 2), dtype=np.float64)
    shifted0 = zero.copy(); shifted1 = zero.copy()
    shifted0[:, 0] = 0.875; shifted1[:, 1] = 0.875
    initial = production.evaluate_pair(rgb0, rgb1, p, p, edges, zero, zero, mode="joint")
    flat = production.evaluate_pair(rgb0, rgb1, p, p, edges, shifted0, shifted1, mode="joint")
    assert initial["eligible_rows"] == flat["eligible_rows"] == [0, 1]
    assert initial["collapse_rows"] == [] and flat["collapse_rows"] == [0, 1]
    assert flat["effective_edges"].tolist() == [[0, 1]]
    assert all(float(x) <= EPS for name in ("norm0", "norm1") for x in flat[name])
    assert flat["loss"] < initial["loss"]
    # Do not claim the four-step optimizer reaches that feasible collapse.
    actual = production.refine_pair(rgb0, rgb1, p, p, edges, mode="joint")
    diag = actual["diagnostics"]
    assert np.isfinite(actual["points0"]).all() and np.isfinite(actual["points1"]).all()
    samples = [diag["initial"], diag["final"]] + [row["evaluation"] for row in diag["trace"]]
    record_property("collapse_witness_initial_loss", initial["loss"])
    record_property("collapse_witness_feasible_flat_loss", flat["loss"])
    record_property("collapse_witness_actual_final_loss", diag["final"]["loss"])
    record_property("collapse_witness_actual_min_norm", min(float(x) for sample in samples for name in ("norm0", "norm1") for x in sample[name]))
    record_property("collapse_witness_actual_final_collapsed_rows", len(diag["final"]["collapse_rows"]))


def test_good_identity_row_can_receive_force_from_bad_neighbor(case):
    np, production = case["np"], case["production"]
    p0 = case["p0"].copy(); p1 = p0.copy()
    p1[1] += np.asarray([1.25, -0.5], dtype=np.float32)
    edges = np.asarray([[0, 1]], dtype=np.int64)
    zero = np.zeros((6, 2), dtype=np.float64)
    independent = production.evaluate_pair(case["rgb0"], case["rgb0"], p0, p1, edges, zero, zero, mode="independent")
    joint = production.evaluate_pair(case["rgb0"], case["rgb0"], p0, p1, edges, zero, zero, mode="joint")
    np.testing.assert_allclose(joint["descriptors0"][0], joint["descriptors1"][0], atol=0, rtol=0)
    assert all(float(x) == 0 for name in ("gradient0", "gradient1") for x in independent[name][0])
    assert math.fsum(float(x) ** 2 for name in ("gradient0", "gradient1") for x in joint[name][0]) > 1e-12


@pytest.mark.parametrize("bad", ["rgb_negative", "rgb_nan", "rgb64", "point_nan", "point64", "delta_nan", "delta_radius", "edge_float", "edge_duplicate", "mode"])
def test_invalid_input_rejection_does_not_silently_return_predictions(case, bad):
    np, production = case["np"], case["production"]
    rgb0, rgb1, p0, p1, edges = [value.copy() for value in _inputs(case)]
    d0, d1 = _deltas(case)
    mode = "joint"
    if bad == "rgb_negative": rgb0[0, 0, 0] = -1
    elif bad == "rgb_nan": rgb0[0, 0, 0] = np.nan
    elif bad == "rgb64": rgb0 = rgb0.astype(np.float64)
    elif bad == "point_nan": p0[0, 0] = np.nan
    elif bad == "point64": p0 = p0.astype(np.float64)
    elif bad == "delta_nan": d0[0, 0] = np.nan
    elif bad == "delta_radius": d0[0, 0] = 2
    elif bad == "edge_float": edges = edges.astype(np.float64)
    elif bad == "edge_duplicate": edges = np.concatenate((edges, edges[:1]))
    elif bad == "mode": mode = "unknown"
    with pytest.raises(ValueError):
        production.evaluate_pair(rgb0, rgb1, p0, p1, edges, d0, d1, mode=mode)


def _graph_oracle(p0, p1, hw0, hw1, ids):
    n = len(p0)
    positions = [[float(p0[i, 0]) / hw0[1], float(p0[i, 1]) / hw0[0],
                  float(p1[i, 0]) / hw1[1], float(p1[i, 1]) / hw1[0]] for i in range(n)]
    result = set()
    for i in range(n):
        choices = [(math.fsum((a - b) ** 2 for a, b in zip(positions[i], positions[j])), int(ids[j]), j)
                   for j in range(n) if j != i]
        for _, _, j in sorted(choices)[:4]:
            result.add(tuple(sorted((i, j))))
    return sorted(result)


def _degrees(edges, n, eligible=None):
    result = [0] * n
    active = None if eligible is None else set(eligible)
    for i, j in edges:
        i, j = int(i), int(j)
        if active is None or i in active and j in active:
            result[i] += 1; result[j] += 1
    return result


def _lengths(edges, p0, p1, hw0, hw1):
    lengths = []
    for i, j in edges:
        components = [((float(p0[i, 0]) - float(p0[j, 0])) / hw0[1]) ** 2,
                      ((float(p0[i, 1]) - float(p0[j, 1])) / hw0[0]) ** 2,
                      ((float(p1[i, 0]) - float(p1[j, 0])) / hw1[1]) ** 2,
                      ((float(p1[i, 1]) - float(p1[j, 1])) / hw1[0]) ** 2]
        lengths.append(math.sqrt(math.fsum(components)))
    return sorted(lengths)


def test_graph_ties_use_explicit_array_indices_or_provided_stable_ids(case):
    np, production = case["np"], case["production"]
    hw0, hw1 = (24, 40), (28, 36)
    assert production.spatial_edges(case["p0"], case["p1"], hw0, hw1).tolist() == list(map(list,
        _graph_oracle(case["p0"], case["p1"], hw0, hw1, range(6))))
    # Identical positions deliberately create exact ties; no physical claim.
    p0 = np.full((7, 2), 10.25, dtype=np.float32); p1 = p0.copy()
    ids = np.asarray([70, 10, 60, 20, 50, 30, 40], dtype=np.int64)
    default = production.spatial_edges(p0, p1, (32, 32), (32, 32))
    stable = production.spatial_edges(p0, p1, (32, 32), (32, 32), row_ids=ids)
    assert default.tolist() == list(map(list, _graph_oracle(p0, p1, (32, 32), (32, 32), range(7))))
    assert stable.tolist() == list(map(list, _graph_oracle(p0, p1, (32, 32), (32, 32), ids)))
    permutation = np.asarray([5, 0, 6, 2, 1, 4, 3], dtype=np.int64)
    reordered = production.spatial_edges(p0[permutation], p1[permutation], (32, 32), (32, 32), row_ids=ids[permutation])
    pairs = lambda graph, labels: {tuple(sorted((int(labels[i]), int(labels[j])))) for i, j in graph}
    assert pairs(stable, ids) == pairs(reordered, ids[permutation])
    for bad in (ids.astype(np.float64), np.asarray([0] * 7, dtype=np.int64), -ids):
        with pytest.raises(ValueError):
            production.spatial_edges(p0, p1, (32, 32), (32, 32), row_ids=bad)


def test_shuffle_real_switches_preserve_full_and_effective_degrees_and_report_lengths(case):
    np, production = case["np"], case["production"]
    hw0, hw1 = (24, 40), (28, 36)
    p0 = np.asarray([[7 + 2 * i, 9 + i] for i in range(8)], dtype=np.float32)
    p1 = np.asarray([[8 + i, 10 + 1.5 * i] for i in range(8)], dtype=np.float32)
    edges = np.asarray([(0, 1), (0, 5), (0, 6), (1, 2), (2, 3), (3, 4), (4, 5), (6, 7)], dtype=np.int64)
    active = list(range(6))
    before = [value.tobytes() for value in (edges, p0, p1)]
    result = production.shuffle_edges(edges, p0, p1, hw0, hw1, seed=42, attempts=256, eligible_rows=active)
    graph, record = result["edges"], result["record"]
    assert graph.dtype == np.dtype("int64") and graph.tolist() == sorted(map(list, set(map(tuple, graph.tolist()))))
    assert record["valid_switches"] > 0 and record["informative"]
    assert set(map(tuple, graph.tolist())) != set(map(tuple, edges.tolist()))
    assert (0, 6) in set(map(tuple, graph.tolist())) and (6, 7) in set(map(tuple, graph.tolist()))
    assert _degrees(edges, 8) == _degrees(graph, 8) == record["degree_before"] == record["degree_after"]
    assert _degrees(edges, 8, active) == _degrees(graph, 8, active) == record["effective_degree_before"] == record["effective_degree_after"]
    assert record["normalized_lengths_before"] == pytest.approx(_lengths(edges, p0, p1, hw0, hw1), abs=1e-14)
    assert record["normalized_lengths_after"] == pytest.approx(_lengths(graph, p0, p1, hw0, hw1), abs=1e-14)
    reconstructed = set(map(tuple, edges.tolist()))
    original = set(reconstructed)
    for switch in record["switches"]:
        removed, added = set(map(tuple, switch["removed"])), set(map(tuple, switch["added"]))
        assert len(removed) == len(added) == 2 and removed <= reconstructed
        assert len({i for edge in removed for i in edge}) == 4
        assert _degrees(removed, 8) == _degrees(added, 8)
        reconstructed = (reconstructed - removed) | added
        assert reconstructed != original
    assert reconstructed == set(map(tuple, graph.tolist()))
    assert [value.tobytes() for value in (edges, p0, p1)] == before


def test_empty_complete_and_zero_attempt_shuffles_are_explicitly_uninformative(case):
    np, production = case["np"], case["production"]
    points = case["p0"][:4].copy()
    complete = np.asarray([(i, j) for i in range(4) for j in range(i + 1, 4)], dtype=np.int64)
    empty = np.empty((0, 2), dtype=np.int64)
    for edges, attempts in ((empty, 32), (complete, 32), (complete, 0)):
        result = production.shuffle_edges(edges, points, points, (32, 32), (32, 32), attempts=attempts)
        assert result["edges"].tobytes() == edges.tobytes()
        assert result["record"]["informative"] is False and result["record"]["valid_switches"] == 0
        assert result["record"]["reason"] == "UNINFORMATIVE_NO_CHANGED_GRAPH"
