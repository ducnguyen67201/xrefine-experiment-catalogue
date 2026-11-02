"""Independent image/energy checks; numerical work starts only in test functions."""
import importlib.util
import itertools
import math
from pathlib import Path
import sys

import numpy as np
import pytest


@pytest.fixture(scope="module")
def api():
    name = "rm032_bridge_under_test"
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name("bridge_context.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _gray_pixel(rgb, y, x):
    return .299 * float(rgb[0, y, x]) + .587 * float(rgb[1, y, x]) + .114 * float(rgb[2, y, x])


def _bilinear(rgb, x, y):
    ix, iy = math.floor(x), math.floor(y)
    tx, ty = x - ix, y - iy
    return math.fsum((
        (1 - tx) * (1 - ty) * _gray_pixel(rgb, iy, ix),
        tx * (1 - ty) * _gray_pixel(rgb, iy, ix + 1),
        (1 - tx) * ty * _gray_pixel(rgb, iy + 1, ix),
        tx * ty * _gray_pixel(rgb, iy + 1, ix + 1),
    ))


def _patch(rgb, point, radius):
    return [_bilinear(rgb, float(point[0]) + x, float(point[1]) + y)
            for y in range(-radius, radius + 1) for x in range(-radius, radius + 1)]


def _unit(values):
    mean = math.fsum(values) / len(values)
    centered = [v - mean for v in values]
    norm = math.sqrt(math.fsum(v * v for v in centered))
    return [v / max(norm, 1e-6) for v in centered], norm


def _texture(height=80, width=128, phase=0.):
    # Nonlinear, positive RGB with different channels and values above one.
    yy, xx = np.mgrid[:height, :width]
    channels = []
    for c in range(3):
        channels.append(2. + .001 * (c + 1) * xx * yy + .004 * xx * xx
                        + .006 * yy * yy + .4 * np.sin(.37 * xx + .23 * yy + c + phase)
                        + .25 * np.cos(.19 * xx - .31 * yy + c * .7 - phase))
    return np.asarray(channels, dtype=np.float32)


def _snap(*arrays):
    return [(a.dtype.str, a.shape, a.strides, a.tobytes()) for a in arrays]


def _decomposition_oracle(unary, edges, factors):
    n, k = unary.shape
    alpha = n / len(edges) if len(edges) else 0.
    augmented = [[float(v) for v in row] for row in unary]
    residuals, constant = [], 0.
    for (i, j), table in zip(edges, factors):
        rows = [math.fsum(float(v) for v in row) / k for row in table]
        cols = [math.fsum(float(table[a, b]) for a in range(k)) / k for b in range(k)]
        grand = math.fsum(rows) / k
        residuals.append([[float(table[a, b]) - rows[a] - cols[b] + grand
                           for b in range(k)] for a in range(k)])
        for a in range(k):
            augmented[int(i)][a] += alpha * rows[a]
            augmented[int(j)][a] += alpha * cols[a]
        constant -= alpha * grand
    return np.asarray(augmented), np.asarray(residuals).reshape((-1, k, k)), constant


def _energy(unary, edges, factors, labels):
    n = len(unary)
    return (math.fsum(float(unary[i, labels[i]]) for i in range(n))
            + (n / len(edges) if len(edges) else 0.)
            * math.fsum(float(w[labels[int(i)], labels[int(j)]])
                        for (i, j), w in zip(edges, factors)))


def _one_pass(unary, edges, factors, valid, mode):
    augmented, residual, _ = _decomposition_oracle(unary, edges, factors)
    n, k = unary.shape
    probabilities = []
    for row, mask in zip(augmented, valid):
        minimum = min(float(row[a]) for a in range(k) if mask[a])
        weights = [math.exp(-(float(row[a]) - minimum)) if mask[a] else 0. for a in range(k)]
        total = math.fsum(weights)
        probabilities.append([v / total for v in weights])
    labels = [min((a for a in range(k) if valid[i, a]), key=lambda a: (augmented[i, a], a))
              for i in range(n)]
    result = augmented.copy()
    alpha = n / len(edges) if len(edges) else 0.
    for (i, j), table in zip(edges, residual):
        i, j = int(i), int(j)
        for a in range(k):
            if mode == "mean":
                result[i, a] += alpha * math.fsum(float(table[a, b]) * probabilities[j][b] for b in range(k))
                result[j, a] += alpha * math.fsum(float(table[b, a]) * probabilities[i][b] for b in range(k))
            else:
                result[i, a] += alpha * table[a, labels[j]]
                result[j, a] += alpha * table[labels[i], a]
    for i in range(n):
        result[i, ~valid[i]] = np.inf
    return result


def test_state_grid_is_norm_then_lex_and_zero_first(api):
    states = api.state_grid()
    expected = sorted(itertools.product((-1., 0., 1.), repeat=4),
                      key=lambda u: (math.fsum(x * x for x in u), u))
    assert states.dtype == np.float64 and states.shape == (81, 4)
    np.testing.assert_array_equal(states, np.asarray(expected))
    assert tuple(states[0]) == (0., 0., 0., 0.)


def test_decomposition_preserves_actual_assignment_energy(api):
    unary = np.asarray([[.1, .7, .3], [.8, .2, .4], [.5, .1, .9]], dtype=np.float64)
    edges = np.asarray([[0, 1], [1, 2]], dtype=np.int64)
    factors = np.asarray([[[1., 3., 2.], [4., 2., 7.], [6., 5., 1.]],
                          [[3., 1., 4.], [5., 9., 2.], [6., 5., 3.]]], dtype=np.float64)
    saved = _snap(unary, edges, factors)
    augmented, residual, constant = api.decompose(unary, edges, factors)
    a, r, c = _decomposition_oracle(unary, edges, factors)
    np.testing.assert_allclose(augmented, a, rtol=0, atol=2e-14)
    np.testing.assert_allclose(residual, r, rtol=0, atol=2e-14)
    assert constant == pytest.approx(c, abs=2e-14)
    np.testing.assert_allclose(residual.mean(axis=1), 0., atol=2e-14)
    np.testing.assert_allclose(residual.mean(axis=2), 0., atol=2e-14)
    for labels in itertools.product(range(3), repeat=3):
        represented = (math.fsum(float(augmented[i, labels[i]]) for i in range(3))
                       + 1.5 * math.fsum(float(w[labels[int(i)], labels[int(j)]])
                                         for (i, j), w in zip(edges, residual)) + constant)
        assert represented == pytest.approx(_energy(unary, edges, factors, labels), abs=5e-14)
    assert _snap(unary, edges, factors) == saved


@pytest.mark.parametrize("mode", ["independent", "mean", "map", "bp", "shuffled"])
def test_separable_factor_has_no_residual_interaction(api, mode):
    unary = np.asarray([[0., .1, .3], [.2, 0., .4], [.5, .2, 0.]], dtype=np.float64)
    edges = np.asarray([[0, 1], [1, 2]], dtype=np.int64)
    factors = np.asarray([[[a + b + 2. for b in (1., 0., .4)] for a in (.3, .1, 0.)],
                          [[a + b + 1. for b in (.6, .2, 0.)] for a in (.1, 0., .4)]], dtype=np.float64)
    a, r, _ = _decomposition_oracle(unary, edges, factors)
    expected = np.argmin(a, axis=1)
    result = api.infer(unary, edges, factors, mode=mode)
    np.testing.assert_array_equal(result["choice"], expected)
    np.testing.assert_allclose(result["centered_factors"], r, rtol=0, atol=2e-14)
    np.testing.assert_allclose(result["centered_factors"], 0., atol=2e-14)


def test_four_bp_steps_recover_unique_tree_map_and_min_marginals(api):
    # Four-node chain, three states: terminal evidence must travel three edges.
    unary = np.asarray([[0., 1., 7.], [0., 1., 7.], [0., 1., 7.], [10., 0., 7.]], dtype=np.float64)
    edges = np.asarray([[0, 1], [1, 2], [2, 3]], dtype=np.int64)
    factors = np.asarray([[[0. if a == b else 3. for b in range(3)] for a in range(3)]] * 3,
                         dtype=np.float64)
    energies = [(labels, _energy(unary, edges, factors, labels))
                for labels in itertools.product(range(3), repeat=4)]
    winner = min(energies, key=lambda item: item[1])
    assert winner == ((1, 1, 1, 1), 3.)
    assert sum(abs(value - winner[1]) < 1e-12 for _, value in energies) == 1
    result = api.infer(unary, edges, factors, mode="bp")
    np.testing.assert_array_equal(result["choice"], winner[0])
    for i in range(4):
        exact = [min(value for labels, value in energies if labels[i] == a) for a in range(3)]
        exact = np.asarray(exact) - exact[0]
        actual = result["belief"][i] - result["belief"][i, 0]
        np.testing.assert_allclose(actual, exact, rtol=0, atol=3e-13)
    np.testing.assert_array_equal(api.infer(unary, edges, factors, mode="local")["choice"], [0, 0, 0, 1])


@pytest.mark.parametrize("mode", ["mean", "map"])
def test_one_pass_controls_have_independent_expected_beliefs(api, mode):
    unary = np.asarray([[.2, .7, 0.], [.4, .1, .9], [.5, .2, .6]], dtype=np.float64)
    edges = np.asarray([[0, 1], [1, 2]], dtype=np.int64)
    factors = np.asarray([[[.2, .8, .4], [.3, .1, .6], [.9, .2, .7]],
                          [[.3, .2, .7], [.9, .1, .4], [.5, .8, .2]]], dtype=np.float64)
    valid = np.asarray([[True, True, False], [True, True, True], [False, True, True]])
    expected = _one_pass(unary, edges, factors, valid, mode)
    result = api.infer(unary, edges, factors, valid, mode=mode)
    np.testing.assert_array_equal(result["choice"], np.argmin(expected, axis=1))
    for i in range(3):
        active = np.flatnonzero(valid[i])
        np.testing.assert_allclose(result["belief"][i, active] - result["belief"][i, active[0]],
                                   expected[i, active] - expected[i, active[0]], rtol=0, atol=2e-14)
    assert np.all(np.isinf(result["belief"][~valid]))


def test_shuffle_changes_only_residual_state_associations(api):
    unary = np.asarray([[.01 * (a - c) ** 2 for a in range(81)] for c in (7, 21, 49)], dtype=np.float64)
    edges = np.asarray([[0, 1], [1, 2]], dtype=np.int64)
    factors = np.asarray([[[5. + .003 * (e + 1) * a * b + .2 * (a % 7) - .1 * (b % 5)
                            for b in range(81)] for a in range(81)] for e in range(2)], dtype=np.float64)
    saved = _snap(unary, edges, factors)
    ordinary = api.infer(unary, edges, factors, mode="bp")
    shuffled = api.infer(unary, edges, factors, mode="shuffled")
    repeated = api.infer(unary, edges, factors, mode="shuffled")
    np.testing.assert_array_equal(shuffled["augmented_unary"], ordinary["augmented_unary"])
    np.testing.assert_array_equal(shuffled["centered_factors"], ordinary["centered_factors"])
    np.testing.assert_array_equal(shuffled["used_centered_factors"], repeated["used_centered_factors"])
    np.testing.assert_array_equal(shuffled["choice"], repeated["choice"])
    original_r, shuffled_r = ordinary["centered_factors"], shuffled["used_centered_factors"]
    assert not np.array_equal(original_r, shuffled_r)
    for old, new in zip(original_r, shuffled_r):
        np.testing.assert_array_equal(np.sort(old.ravel()), np.sort(new.ravel()))
        np.testing.assert_allclose(new.mean(axis=0), 0., atol=2e-14)
        np.testing.assert_allclose(new.mean(axis=1), 0., atol=2e-14)
    movements = np.sum(api.state_grid() ** 2, axis=1)
    for e, record in enumerate(shuffled["shuffle_permutations"]):
        first, second = np.asarray(record["first"]), np.asarray(record["second"])
        assert first[0] == second[0] == 0
        np.testing.assert_array_equal(np.sort(first), np.arange(81))
        np.testing.assert_array_equal(np.sort(second), np.arange(81))
        np.testing.assert_array_equal(movements[first], movements)
        np.testing.assert_array_equal(movements[second], movements)
        np.testing.assert_array_equal(shuffled_r[e], original_r[e][np.ix_(first, second)])
    assert _snap(unary, edges, factors) == saved


@pytest.mark.parametrize("mode", ["local", "independent", "mean", "map", "bp", "shuffled"])
def test_no_edges_and_masked_states_are_exact_independent_limits(api, mode):
    unary = np.asarray([[1., 0., 2.], [3., 2., 1.]], dtype=np.float64)
    edges = np.empty((0, 2), dtype=np.int64)
    factors = np.empty((0, 3, 3), dtype=np.float64)
    valid = np.asarray([[True, False, False], [True, True, True]])
    result = api.infer(unary, edges, factors, valid, mode=mode)
    np.testing.assert_array_equal(result["choice"], [0, 2])
    assert result["belief"].shape == (2, 3)
    assert result["centered_factors"].shape == (0, 3, 3)
    assert np.isinf(result["belief"][0, 1:]).all()


def _footprint(point, radius):
    pixels = set()
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            x, y = math.floor(float(point[0]) + dx), math.floor(float(point[1]) + dy)
            pixels.update(((y, x), (y, x + 1), (y + 1, x), (y + 1, x + 1)))
    return pixels


def _all_unary_support(points, states, side):
    support = set()
    for p in points:
        for state in states:
            support.update(_footprint((float(p[0]) + state[side * 2],
                                       float(p[1]) + state[side * 2 + 1]), 5))
    return support


def _unary_oracle(rgb0, rgb1, p0, p1, state):
    a, na = _unit(_patch(rgb0, (float(p0[0]) + state[0], float(p0[1]) + state[1]), 5))
    b, nb = _unit(_patch(rgb1, (float(p1[0]) + state[2], float(p1[1]) + state[3]), 5))
    appearance = math.fsum((x - y) ** 2 for x, y in zip(a, b)) if min(na, nb) > 1e-6 else 4.
    return appearance + .05 * math.fsum(float(x) ** 2 for x in state)


def _bridge_values(rgb, points, i, j, si, sj, side):
    a = [float(points[i, d]) + float(si[2 * side + d]) for d in range(2)]
    b = [float(points[j, d]) + float(sj[2 * side + d]) for d in range(2)]
    result = []
    for t in (.25, .5, .75):
        result.extend(_patch(rgb, ((1 - t) * a[0] + t * b[0], (1 - t) * a[1] + t * b[1]), 1))
    return result


def test_actual_rgb_sampler_unary_and_whole_bridge_vector_against_scalar_oracle(api):
    rgb0, rgb1 = _texture(80, 128), _texture(96, 160, phase=.41)
    p0 = np.asarray([[24.25, 32.25], [88.25, 32.25]], dtype=np.float32)
    p1 = np.asarray([[24.5, 33.25], [88.5, 33.25]], dtype=np.float32)
    ev = api.build_evidence(rgb0, rgb1, p0, p1)
    states = api.state_grid()
    assert ev["unary"].dtype == ev["factors"].dtype == np.float64
    np.testing.assert_array_equal(ev["edges"], [[0, 1]])
    np.testing.assert_array_equal(ev["valid"], np.ones((2, 81), dtype=bool))
    for side, points in enumerate((p0, p1)):
        expected = [[[float(p[d]) + float(state[2 * side + d]) for d in range(2)]
                     for state in states] for p in points]
        np.testing.assert_array_equal(ev[f"positions{side}"], expected)
    for row in range(2):
        for a in (0, 1, 7, 17, 40, 80):
            expected = _unary_oracle(rgb0, rgb1, p0[row], p1[row], states[a])
            assert ev["unary"][row, a] == pytest.approx(expected, rel=2e-11, abs=2e-12)
    zero = (0., 0., 0., 0.)
    base0 = _bridge_values(rgb0, p0, 0, 1, zero, zero, 0)
    base1 = _bridge_values(rgb1, p1, 0, 1, zero, zero, 1)
    pooled_mean = math.fsum(base0 + base1) / 54
    pooled_scale = math.sqrt(math.fsum((v - pooled_mean) ** 2 for v in base0 + base1) / 2)
    assert pooled_scale > 1e-6
    assert ev["diagnostics"]["bridge_mu"][0] == pytest.approx(pooled_mean, rel=0, abs=2e-12)
    assert ev["diagnostics"]["bridge_scale"][0] == pytest.approx(pooled_scale, rel=2e-12)
    for a, b in ((0, 0), (0, 80), (17, 40), (80, 0), (80, 80), (1, 2)):
        values0 = _bridge_values(rgb0, p0, 0, 1, states[a], states[b], 0)
        values1 = _bridge_values(rgb1, p1, 0, 1, states[a], states[b], 1)
        expected = min(4., math.fsum(((x - y) / pooled_scale) ** 2 for x, y in zip(values0, values1)))
        assert ev["factors"][0, a, b] == pytest.approx(expected, rel=2e-11, abs=2e-12)


def test_actual_edges_query_bridge_support_outside_every_unary_footprint(api):
    rgb0, rgb1 = _texture(), _texture(phase=.31)
    p0 = np.asarray([[24.25, 32.25], [88.25, 32.25]], dtype=np.float32)
    p1 = np.asarray([[24.5, 33.25], [88.5, 33.25]], dtype=np.float32)
    ev = api.build_evidence(rgb0, rgb1, p0, p1)
    states = api.state_grid()
    np.testing.assert_array_equal(ev["edges"], [[0, 1]])
    assert ev["factors"].shape == (1, 81, 81)
    for side, points in enumerate((p0, p1)):
        forbidden = _all_unary_support(points, states, side)
        for (i, j) in ev["edges"]:
            for si, sj in itertools.product(states, repeat=2):
                for t in (.25, .5, .75):
                    qi = points[i].astype(np.float64) + si[side * 2:side * 2 + 2]
                    qj = points[j].astype(np.float64) + sj[side * 2:side * 2 + 2]
                    bridge = _footprint((1 - t) * qi + t * qj, 1)
                    assert bridge.isdisjoint(forbidden)


def test_third_row_support_and_distance_cap_remove_edges_before_appearance(api):
    rgb0, rgb1 = _texture(80, 160), _texture(80, 160, phase=.2)
    points = np.asarray([[24., 32.], [88., 32.], [56., 32.]], dtype=np.float32)
    blocked = api.build_evidence(rgb0, rgb1, points, points)
    assert [0, 1] not in blocked["edges"].tolist()  # Midpoint hits row 2's unary support.
    far = np.asarray([[24., 32.], [89., 32.]], dtype=np.float32)
    cap = api.build_evidence(rgb0, rgb1, far, far)
    assert cap["edges"].shape == (0, 2)
    assert blocked["diagnostics"]["graph_before_appearance"] is True


def test_mixed_flat_row_keeps_index_and_zeros_its_incident_factors(api):
    rgb0, rgb1 = _texture(96, 128), _texture(96, 128, phase=.2)
    points = np.asarray([[24., 24.], [88., 24.], [24., 64.]], dtype=np.float32)
    for rgb in (rgb0, rgb1):
        rgb[:, 17:32, 17:32] = 5.
    before = _snap(rgb0, rgb1, points)
    result = api.refine_pair(rgb0, rgb1, points, points, mode="bp")
    ev = result["evidence"]
    assert ev["diagnostics"]["initial_eligible"] == [1, 2]
    assert ev["unary"].shape == (3, 81)
    np.testing.assert_array_equal(ev["valid"][0], [True] + [False] * 80)
    np.testing.assert_array_equal(ev["edges"], [[0, 1], [0, 2]])
    np.testing.assert_array_equal(ev["factors"], np.zeros((2, 81, 81)))
    assert result["inference"]["choice"][0] == 0
    assert result["points0"][0].tobytes() == result["points1"][0].tobytes() == points[0].tobytes()
    assert _snap(rgb0, rgb1, points) == before


def test_flat_bridge_is_neutral_and_whole_vector_cost_is_clipped(api):
    points = np.asarray([[24., 32.], [88., 32.]], dtype=np.float32)
    texture = _texture(80, 128)
    sparse = np.full_like(texture, 2.)
    for x in (24, 88):
        sparse[:, 24:41, x - 8:x + 9] = texture[:, 24:41, x - 8:x + 9]
    flat = api.build_evidence(sparse, sparse, points, points)
    assert flat["diagnostics"]["initial_eligible"] == [0, 1]
    np.testing.assert_array_equal(flat["edges"], [[0, 1]])
    assert flat["diagnostics"]["zeroed_bridge_edges"] == [0]
    assert flat["diagnostics"]["bridge_scale"][0] == 0.
    np.testing.assert_array_equal(flat["factors"], np.zeros((1, 81, 81)))
    target = texture.copy()
    target[:, 32, 58] = 512.  # Outside zero-state 3x3 support, inside a shifted query.
    clipped = api.build_evidence(texture, target, points, points)
    states = api.state_grid()
    a = next(i for i, z in enumerate(states) if tuple(z) == (0., 0., 1., 0.))
    zero = (0., 0., 0., 0.)
    base0 = _bridge_values(texture, points, 0, 1, zero, zero, 0)
    base1 = _bridge_values(target, points, 0, 1, zero, zero, 1)
    mean = math.fsum(base0 + base1) / 54
    scale = math.sqrt(math.fsum((v - mean) ** 2 for v in base0 + base1) / 2)
    v0 = _bridge_values(texture, points, 0, 1, states[a], states[a], 0)
    v1 = _bridge_values(target, points, 0, 1, states[a], states[a], 1)
    assert math.fsum(((x - y) / scale) ** 2 for x, y in zip(v0, v1)) > 4.
    assert clipped["factors"][0, a, a] == 4.
    assert clipped["factors"][0, 0, 0] == pytest.approx(0., abs=1e-14)


def _aperture_pair(sign):
    source = _texture(80, 128)
    points = np.asarray([[24., 32.], [88., 32.]], dtype=np.float32)
    for qx, qy in points:
        for y in range(int(qy) - 8, int(qy) + 9):
            for x in range(int(qx) - 8, int(qx) + 9):
                source[:, y, x] = .5 + ((x - float(qx)) - (y - float(qy))) / 64.
    target = np.zeros_like(source)
    if sign == 1:
        target[:, 1:, 1:] = source[:, :-1, :-1]
    else:
        target[:, :-1, :-1] = source[:, 1:, 1:]
    return source, target, points


def test_unary_fixed_aperture_worlds_change_bridge_preference(api):
    plus = _aperture_pair(1)
    minus = _aperture_pair(-1)
    eplus = api.build_evidence(plus[0], plus[1], plus[2], plus[2])
    eminus = api.build_evidence(minus[0], minus[1], minus[2], minus[2])
    np.testing.assert_array_equal(eplus["unary"], eminus["unary"])
    np.testing.assert_array_equal(eplus["valid"], eminus["valid"])
    np.testing.assert_array_equal(eplus["edges"], [[0, 1]])
    np.testing.assert_array_equal(eminus["edges"], [[0, 1]])
    states = api.state_grid()
    positive = next(i for i, z in enumerate(states) if tuple(z) == (0., 0., 1., 1.))
    negative = next(i for i, z in enumerate(states) if tuple(z) == (0., 0., -1., -1.))
    for ev, correct, wrong in ((eplus, positive, negative), (eminus, negative, positive)):
        assert ev["valid"].all()
        assert ev["factors"][0, correct, correct] == pytest.approx(0., abs=2e-12)
        assert ev["factors"][0, wrong, wrong] > 1e-6
        _, residual, _ = api.decompose(ev["unary"], ev["edges"], ev["factors"])
        assert np.max(np.abs(residual)) > 1e-6
    assert not np.array_equal(eplus["factors"], eminus["factors"])
    # No optimizer-success claim: this only establishes extra nonseparable evidence.


@pytest.mark.parametrize("mode", ["local", "independent", "mean", "map", "bp", "shuffled"])
def test_flat_border_signed_zero_and_empty_preserve_whole_population(api, mode):
    rgb = np.full((3, 48, 64), 2., dtype=np.float32)
    p0 = np.asarray([[-0., 0.], [20., 20.], [100., -0.]], dtype=np.float32)
    p1 = np.asarray([[0., -0.], [20.5, 20.25], [100., -0.]], dtype=np.float32)
    saved = _snap(rgb, p0, p1)
    for a in (rgb, p0, p1):
        a.setflags(write=False)
    result = api.refine_pair(rgb, rgb, p0, p1, mode=mode)
    assert result["points0"].dtype == np.float32 and result["points1"].dtype == np.float32
    assert result["points0"].tobytes() == p0.tobytes()
    assert result["points1"].tobytes() == p1.tobytes()
    ev = result["evidence"]
    np.testing.assert_array_equal(ev["valid"][:, 0], [True, True, True])
    assert not ev["valid"][:, 1:].any()
    assert ev["edges"].shape == (0, 2)
    assert ev["factors"].shape == (0, 81, 81)
    assert _snap(rgb, p0, p1) == saved
    empty = np.empty((0, 2), dtype=np.float32)
    out = api.refine_pair(rgb, rgb, empty, empty, mode=mode)
    assert out["points0"].shape == out["points1"].shape == (0, 2)
    assert out["evidence"]["unary"].shape == (0, 81)
    assert out["inference"]["choice"].shape == (0,)


def test_positive_rgb_and_read_only_inputs_are_not_quantized_or_mutated(api):
    rgb0, rgb1 = _texture(), _texture(phase=.23)
    p0 = np.asarray([[24.25, 32.25], [88.25, 32.25]], dtype=np.float32)
    p1 = np.asarray([[24.5, 32.5], [88.5, 32.5]], dtype=np.float32)
    saved = _snap(rgb0, rgb1, p0, p1)
    assert rgb0.max() > 1. and rgb1.max() > 1.
    for a in (rgb0, rgb1, p0, p1):
        a.setflags(write=False)
    result = api.refine_pair(rgb0, rgb1, p0, p1, mode="bp")
    assert np.isfinite(result["evidence"]["unary"]).all()
    assert _snap(rgb0, rgb1, p0, p1) == saved
    states = api.state_grid()
    for side, points in enumerate((p0, p1)):
        expected = points.astype(np.float64) + states[result["inference"]["choice"], side * 2:side * 2 + 2]
        np.testing.assert_array_equal(result[f"points{side}"], expected.astype(np.float32))


@pytest.mark.parametrize("problem", ["negative", "nan_image", "layout", "image_dtype", "nan_points",
                                      "point_dtype", "count", "over64", "ids_duplicate", "ids_dtype"])
def test_public_evidence_rejects_malformed_inputs(api, problem):
    rgb0, rgb1 = _texture(), _texture()
    p0 = np.asarray([[24., 32.], [88., 32.]], dtype=np.float32)
    p1, kwargs = p0.copy(), {}
    if problem == "negative":
        rgb0[0, 0, 0] = -1.
    elif problem == "nan_image":
        rgb1[1, 0, 0] = np.nan
    elif problem == "layout":
        rgb0 = rgb0.transpose(1, 2, 0)
    elif problem == "image_dtype":
        rgb0 = rgb0.astype(np.float64)
    elif problem == "nan_points":
        p0[0, 0] = np.nan
    elif problem == "point_dtype":
        p0 = p0.astype(np.float64)
    elif problem == "count":
        p1 = p1[:1]
    elif problem == "over64":
        p0 = np.repeat(p0[:1], 65, axis=0)
        p1 = p0.copy()
    elif problem == "ids_duplicate":
        kwargs["row_ids"] = np.asarray([7, 7], dtype=np.int64)
    else:
        kwargs["row_ids"] = np.asarray([7., 8.], dtype=np.float64)
    with pytest.raises((TypeError, ValueError)):
        api.build_evidence(rgb0, rgb1, p0, p1, **kwargs)


@pytest.mark.parametrize("problem", ["bad_mode", "bad_valid", "no_valid", "bad_edges", "self_edge", "nan_factor"])
def test_inference_rejects_invalid_contracts(api, problem):
    unary = np.asarray([[0., 1.], [1., 0.]], dtype=np.float64)
    edges = np.asarray([[0, 1]], dtype=np.int64)
    factors = np.asarray([[[0., 1.], [1., 0.]]], dtype=np.float64)
    valid, mode = np.ones((2, 2), dtype=bool), "bp"
    if problem == "bad_mode":
        mode = "unknown"
    elif problem == "bad_valid":
        valid = np.ones((2, 3), dtype=bool)
    elif problem == "no_valid":
        valid[1] = False
    elif problem == "bad_edges":
        edges[0, 1] = 2
    elif problem == "self_edge":
        edges[0] = [0, 0]
    else:
        factors[0, 0, 0] = np.nan
    with pytest.raises((TypeError, ValueError)):
        api.infer(unary, edges, factors, valid, mode=mode)
