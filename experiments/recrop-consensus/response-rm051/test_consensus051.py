"""Independent scalar fixtures for fixed RM051 arithmetic; no real payloads."""
import hashlib
import math

import numpy as np
import pytest

import consensus051 as c


def inputs(n):
    return (np.zeros((9, n, 2), np.float32), np.zeros((9, n, 2), np.float32),
            np.zeros(n, np.bool_), np.arange(n, dtype=np.int64))


def scalar_reference(a, b):
    """Slow scalar loops with an independent sqrt/max distance computation."""
    bests, counts, energies = [], [], []
    for row in range(a.shape[1]):
        candidates = []
        for k in range(9):
            energy, support = 0., 0
            for j in range(9):
                left = math.hypot(float(a[k, row, 0]) - float(a[j, row, 0]),
                                  float(a[k, row, 1]) - float(a[j, row, 1]))
                right = math.hypot(float(b[k, row, 0]) - float(b[j, row, 0]),
                                   float(b[k, row, 1]) - float(b[j, row, 1]))
                distance = max(left, right)
                if distance <= 1.:
                    support += 1
                    energy += distance * distance
            candidates.append((-support, energy, k))
        selected = min(candidates)
        bests.append(selected[2])
        counts.append(-selected[0])
        energies.append(selected[1])
    return np.array(bests, np.int64), np.array(counts, np.int64), np.array(energies)


def test_scalar_reference_random_rows_and_recipient_branch_bytes():
    rng = np.random.default_rng(451)
    a, b, edge, ids = inputs(31)
    a[:] = rng.normal(size=a.shape).astype(np.float32)
    b[:] = rng.normal(size=b.shape).astype(np.float32)
    original = (a.tobytes(), b.tobytes(), ids.tobytes(), edge.tobytes())
    r = c.outputs(a, b, edge, ids, 'pair_0007')
    best, support, energy = scalar_reference(a, b)
    np.testing.assert_array_equal(r['best'], best)
    np.testing.assert_array_equal(r['support'], support)
    np.testing.assert_allclose(r['energy'], energy, rtol=1e-14, atol=1e-14)
    np.testing.assert_array_equal(r['choice'], np.where(support >= 5, best, 0))
    for name, selected in (('C', r['choice']), ('U', r['ungated_choice']),
                           ('F', r['fixed_choice']), ('S', r['shuffled_choice'])):
        for side, branches in ((0, a), (1, b)):
            for row, branch in enumerate(selected):
                assert r[name + str(side)][row].tobytes() == branches[branch, row].tobytes()
    assert original == (a.tobytes(), b.tobytes(), ids.tobytes(), edge.tobytes())


def test_identical_responses_prefer_baseline_and_preserve_signed_zero():
    a, b, edge, ids = inputs(3)
    a[:, :, 0] = np.float32(-0.)
    r = c.outputs(a, b, edge, ids, 'same')
    np.testing.assert_array_equal(r['choice'], np.zeros(3, np.int64))
    np.testing.assert_array_equal(r['support'], np.full(3, 9))
    np.testing.assert_array_equal(r['energy'], np.zeros(3))
    assert r['C0'].tobytes() == a[0].tobytes()


def test_no_majority_compact_majority_energy_and_branch_rank_ties():
    a, b, edge, ids = inputs(4)
    # No two branches agree: all support one, energy zero, rank0 wins.
    a[:, 0, 0] = np.arange(9) * 3.
    # Five equal branches at 10 agree, baseline belongs to another cluster.
    a[:, 1, 0] = np.array([0., 10., 10., 10., 10., 10., 30., 40., 50.])
    # Maximal support tie, but central branch3 has lowest supported energy.
    a[:, 2, 0] = np.array([20., 0., .125, .25, .375, .5, 30., 40., 50.])
    # Four coherent responses cannot trigger C, but U emits their first branch.
    a[:, 3, 0] = np.array([0., 10., 10., 10., 10., 30., 40., 50., 60.])
    r = c.outputs(a, b, edge, ids, 'cases')
    np.testing.assert_array_equal(r['best'], [0, 1, 3, 1])
    np.testing.assert_array_equal(r['support'], [1, 5, 5, 4])
    np.testing.assert_array_equal(r['choice'], [0, 1, 3, 0])
    np.testing.assert_array_equal(r['ungated_choice'], [0, 1, 3, 1])


def test_inclusive_one_pixel_and_maximum_endpoint_distance():
    a, b, edge, ids = inputs(2)
    a[:, 0, 0] = np.array([0., 1., 1., 1., 1., 5., 6., 7., 8.])
    b[:, 0, 0] = a[:, 0, 0]
    # Endpoint1 alone separates an otherwise identical endpoint0 cluster.
    b[:, 1, 0] = np.arange(9) * 2.
    r = c.outputs(a, b, edge, ids, 'boundary')
    assert r['support'][0] >= 5
    assert r['support'][1] == 1
    assert r['choice'][1] == 0


def test_edge_rule_and_common_exact_baseline_fallback():
    p0 = np.array([[0., 2.], [.999, 2.], [1., 1.], [8.9, 8.9],
                   [9., 3.], [-.1, 2.], [10., 3.]], np.float32)
    p1 = np.full_like(p0, 3.)
    edge = c.edge_mask(p0, p1, (10, 10), (10, 10))
    np.testing.assert_array_equal(edge, [True, True, False, False, True, True, True])
    a, b, _, ids = inputs(7)
    a[0] = np.arange(14, dtype=np.float32).reshape(7, 2)
    b[0] = -a[0]
    a[1:] = 100.
    b[1:] = -100.
    r = c.outputs(a, b, edge, ids, 'edge')
    for prefix in ('C', 'U', 'M', 'F', 'S'):
        for side, branches in ((0, a), (1, b)):
            assert r[prefix + str(side)][edge].tobytes() == branches[0, edge].tobytes()
    np.testing.assert_array_equal(r['shuffle_mapping'][edge], np.flatnonzero(edge))


def test_shuffle_exact_seed_eligible_marginals_and_fallback_zero_indices():
    a, b, edge, ids = inputs(24)
    # Choose different coherent majority branch ranks, or fallback, by row.
    for row in range(24):
        a[:, row, 0] = np.arange(9) * 20.
        start = row % 5
        a[start:start + 5, row, 0] = 200.
    edge[[2, 8, 19]] = True
    r = c.outputs(a, b, edge, ids, 'shuffle')
    eligible = np.flatnonzero(~edge)
    seed = int.from_bytes(hashlib.sha256(b'RM051/20261102/shuffle').digest()[:8], 'little')
    expected = np.arange(24, dtype=np.int64)
    expected[eligible] = eligible[np.random.default_rng(seed).permutation(len(eligible))]
    np.testing.assert_array_equal(r['shuffle_mapping'], expected)
    np.testing.assert_array_equal(r['shuffled_choice'], r['choice'][expected])
    np.testing.assert_array_equal(np.sort(r['shuffled_choice'][eligible]), np.sort(r['choice'][eligible]))
    assert not np.any(r['shuffled_choice'][edge])
    np.testing.assert_array_equal(r['row_ids'], ids)
    again = c.outputs(a, b, edge, ids, 'shuffle')
    np.testing.assert_array_equal(r['shuffle_mapping'], again['shuffle_mapping'])


def test_median_is_coordinatewise_fp64_then_fp32_and_empty_rows_supported():
    a, b, edge, ids = inputs(2)
    a[:] = np.arange(36, dtype=np.float32).reshape(9, 2, 2)
    b[:] = -a
    r = c.outputs(a, b, edge, ids, 'median')
    for side, branches in ((0, a), (1, b)):
        expected = np.median(branches.astype(np.float64), axis=0).astype(np.float32)
        assert r['M' + str(side)].tobytes() == expected.tobytes()
    e = c.outputs(*inputs(0), 'empty')
    assert e['C0'].shape == (0, 2) and e['choice'].shape == (0,)
    assert c.edge_mask(np.empty((0, 2), np.float32), np.empty((0, 2), np.float32), (1, 1), (1, 1)).shape == (0,)


@pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf])
def test_nonfinite_response_or_raw_coordinate_rejects_whole_population(bad):
    a, b, edge, ids = inputs(2)
    a[3, 1, 0] = bad
    with pytest.raises(ValueError):
        c.outputs(a, b, edge, ids, 'bad')
    p = np.array([[bad, 2.]], np.float32)
    with pytest.raises(ValueError):
        c.edge_mask(p, np.ones((1, 2), np.float32), (10, 10), (10, 10))


def test_invalid_identity_shape_type_and_image_dimensions_rejected():
    a, b, edge, ids = inputs(2)
    for invalid in (np.array([0, 0], np.int64), np.array([-1, 2], np.int64),
                    np.array([0., 1.]), np.array([True, False]), np.array([0], np.int64)):
        with pytest.raises(ValueError):
            c.outputs(a, b, edge, invalid, 'bad')
    with pytest.raises(ValueError):
        c.outputs(a.astype(np.float64), b, edge, ids, 'bad')
    with pytest.raises(ValueError):
        c.outputs(a[:8], b, edge, ids, 'bad')
    with pytest.raises(ValueError):
        c.outputs(a, b, edge.astype(np.int64), ids, 'bad')
    with pytest.raises(ValueError):
        c.outputs(a, b, edge, ids, 'bad\nuid')
    with pytest.raises(ValueError):
        c.edge_mask(np.ones((1, 2), np.float32), np.ones((1, 2), np.float32), (True, 10), (10, 10))
