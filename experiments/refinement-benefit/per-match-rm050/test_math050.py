"""Fabricated boundary/algebra checks for RM050; no external data."""
import numpy as np
import pytest

import math050 as m


def cache(n=4):
    p0 = np.arange(n * 2, dtype=np.float32).reshape(n, 2) + np.float32(0.125)
    p1 = p0 + np.float32(0.25)
    return dict(p0=p0, p1=p1, q0=p0 + np.float32(1.0),
                q1=p1 - np.float32(2.0),
                x0=np.arange(n * 576, dtype=np.float32).reshape(n, 64, 3, 3),
                x1=-np.arange(n * 576, dtype=np.float32).reshape(n, 64, 3, 3))


def horizontal_F():
    return m.fundamental(np.eye(3), np.eye(3), np.eye(3), np.array([1., 0., 0.]))


def test_horizontal_risk_and_unobserved_along_line_motion():
    p0 = np.array([[1., 2.], [3., 4.]])
    p1 = np.array([[7., 2.], [20., 6.]])
    expected = np.array([0., np.log1p(2.)])
    np.testing.assert_allclose(m.risk(p0, p1, horizontal_F()), expected, atol=1e-14)
    along = p1.copy()
    along[:, 0] += 10000.
    np.testing.assert_array_equal(m.risk(p0, p1, horizontal_F()), m.risk(p0, along, horizontal_F()))
    np.testing.assert_allclose(m.risk(p0, p1, horizontal_F() * -7.), expected, atol=1e-14)


def test_directed_fundamental_transposition_and_translation_scale():
    angle = .3
    R = np.array([[np.cos(angle), -np.sin(angle), 0.],
                  [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
    K0 = np.array([[400., 0., 200.], [0., 410., 150.], [0., 0., 1.]])
    K1 = np.array([[500., 0., 210.], [0., 510., 160.], [0., 0., 1.]])
    t = np.array([1., .2, .3])
    F = m.fundamental(K0, K1, R, t)
    reverse = m.fundamental(K1, K0, R.T, -R.T @ t)
    np.testing.assert_allclose(F.T, reverse, atol=1e-15)
    np.testing.assert_allclose(F, m.fundamental(K0, K1, R, t * 20.), atol=1e-15)
    assert np.isclose(np.linalg.norm(F), 1.)


def test_features_use_actual_native_raw_and_spatial_channel_order():
    c = cache()
    before = {k: v.tobytes() for k, v in c.items()}
    result = m.features(c)
    assert result.shape == (4, 1156) and result.dtype == np.float64
    np.testing.assert_array_equal(result[:, :576], c['x0'].reshape(4, 576))
    np.testing.assert_array_equal(result[:, 576:1152], c['x1'].reshape(4, 576))
    np.testing.assert_array_equal(result[:, -4:], np.tile([1., 1., -2., -2.], (4, 1)))
    assert not np.array_equal(c['q0'] - (np.floor(c['p0']) + .5), result[:, -4:-2])
    assert before == {k: v.tobytes() for k, v in c.items()}


def test_endpoint_actions_tie_preference_and_exact_bytes():
    c = cache()
    scores = np.array([[0., 0., 0., 0.], [3., -4., 3., 3.],
                       [2., 2., -1., 2.], [2., 2., 2., -1.]])
    actions = m.choose(scores)
    np.testing.assert_array_equal(actions, np.arange(4, dtype=np.int64))
    out0, out1 = m.endpoints(c, actions)
    for i, (k0, k1) in enumerate((('q0', 'q1'), ('p0', 'p1'),
                                 ('p0', 'q1'), ('q0', 'p1'))):
        assert out0[i].tobytes() == c[k0][i].tobytes()
        assert out1[i].tobytes() == c[k1][i].tobytes()
    all_risks = m.action_risks(c, horizontal_F())
    np.testing.assert_array_equal(m.policy_loss(c, horizontal_F(), actions),
                                  all_risks[np.arange(4), actions])
    half0, half1 = m.endpoints(c, np.full(4, 4, np.int64))
    for actual, p, q in ((half0, c['p0'], c['q0']), (half1, c['p1'], c['q1'])):
        assert actual.tobytes() == ((p.astype(np.float64) + q) / 2).astype(np.float32).tobytes()


def test_streamed_population_moments_constant_and_tiny_features():
    x = np.array([[2., 4., 1e-14], [4., 4., 3e-14], [6., 4., 2e-14]])
    moments = m.FeatureMoments(3)
    moments.add(x[:1])
    moments.add(x[1:])
    mean, std = moments.finish()
    np.testing.assert_allclose(mean, x.mean(axis=0), atol=1e-15)
    np.testing.assert_allclose(std, [np.sqrt(8. / 3.), 1., 1.], atol=1e-15)


def test_ridge_closed_form_unpenalized_intercept_and_streaming():
    x = np.array([[-1.], [1.]])
    multipliers = np.arange(1., 5.)
    y = 3. + x * multipliers
    ridge = m.RidgeAccumulator(np.zeros(1), np.ones(1))
    ridge.add(x[:1], y[:1])
    ridge.add(x[1:], y[1:])
    w = ridge.solve()
    np.testing.assert_allclose(w[0], (2. / 3.) * multipliers, atol=1e-14)
    np.testing.assert_allclose(w[-1], np.full(4, 3.), atol=1e-14)
    z = np.column_stack((x, np.ones(2)))
    np.testing.assert_allclose(w, np.linalg.solve(z.T @ z + np.diag([1., 0.]), z.T @ y), atol=1e-14)
    np.testing.assert_allclose(m.predict(x, np.zeros(1), np.ones(1), w), z @ w)


def test_empty_rows_are_retained_and_empty_fitting_rejected():
    c = cache(0)
    assert m.features(c).shape == (0, 1156)
    assert m.action_risks(c, horizontal_F()).shape == (0, 5)
    assert m.choose(np.empty((0, 4))).shape == (0,)
    with pytest.raises(ValueError):
        m.FeatureMoments(1).finish()
    with pytest.raises(ValueError):
        m.RidgeAccumulator(np.zeros(1), np.ones(1)).solve()


@pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf])
def test_nonfinite_inputs_reject_entire_population(bad):
    c = cache()
    c['x1'][2, 0, 0, 0] = bad
    with pytest.raises(ValueError):
        m.features(c)
    c = cache()
    c['q0'][1, 0] = bad
    with pytest.raises(ValueError):
        m.action_risks(c, horizontal_F())
    with pytest.raises(ValueError):
        m.choose(np.array([[0., 1., bad, 2.]]))


def test_invalid_geometry_and_selector_rejected_without_row_dropping():
    with pytest.raises(ValueError):
        m.fundamental(np.eye(3), np.eye(3), np.eye(3), np.zeros(3))
    with pytest.raises(ValueError):
        m.fundamental(np.zeros((3, 3)), np.eye(3), np.eye(3), np.ones(3))
    with pytest.raises(ValueError):
        m.fundamental(np.eye(3), np.eye(3), np.diag([-1., 1., 1.]), np.ones(3))
    with pytest.raises(ValueError):
        m.risk(np.zeros((1, 2)), np.zeros((1, 2)), np.zeros((3, 3)))
    with pytest.raises(ValueError):
        m.endpoints(cache(), np.array([0, 1, 2, 5], dtype=np.int64))
    with pytest.raises(ValueError):
        m.endpoints(cache(), np.zeros(4, dtype=bool))
    with pytest.raises(ValueError):
        m.RidgeAccumulator(np.zeros(1), np.zeros(1))
