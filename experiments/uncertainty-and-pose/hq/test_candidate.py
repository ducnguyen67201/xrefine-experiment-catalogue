"""Prospective synthetic checks for the locked joint-HQ pose adaptation.

SOURCE ONLY until ROOT admits the offline stage. No real images, saved poses,
ground truth files, model inference, GC fits or OpenCV recovery are used here.
Numerical references deliberately use different constructions: Rodrigues
projection, central differences and a full tiny block system.
"""
import importlib.util
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


HERE = Path(__file__).resolve().parent
MODES = ('ba_continue25', 'gm_irls25', 'gm_hq_gc25', 'gm_hq_all25',
         'ba_continue_time')


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def modules():
    # These imports happen only during a separately admitted offline run.
    import numpy as np
    candidate = _load('joint_hq_candidate_offline', 'candidate.py')
    return np, candidate


@pytest.fixture(scope='module')
def adapter():
    return _load('joint_hq_adapter_offline', 'pose_adapter.py')


def _cross_matrix(np, vector):
    return np.cross(vector, np.eye(3)).T


def _rotation(np, omega):
    angle = float(np.linalg.norm(omega))
    if angle == 0:
        return np.eye(3)
    axis = omega / angle
    outer = np.outer(axis, axis)
    return (math.cos(angle) * np.eye(3) + (1 - math.cos(angle)) * outer
            + math.sin(angle) * _cross_matrix(np, axis))


def _unit(np, vector):
    return vector / np.linalg.norm(vector)


def _scene(np):
    """Explicit two-camera geometry, independent of the candidate algorithm."""
    points = np.array([
        [-1.10 + .42 * (i % 6), -.75 + .46 * (i // 6),
         4.5 + .23 * i + .17 * (i % 3)] for i in range(24)
    ], dtype=np.float64)
    K0 = np.array([[520., 3., 320.], [0., 510., 240.], [0., 0., 1.]])
    K1 = np.array([[610., 0., 300.], [0., 590., 230.], [0., 0., 1.]])
    actual_R = _rotation(np, np.array([.015, -.025, .01]))
    actual_t = _unit(np, np.array([.60, -.07, .11]))

    def project(camera, xyz):
        homogeneous = xyz @ camera.T
        return homogeneous[:, :2] / homogeneous[:, 2:]

    p0 = project(K0, points).astype(np.float32)
    p1 = project(K1, points @ actual_R.T + actual_t).astype(np.float32)
    seed_R = _rotation(np, np.array([.0015, -.002, .0007])) @ actual_R
    seed_t = _unit(np, actual_t + np.array([.001, .004, -.002]))
    mask = np.array([i % 4 != 0 for i in range(24)], dtype=np.uint8)
    return p0, p1, K0, K1, seed_R, seed_t, mask


def _independent_residual(np, x0, x1, R, t, fbar):
    E = _cross_matrix(np, t) @ R
    answer = []
    for first, second in zip(x0, x1):
        forward = E @ first
        reverse = E.T @ second
        squared = sum(float(value) ** 2 for value in
                      (*forward[:2], *reverse[:2]))
        answer.append(fbar * float(second @ E @ first) / math.sqrt(squared))
    return np.array(answer)


def _problem(np):
    residual = np.array([-3., -1., .25, .8, 1.5, 4., 8.])
    jacobian = np.array([
        [1., .2, -.3, .4, -.1], [.1, 1., .4, -.2, .3],
        [-.2, .3, 1., .5, .1], [.4, -.1, .2, 1., .3],
        [.2, .5, -.1, .3, 1.], [1., -.3, .6, .2, -.5],
        [-.4, 1., .2, -.6, .7],
    ])
    return residual, jacobian


def _full_damped_step(np, residual, jacobian, auxiliary, width, damping):
    """Build the full residual Jacobian, not the reduced Schur expression."""
    n = len(residual)
    full_J = np.zeros((2 * n, 5 + n))
    full_J[:n, :5] = auxiliary[:, None] * jacobian
    full_J[:n, 5:] = np.diag(residual)
    full_J[n:, 5:] = -width * np.eye(n)
    full_r = np.concatenate((auxiliary * residual, width * (1 - auxiliary)))
    normal = full_J.T @ full_J
    normal += np.diag(damping * np.maximum(np.diag(normal), 1e-9))
    return np.linalg.solve(normal, -(full_J.T @ full_r))


def _gm(np, residual, width=1.):
    return .5 * float(np.sum(width ** 2 * residual ** 2 /
                             (width ** 2 + residual ** 2)))


def _lift(np, residual, auxiliary, width=1.):
    return .5 * float(np.sum((auxiliary * residual) ** 2 +
                             (width * (1 - auxiliary)) ** 2))


def test_default_configuration_is_the_exact_prospective_lock(modules):
    _, candidate = modules
    lock = json.loads((HERE / 'lock.json').read_text())
    assert candidate.DEFAULT_CONFIG == lock['config']
    assert tuple(candidate.MODES) == MODES


def test_calibration_preserves_all_rows_and_does_not_mutate(modules):
    np, candidate = modules
    original = list(_scene(np))
    # A finite out-of-raster point is retained, not clipped or silently removed.
    original[0] = original[0].copy()
    original[0][0] = [-12., 900.]
    before = [array.copy() for array in original]
    x0, x1, R, t, mask, fbar = candidate.validate_inputs(np, *original)
    assert x0.shape == x1.shape == (24, 3)
    assert np.all(x0[:, 2] == 1.) and np.all(x1[:, 2] == 1.)
    expected = np.linalg.solve(original[2], np.array([-12., 900., 1.]))
    assert np.allclose(x0[0], expected / expected[2], atol=1e-12, rtol=0)
    assert np.array_equal(mask, original[6].astype(bool))
    assert fbar == .25 * (520 + 510 + 610 + 590)
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-12)
    assert abs(np.linalg.norm(t) - 1) < 1e-14
    for old, now in zip(before, original):
        assert np.array_equal(old, now)


@pytest.mark.parametrize('case', ('point_shape', 'point_nonfinite', 'point_count',
                                  'bad_mask', 'bad_R', 'zero_t', 'bad_K'))
def test_malformed_inputs_are_rejected_not_trimmed(modules, case):
    np, candidate = modules
    arguments = list(_scene(np))
    if case == 'point_shape':
        arguments[0] = np.ones((24, 3))
    elif case == 'point_nonfinite':
        arguments[1] = arguments[1].copy()
        arguments[1][2, 1] = np.nan
    elif case == 'point_count':
        arguments[1] = arguments[1][:-1]
    elif case == 'bad_mask':
        arguments[6] = arguments[6].copy()
        arguments[6][3] = 2
    elif case == 'bad_R':
        arguments[4] = np.diag([1., 1., -1.])
    elif case == 'zero_t':
        arguments[5] = np.zeros(3)
    else:
        arguments[2] = arguments[2].copy()
        arguments[2][0, 0] = np.nan
    with pytest.raises(ValueError):
        candidate.validate_inputs(np, *arguments)


def test_analytic_five_dof_jacobian_against_independent_central_differences(modules):
    np, candidate = modules
    x0, x1, R, t, _, fbar = candidate.validate_inputs(np, *_scene(np))
    residual, jacobian, basis = candidate.linearize(np, x0, x1, R, t, fbar)
    assert jacobian.shape == (24, 5)
    assert np.allclose(basis.T @ basis, np.eye(2), atol=1e-14)
    assert np.allclose(t @ basis, np.zeros(2), atol=1e-14)
    assert np.allclose(residual, _independent_residual(np, x0, x1, R, t, fbar),
                       atol=1e-10, rtol=1e-12)
    numerical = np.zeros_like(jacobian)
    step = 2e-7
    for column in range(5):
        delta = np.zeros(5)
        delta[column] = step
        values = []
        for sign in (-1., 1.):
            moved_R = _rotation(np, sign * delta[:3]) @ R
            moved_t = _unit(np, t + basis @ (sign * delta[3:]))
            values.append(_independent_residual(np, x0, x1, moved_R, moved_t, fbar))
        numerical[:, column] = (values[1] - values[0]) / (2 * step)
    assert np.allclose(jacobian, numerical, atol=3e-5, rtol=2e-6)


def test_squared_denominator_guard_rejects_instead_of_clamping(modules):
    np, candidate = modules
    R = np.eye(3)
    t = np.array([0., 0., 1.])
    for offset in (0., .5e-12):
        rays = np.array([[offset, 0., 1.]] * 5)
        # D=2*offset**2 is zero or 5e-25, below the frozen squared floor.
        with pytest.raises(candidate.DegenerateGeometry):
            candidate.linearize(np, rays, rays, R, t, 1., False)
    rays = np.array([[1e-12, 0., 1.]] * 5)
    residual, jacobian, _ = candidate.linearize(np, rays, rays, R, t, 1., False)
    assert np.array_equal(residual, np.zeros(5))
    assert jacobian is None


@pytest.mark.parametrize('need_jacobian', (False, True))
def test_nonfinite_residual_is_rejected_also_on_the_cost_only_path(modules, need_jacobian):
    np, candidate = modules
    first = np.array([[0., 0., 1.]] * 5)
    second = np.array([[0., 1e10, 1.]] * 5)
    # All operands and D=2 are finite, but multiplying by fbar overflows.
    with np.errstate(over='ignore', invalid='ignore'):
        with pytest.raises(candidate.DegenerateGeometry):
            candidate.linearize(np, first, second, np.eye(3),
                                np.array([1., 0., 0.]), 1e308, need_jacobian)


def test_canonical_fbar_scaling_and_translation_axis_sign(modules):
    np, candidate = modules
    data = _scene(np)
    x0, x1, R, t, _, fbar = candidate.validate_inputs(np, *data)
    r, J, _ = candidate.linearize(np, x0, x1, R, t, fbar)
    scaled_r, scaled_J, _ = candidate.linearize(np, x0, x1, R, t, 2 * fbar)
    assert np.allclose(scaled_r, 2 * r, atol=1e-12, rtol=1e-12)
    assert np.allclose(scaled_J, 2 * J, atol=1e-12, rtol=1e-12)
    opposite, _, _ = candidate.linearize(np, x0, x1, R, -t, fbar, False)
    assert np.allclose(opposite, -r, atol=1e-12, rtol=1e-12)
    assert abs(_gm(np, opposite) - _gm(np, r)) < 1e-12


@pytest.mark.parametrize('damping', (1e-6, .001, 3.))
def test_unconstrained_schur_step_matches_full_tiny_block_solve(modules, damping):
    np, candidate = modules
    residual, jacobian = _problem(np)
    auxiliary = np.array([1., .9, .7, .4, .2, .05, .8])
    expected = _full_damped_step(np, residual, jacobian, auxiliary, 1., damping)
    pose_step, auxiliary_step = candidate.joint_step(
        np, residual, jacobian, auxiliary, 1., damping)
    actual = np.concatenate((pose_step, auxiliary_step))
    assert pose_step.shape == (5,) and auxiliary_step.shape == (7,)
    assert np.allclose(actual, expected, atol=1e-9, rtol=1e-9)


def test_exact_gm_lift_and_conditional_auxiliary_stationarity(modules):
    np, candidate = modules
    residual = np.array([0., .01, .5, 1., 3., 100.])
    width = 1.
    optimum = width ** 2 / (width ** 2 + residual ** 2)
    assert abs(_lift(np, residual, optimum, width) - _gm(np, residual, width)) < 1e-12
    for alternative in (np.zeros(6), np.ones(6), np.full(6, .37)):
        assert _lift(np, residual, alternative, width) >= _gm(np, residual, width) - 1e-12
    dp, du = candidate.joint_step(np, residual, np.zeros((6, 5)), optimum, width, .001)
    assert np.allclose(dp, np.zeros(5), atol=1e-12)
    assert np.allclose(du, np.zeros(6), atol=1e-12)


def test_hq_joint_first_step_is_not_residual_reset_irls(modules):
    np, candidate = modules
    residual, jacobian = _problem(np)
    damping = .001
    hq_step, _ = candidate.joint_step(np, residual, jacobian, np.ones(7), 1., damping)
    square_root_weight = 1 / (1 + residual ** 2)
    weighted = square_root_weight[:, None] * jacobian
    normal = weighted.T @ weighted
    normal += np.diag(damping * np.maximum(np.diag(normal), 1e-9))
    irls_step = np.linalg.solve(normal, -(jacobian.T @ (square_root_weight ** 2 * residual)))
    assert np.linalg.norm(hq_step - irls_step) > 1e-5


def test_retraction_limits_are_separate_from_unconstrained_schur_math(modules):
    np, candidate = modules
    _, _, _, _, R, t, _ = _scene(np)
    t = _unit(np, t)
    step = np.array([1., -2., 3., -4., 5.])
    untouched = step.copy()
    small = step.copy()
    small[:3] *= .25 / np.linalg.norm(small[:3])
    small[3:] *= .25 / np.linalg.norm(small[3:])
    basis = candidate.tangent_basis(np, t)
    moved_R, moved_t = candidate.retract(np, R, t, step)
    assert np.allclose(moved_R, _rotation(np, small[:3]) @ R, atol=1e-12)
    assert np.allclose(moved_t, _unit(np, t + basis @ small[3:]), atol=1e-12)
    assert np.allclose(moved_R.T @ moved_R, np.eye(3), atol=1e-12)
    assert abs(float(np.linalg.det(moved_R)) - 1) < 1e-12
    assert abs(float(np.linalg.norm(moved_t)) - 1) < 1e-12
    assert np.array_equal(step, untouched)


@pytest.mark.parametrize('mode', MODES)
def test_locked_modes_are_finite_bounded_and_leave_inputs_unchanged(modules, mode):
    np, candidate = modules
    data = _scene(np)
    before = [value.copy() for value in data]
    arguments = {'wall_seconds': 1e6} if mode == 'ba_continue_time' else {}
    result = candidate.polish(np, *data, mode, **arguments)
    assert result['valid'] is True and result['mode'] == mode
    assert result['native_calls'] == result['model_calls'] == 0
    assert result['recovery_pending'] is True
    assert 0 <= result['accepted_steps'] <= result['attempts'] <= (100 if mode == 'ba_continue_time' else 25)
    assert len(result['trace']) == result['attempts']
    assert result['final_auxiliary'].shape == (24,)
    assert np.isfinite(result['final_auxiliary']).all()
    assert ((result['final_auxiliary'] >= 0) & (result['final_auxiliary'] <= 1)).all()
    assert np.allclose(result['R'].T @ result['R'], np.eye(3), atol=1e-10)
    assert abs(float(np.linalg.det(result['R'])) - 1) < 1e-10
    assert abs(float(np.linalg.norm(result['t'])) - 1) < 1e-12
    assert np.allclose(result['E'], _cross_matrix(np, result['t']) @ result['R'], atol=1e-12)
    for entry in result['trace']:
        assert entry['pose_increment_norm'] <= math.sqrt(.25 ** 2 + .25 ** 2) + 1e-12
        if entry['accepted']:
            assert entry['objective_trial'] < entry['objective_before']
        elif entry['objective_trial'] is not None:
            assert entry['objective_trial'] >= entry['objective_before']
    if mode.startswith('gm_'):
        assert result['final_objective'] <= result['initial_objective'] + 1e-10
    for old, now in zip(before, data):
        assert np.array_equal(old, now)


def test_terminal_joint_cost_uses_projected_weights_and_all_rows(modules):
    np, candidate = modules
    data = _scene(np)
    result = candidate.polish(np, *data, 'gm_hq_all25')
    x0, x1, R0, t0, _, fbar = candidate.validate_inputs(np, *data)
    initial = _independent_residual(np, x0, x1, R0, t0, fbar)
    terminal = _independent_residual(np, x0, x1, result['R'], result['t'], fbar)
    assert abs(result['initial_objective'] - .5 * float(initial @ initial)) < 1e-9
    assert abs(result['final_objective'] - _lift(np, terminal, result['final_auxiliary'])) < 1e-9
    assert abs(result['initial_gm_cost'] - _gm(np, initial)) < 1e-9
    assert abs(result['final_gm_cost'] - _gm(np, terminal)) < 1e-9
    assert result['selected_rows'] == len(initial)


def _policy_residual(np, candidate, value=2., jacobian_entry=100.):
    """Policy-only synthetic scalar model; not a Sampson/accuracy fixture."""
    def linearize(np_, x0, x1, R, t, fbar, need_jacobian=True):
        residual = np_.full(len(x0), value)
        jacobian = None
        if need_jacobian:
            jacobian = np_.zeros((len(x0), 5))
            jacobian[:, 0] = jacobian_entry
        return residual, jacobian, candidate.tangent_basis(np_, t)
    return linearize


def test_clipped_auxiliary_acceptance_uses_full_lift_not_marginal_gm(modules, monkeypatch):
    np, candidate = modules
    data = _scene(np)
    monkeypatch.setattr(candidate, 'linearize', _policy_residual(np, candidate))

    def specified_step(np_, r, J, u, c, damping):
        return np_.array([.25, 0., 0., 0., 0.]), np_.zeros(len(r))

    monkeypatch.setattr(candidate, 'joint_step', specified_step)
    result = candidate.polish(np, *data, 'gm_hq_all25')
    first = result['trace'][0]
    n = len(data[0])
    # u+du is strictly negative for the injected first step, hence clipped to0.
    # L decreases2N->N/2 while marginalGM is unchanged: checking marginalGM
    # instead of the actual projected joint objective would reject this step.
    assert first['accepted'] is True
    assert abs(first['objective_before'] - 2 * n) < 1e-10
    assert abs(first['objective_trial'] - .5 * n) < 1e-10
    assert abs(first['auxiliary_increment_norm'] - math.sqrt(n)) < 1e-10
    assert result['initial_gm_cost'] == result['final_gm_cost']
    assert ((result['final_auxiliary'] >= 0) & (result['final_auxiliary'] <= 1)).all()


def test_auxiliary_only_progress_is_not_reported_as_pose_movement(modules, monkeypatch):
    np, candidate = modules
    data = _scene(np)
    _, _, initial_R, initial_t, _, _ = candidate.validate_inputs(np, *data)
    monkeypatch.setattr(candidate, 'linearize', _policy_residual(np, candidate))
    monkeypatch.setattr(candidate, 'joint_step',
                        lambda np_, r, J, u, c, damping: (np_.zeros(5), np_.zeros(len(r))))
    result = candidate.polish(np, *data, 'gm_hq_all25')
    assert result['accepted_steps'] > 0
    assert result['final_objective'] < result['initial_objective']
    assert result['initial_gm_cost'] == result['final_gm_cost']
    assert np.allclose(result['R'], initial_R, atol=2e-15, rtol=0)
    assert np.allclose(result['t'], initial_t, atol=2e-15, rtol=0)
    assert result['rotation_matrix_movement'] <= 2e-15
    assert result['translation_vector_movement'] <= 2e-15
    assert all(entry['pose_increment_norm'] == 0 for entry in result['trace'])


@pytest.mark.parametrize('failure', ('singular', 'nonfinite'))
def test_numerical_trial_failure_is_bounded_recorded_and_keeps_seed(modules, monkeypatch, failure):
    np, candidate = modules
    data = _scene(np)
    _, _, seed_R, seed_t, _, _ = candidate.validate_inputs(np, *data)

    def rejected_step(np_, r, J, u, c, damping):
        if failure == 'singular':
            raise np_.linalg.LinAlgError('Explicit synthetic singular step')
        return np_.full(5, np_.nan), np_.zeros(len(r))

    monkeypatch.setattr(candidate, 'joint_step', rejected_step)
    result = candidate.polish(np, *data, 'gm_hq_all25')
    assert result['valid'] is True and 0 < result['attempts'] <= 25
    assert result['accepted_steps'] == 0
    assert np.array_equal(result['R'], seed_R) and np.array_equal(result['t'], seed_t)
    count = result['solve_rejections'] if failure == 'singular' else result['rejected_trial_geometry']
    assert count == result['attempts']
    for entry in result['trace']:
        assert entry['accepted'] is False and entry['rejection'] is not None
        assert entry['objective_trial'] is None or math.isfinite(entry['objective_trial'])
        assert math.isfinite(entry['objective_before'])
        assert math.isfinite(entry['pose_increment_norm'])
        assert math.isfinite(entry['auxiliary_increment_norm'])


def test_malformed_step_is_not_disguised_as_a_scientific_rejection(modules, monkeypatch):
    np, candidate = modules
    monkeypatch.setattr(candidate, 'joint_step',
                        lambda np_, r, J, u, c, damping: (np_.zeros(4), np_.zeros(len(r))))
    with pytest.raises(ValueError):
        candidate.polish(np, *_scene(np), 'gm_hq_all25')


def test_gc_only_population_and_native_style_scale_weights_remain_explicit(modules):
    np, candidate = modules
    data = _scene(np)
    x0, x1, R0, t0, mask, fbar = candidate.validate_inputs(np, *data)
    residual = _independent_residual(np, x0, x1, R0, t0, fbar)
    gc_only = candidate.polish(np, *data, 'gm_hq_gc25')
    assert gc_only['selected_rows'] == int(mask.sum())
    assert abs(gc_only['initial_objective'] - .5 * float(residual[mask] @ residual[mask])) < 1e-9
    native = candidate.polish(np, *data, 'ba_continue_time', wall_seconds=0., clock=lambda: 0.)
    static = np.maximum(0., 1 - residual[mask] ** 2 / 1.5 ** 2)
    expected = .5 * float(np.sum(static * np.minimum(residual[mask] ** 2, fbar ** 2)))
    assert abs(native['initial_objective'] - expected) < 1e-9
    assert np.allclose(native['final_auxiliary'][mask], np.sqrt(static), atol=1e-12)
    assert np.array_equal(native['final_auxiliary'][~mask], np.zeros(int((~mask).sum())))
    assert native['native_style_msac_selected'] is False  # Exact tie retains seed.
    assert np.array_equal(native['R'], R0) and np.array_equal(native['t'], t0)


def test_zero_time_allocation_does_not_fill_time_or_take_one_extra_attempt(modules):
    np, candidate = modules
    calls = []

    def clock():
        calls.append(None)
        return .01 * (len(calls) - 1)

    result = candidate.polish(np, *_scene(np), 'ba_continue_time', wall_seconds=0., clock=clock)
    assert result['attempts'] == result['accepted_steps'] == 0
    assert result['trace'] == [] and result['reason'] == 'time_allowance'
    assert result['time_allowance_seconds'] == 0.
    assert result['time_overrun_seconds'] == result['elapsed_seconds']
    assert len(calls) <= 4


def test_time_control_requires_its_own_finite_allowance(modules):
    np, candidate = modules
    for allowance in (None, -1., math.inf, math.nan):
        with pytest.raises(ValueError):
            candidate.polish(np, *_scene(np), 'ba_continue_time', wall_seconds=allowance)
    with pytest.raises(ValueError):
        candidate.polish(np, *_scene(np), 'gm_hq_all25', wall_seconds=1.)


def test_time_control_has_exact_100_attempt_ceiling_even_with_continued_progress(modules, monkeypatch):
    np, candidate = modules
    data = _scene(np)

    def monotone_policy(np_, x0, x1, R, t, fbar, need_jacobian=True):
        # Deliberate policy model with continued strict improvement; physical
        # Sampson derivatives are checked independently in the camera fixture.
        r = np_.full(len(x0), .5 - .01 * float(R[2, 1]))
        J = None
        if need_jacobian:
            J = np_.zeros((len(x0), 5))
            J[:, 0] = -.01
        return r, J, candidate.tangent_basis(np_, t)

    solve_calls = []

    def small_pose_step(matrix, rhs):
        solve_calls.append(None)
        assert matrix.shape == (5, 5) and rhs.shape == (5,)
        return np.array([1e-5, 0., 0., 0., 0.])

    monkeypatch.setattr(candidate, 'linearize', monotone_policy)
    monkeypatch.setattr(np.linalg, 'solve', small_pose_step)
    result = candidate.polish(np, *data, 'ba_continue_time',
                              wall_seconds=1., clock=lambda: 0.)
    assert result['attempts'] == result['accepted_steps'] == 100
    assert len(result['trace']) == len(solve_calls) == 100
    assert result['reason'] == 'attempt_limit'
    assert result['native_style_msac_selected'] is True


def test_permuting_correspondences_preserves_hq_pose_and_reorders_weights(modules):
    np, candidate = modules
    data = _scene(np)
    order = np.arange(23, -1, -1)
    permuted = (data[0][order], data[1][order], *data[2:6], data[6][order])
    direct = candidate.polish(np, *data, 'gm_hq_all25')
    reversed_ = candidate.polish(np, *permuted, 'gm_hq_all25')
    assert np.allclose(direct['R'], reversed_['R'], atol=2e-7, rtol=0)
    assert np.allclose(direct['t'], reversed_['t'], atol=2e-7, rtol=0)
    assert np.allclose(direct['final_auxiliary'][order], reversed_['final_auxiliary'], atol=2e-6, rtol=0)
    assert abs(direct['final_gm_cost'] - reversed_['final_gm_cost']) < 1e-7


def test_invalid_initial_geometry_is_explicit_not_valid_seed_fallback(modules):
    np, candidate = modules
    p = np.zeros((8, 2), dtype=np.float32)
    result = candidate.polish(np, p, p.copy(), np.eye(3), np.eye(3), np.eye(3),
                              np.array([0., 0., 1.]), np.ones(8, dtype=np.uint8), 'gm_hq_all25')
    assert result['valid'] is False
    assert result['reason'] == 'invalid_initial_geometry'
    assert result['attempts'] == 0
    assert result['recovery_pending'] is True


def _fake_candidate(np, candidate, calls, invalid=False, fixed_residual=None):
    def polish(np_, p0, p1, K0, K1, R, t, mask, mode, config=None,
               wall_seconds=None, clock=None):
        calls.append((mode, wall_seconds))
        return dict(valid=not invalid, R=R.copy(), t=t.copy(),
                    E=_cross_matrix(np_, t) @ R, mode=mode, reason='synthetic_fixture',
                    attempts=0, accepted_steps=0, trace=[],
                    final_auxiliary=np_.ones(len(p0)), elapsed_seconds=0.,
                    initial_gm_cost=0., final_gm_cost=0., initial_objective=0.,
                    final_objective=0., native_calls=0, model_calls=0, recovery_pending=True)

    def linearize(np_, x0, x1, R, t, fbar, need_jacobian=True):
        if fixed_residual is None:
            return candidate.linearize(np_, x0, x1, R, t, fbar, need_jacobian)
        return np_.array(fixed_residual, dtype=np_.float64), None, candidate.tangent_basis(np_, t)

    return SimpleNamespace(DEFAULT_CONFIG=candidate.DEFAULT_CONFIG,
                           validate_inputs=candidate.validate_inputs,
                           linearize=linearize, polish=polish, skew=candidate.skew,
                           DegenerateGeometry=candidate.DegenerateGeometry)


def test_adapter_keeps_literal_seed_and_defers_new_pose_recovery(modules, adapter):
    np, candidate = modules
    p0, p1, K0, K1, R, t, mask = _scene(np)
    seed = dict(success=True, R=R.copy(), t=t.copy(), cheirality=18, sentinel='original')
    calls = []
    fake = _fake_candidate(np, candidate, calls)
    result = adapter.propose(np, fake, p0, p1, K0, K1, seed, mask)
    expected_order = ('gm_hq_all25', 'ba_continue25', 'gm_irls25', 'gm_hq_gc25', 'ba_continue_time')
    assert tuple(result['optimization_order']) == expected_order
    assert tuple(mode for mode, _ in calls) == expected_order
    assert calls[-1][1] is not None and calls[-1][1] >= 0
    assert calls[-1][1] == result['optimizer_timings']['gm_hq_all25']['outer_elapsed_seconds']
    assert result['optimizer_timings']['ba_continue_time']['wall_allowance_seconds'] == calls[-1][1]
    assert all(timing['exact_time_match_claim'] is False
               for timing in result['optimizer_timings'].values())
    original = result['arms']['improved']['original_seed']
    assert original['success'] is True and original['sentinel'] == 'original'
    assert original['R'].tobytes() == seed['R'].tobytes()
    assert original['t'].tobytes() == seed['t'].tobytes()
    assert not original['R'].flags.writeable and not original['t'].flags.writeable
    assert seed['R'].flags.writeable and seed['t'].flags.writeable
    for name in ('remask_only', *MODES):
        arm = result['arms'][name]
        assert arm['score_pending'] is True
        assert 'success' not in arm and 'cheirality' not in arm
        if arm['proposal_valid']:
            assert arm['production_recovery'] == 'PENDING'
            assert arm['failure_error_degrees'] is None
            assert not arm['E'].flags.writeable and not arm['mask'].flags.writeable


def test_adapter_common_mask_is_inclusive_and_retains_all_rows(modules, adapter):
    np, candidate = modules
    p0, p1, K0, K1, R, t, mask = _scene(np)
    known = [0., .5, 1., 1.000001, -1., -1.000001] * 4
    fake = _fake_candidate(np, candidate, [], fixed_residual=known)
    result = adapter.propose(np, fake, p0, p1, K0, K1,
                             dict(success=True, R=R, t=t), mask)
    expected = np.array([abs(value) <= 1 for value in known], dtype=np.uint8)
    for name in ('remask_only', *MODES):
        arm = result['arms'][name]
        assert arm['proposal_valid'] is True
        assert arm['mask'].shape == (24,) and np.array_equal(arm['mask'], expected)
        assert arm['production_recovery'] == 'PENDING'


def test_adapter_failed_seed_never_calls_optimizer_or_claims_recovery(modules, adapter):
    np, candidate = modules
    p0, p1, K0, K1, _, _, mask = _scene(np)
    calls = []
    fake = _fake_candidate(np, candidate, calls)
    seed = dict(success=False, R=np.eye(3), t=np.zeros(3), cheirality=0)
    result = adapter.propose(np, fake, p0, p1, K0, K1, seed, mask)
    assert calls == [] and tuple(result['optimization_order']) == ()
    assert result['arms']['improved']['original_seed']['success'] is False
    for name in ('remask_only', *MODES):
        arm = result['arms'][name]
        assert arm['proposal_valid'] is False and arm['failure_error_degrees'] == 180
        assert arm['production_recovery'] == 'NOT_ELIGIBLE_FAILED_SEED'


def test_adapter_invalid_proposal_does_not_fall_back_to_valid_seed(modules, adapter):
    np, candidate = modules
    p0, p1, K0, K1, R, t, mask = _scene(np)
    fake = _fake_candidate(np, candidate, [], invalid=True)
    result = adapter.propose(np, fake, p0, p1, K0, K1,
                             dict(success=True, R=R, t=t), mask)
    for name in MODES:
        arm = result['arms'][name]
        assert arm['proposal_valid'] is False and arm['failure_error_degrees'] == 180
        assert arm['production_recovery'] == 'NOT_ELIGIBLE_INVALID_PROPOSAL'

