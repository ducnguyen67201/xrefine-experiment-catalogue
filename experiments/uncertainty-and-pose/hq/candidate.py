"""Known joint Geman--McClure HQ adaptation, SOURCE/NUMERICALLY UNTESTED.

Pritts and Koser, https://arxiv.org/html/2610.05743, is prior art.
NumPy is injected; this module never loads images, models, GT or native solvers.
The original GC/matcher/recovery implementation is not modified here.
"""
import math
import time


DEFAULT_CONFIG = dict(width=1., attempts=25, time_control_attempts=100,
    initial_lambda=.001, lambda_min=1e-12, lambda_max=1e12,
    damping_diagonal_floor=1e-9, gradient_tolerance=1e-10,
    denominator_floor=1e-24, rotation_step_max=.25,
    translation_step_max=.25, auxiliary_start=1., auxiliary_min=0.,
    auxiliary_max=1., readout_threshold=1., native_ba_scale='fbar',
    native_ba_weight_threshold=1.5)
MODES = ('ba_continue25', 'gm_irls25', 'gm_hq_gc25', 'gm_hq_all25',
         'ba_continue_time')


class DegenerateGeometry(ValueError):
    """A valid input has an undefined geometric residual; no rows are removed."""


def need(condition, message):
    if not condition:
        raise ValueError(message)


def finite_geometry(np, *values):
    if not all(np.isfinite(value).all() for value in values):
        raise DegenerateGeometry('Nonfinite geometric arithmetic; no clamping/removal')


def finite_cost(value):
    result = float(value)
    if not math.isfinite(result):
        raise DegenerateGeometry('Nonfinite objective arithmetic')
    return result


def skew(np, vector):
    x, y, z = vector
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]], dtype=np.float64)


def tangent_basis(np, t):
    axis = np.zeros(3, dtype=np.float64)
    axis[int(np.argmin(np.abs(t)))] = 1.
    first = np.cross(t, axis)
    first /= np.linalg.norm(first)
    return np.column_stack((first, np.cross(t, first)))


def validate_inputs(np, p0, p1, K0, K1, R, t, mask):
    points = [np.asarray(p) for p in (p0, p1)]
    need(all(p.ndim == 2 and p.shape[1] == 2 and p.dtype.kind == 'f' and
        p.dtype.itemsize in (4, 8) and np.isfinite(p).all() for p in points),
        'Finite unfiltered floating point coordinates')
    n = len(points[0])
    need(len(points[1]) == n and 0 <= n <= 10000, 'Aligned bounded population')
    m = np.asarray(mask)
    need(m.shape == (n,) and np.isin(m, (0, 1)).all(), 'Original binary GC mask')
    cameras = [np.asarray(k, dtype=np.float64) for k in (K0, K1)]
    need(all(k.shape == (3, 3) and np.isfinite(k).all() and
        k[0, 0] > 0 and k[1, 1] > 0 for k in cameras), 'Finite processed intrinsics')
    r = np.asarray(R, dtype=np.float64)
    translation = np.asarray(t, dtype=np.float64)
    need(r.shape == (3, 3) and translation.shape == (3,) and np.isfinite(r).all() and
        np.isfinite(translation).all() and np.allclose(r.T @ r, np.eye(3), atol=1e-5, rtol=0) and
        abs(float(np.linalg.det(r)) - 1) <= 1e-5, 'Proper finite production pose')
    length = float(np.linalg.norm(translation))
    need(length > 1e-15, 'Successful production translation is nonzero')
    normalized = []
    for p, k in zip(points, cameras):
        h = np.column_stack((p.astype(np.float64), np.ones(n))) @ np.linalg.inv(k).T
        need(np.isfinite(h).all() and (np.abs(h[:, 2]) > 1e-15).all(),
            'Valid homogeneous calibration; no clipping or row filtering')
        normalized.append(h / h[:, 2:3])
    fbar = .25 * sum(float(k[j, j]) for k in cameras for j in (0, 1))
    need(math.isfinite(fbar) and fbar > 0, 'Native threshold normalizer')
    return (*normalized, r.copy(), translation.copy() / length, m.astype(bool), fbar)


def linearize(np, x0, x1, R, t, fbar, need_jacobian=True):
    """Signed native-scale Sampson residual and local SO3 x S2 derivative."""
    basis = tangent_basis(np, t)
    E = skew(np, t) @ R
    l1, l0 = x0 @ E.T, x1 @ E
    numerator = np.sum(x1 * l1, axis=1)
    squared = np.sum(l1[:, :2] ** 2 + l0[:, :2] ** 2, axis=1)
    if not (np.isfinite(squared).all() and np.isfinite(numerator).all() and
            (squared > DEFAULT_CONFIG['denominator_floor']).all()):
        raise DegenerateGeometry('Undefined squared Sampson denominator; no clamping/removal')
    root = np.sqrt(squared)
    residual = fbar * numerator / root
    # Squared residuals are also required by every locked loss/normal equation.
    finite_geometry(np, residual, residual * residual)
    if not need_jacobian:
        return residual, None, basis
    axes = np.eye(3, dtype=np.float64)
    derivatives = [skew(np, t) @ skew(np, axes[j]) @ R for j in range(3)]
    derivatives += [skew(np, basis[:, j]) @ R for j in range(2)]
    J = np.empty((len(x0), 5), dtype=np.float64)
    for j, derivative in enumerate(derivatives):
        dl1, dl0 = x0 @ derivative.T, x1 @ derivative
        de = np.sum(x1 * dl1, axis=1)
        half_d_squared = np.sum(l1[:, :2] * dl1[:, :2] + l0[:, :2] * dl0[:, :2], axis=1)
        J[:, j] = fbar * (de / root - numerator * half_d_squared / (squared * root))
    if not (np.isfinite(residual).all() and np.isfinite(J).all()):
        raise DegenerateGeometry('Nonfinite geometric linearization')
    return residual, J, basis


def limited_step(np, step):
    actual = np.asarray(step, dtype=np.float64).copy()
    need(actual.shape == (5,), 'Five dimensional pose increment')
    finite_geometry(np, actual)
    for section, bound in ((slice(0, 3), .25), (slice(3, 5), .25)):
        length = float(np.linalg.norm(actual[section]))
        if not math.isfinite(length):
            raise DegenerateGeometry('Nonfinite pose increment norm')
        if length > bound:
            actual[section] *= bound / length
    return actual


def retract(np, R, t, step):
    actual = limited_step(np, step)
    w = actual[:3]
    angle = float(np.linalg.norm(w))
    W = skew(np, w)
    if angle < 1e-8:
        a = 1 - angle * angle / 6
        b = .5 - angle * angle / 24
    else:
        a, b = math.sin(angle) / angle, (1 - math.cos(angle)) / (angle * angle)
    updated_R = (np.eye(3) + a * W + b * (W @ W)) @ R
    updated_t = t + tangent_basis(np, t) @ actual[3:]
    updated_t /= np.linalg.norm(updated_t)
    return updated_R, updated_t


def joint_step(np, r, J, u, c, damping):
    """Damped joint GN step via a 5x5 Schur complement; no dense NxN block."""
    weighted_J = J * u[:, None]
    A = weighted_J.T @ weighted_J
    g = J.T @ (u * u * r)
    raw_D = r * r + c * c
    D = raw_D + damping * np.maximum(raw_D, 1e-9)
    gu = u * r * r + c * c * (u - 1)
    B = J * (u * r)[:, None]
    pose_damping = damping * np.maximum(np.diag(A), 1e-9)
    S = A + np.diag(pose_damping) - B.T @ (B / D[:, None])
    rhs = -g + B.T @ (gu / D)
    finite_geometry(np, S, rhs, D, B, gu)
    dp = np.linalg.solve(.5 * (S + S.T), rhs)
    du = -(gu + B @ dp) / D
    return dp, du


def gm_cost(np, r, c):
    square = r * r
    finite_geometry(np, square, c * c)
    return finite_cost(.5 * np.sum(c * c * (square / (c * c + square))))


def lifted_cost(np, r, u, c):
    return finite_cost(.5 * np.sum((u * r) ** 2 + c * c * (1 - u) ** 2))


def _cost(np, r, u, c, mode, static):
    if mode.startswith('gm_hq'):
        return lifted_cost(np, r, u, c)
    if mode == 'gm_irls25':
        return gm_cost(np, r, c)
    return finite_cost(.5 * np.sum(static * np.minimum(r * r, c * c)))


def polish(np, p0, p1, K0, K1, R, t, gc_mask, mode,
           config=DEFAULT_CONFIG, wall_seconds=None, clock=time.monotonic):
    began = clock()
    need(config == DEFAULT_CONFIG and mode in MODES, 'Exact predeclared configuration and arm')
    need((mode == 'ba_continue_time' and wall_seconds is not None and
        math.isfinite(wall_seconds) and wall_seconds >= 0) or
        (mode != 'ba_continue_time' and wall_seconds is None),
        'Time control requires its finite same-cell allowance; other arms have none')
    x0, x1, pose_R, pose_t, mask, fbar = validate_inputs(np, p0, p1, K0, K1, R, t, gc_mask)
    seed_R, seed_t = pose_R.copy(), pose_t.copy()
    n = len(x0)
    use = mask.copy() if mode in ('gm_hq_gc25', 'ba_continue25', 'ba_continue_time') else np.ones(n, dtype=bool)
    selected = int(use.sum())
    c = fbar if mode.startswith('ba_continue') else config['width']
    u = np.ones(selected, dtype=np.float64)
    trace = []
    accepted = 0
    attempts = 0
    damping = config['initial_lambda']
    residual_evaluations = 0
    jacobian_evaluations = 0
    rejected_geometry = 0
    solve_rejections = 0
    reason = 'attempt_limit'
    valid = True
    initial_gm = final_gm = initial_objective = final_objective = None
    static = np.ones(selected, dtype=np.float64)
    before_msac = None
    msac_selected = None
    geometry_phase = 'initial'
    try:
        finite_geometry(np, c * c)
        full_r, _, _ = linearize(np, x0, x1, pose_R, pose_t, fbar, False)
        residual_evaluations += 1
        initial_gm = gm_cost(np, full_r, 1.)
        before_msac = float(np.sum(np.minimum(full_r * full_r, 1.5 ** 2)))
        if mode.startswith('ba_continue'):
            static = np.maximum(0., 1 - full_r[use] ** 2 / (1.5 ** 2))
        if mode == 'gm_irls25':
            u = c * c / (c * c + full_r[use] ** 2)
        initial_objective = _cost(np, full_r[use], u, c, mode, static)
        geometry_phase = 'current'
        limit = config['time_control_attempts'] if mode == 'ba_continue_time' else config['attempts']
        if selected < 5:
            reason = 'insufficient_optimization_support'
        else:
            for attempt in range(limit):
                if wall_seconds is not None and clock() - began >= wall_seconds:
                    reason = 'time_allowance'
                    break
                full_r, full_J, _ = linearize(np, x0, x1, pose_R, pose_t, fbar)
                residual_evaluations += 1
                jacobian_evaluations += 1
                r, J = full_r[use], full_J[use]
                if mode == 'gm_irls25':
                    u = c * c / (c * c + r * r)
                cost_before = _cost(np, r, u, c, mode, static)
                if mode.startswith('gm_hq'):
                    weights = u * u
                    gu = u * r * r + c * c * (u - 1)
                else:
                    weights = u * u if mode == 'gm_irls25' else static * (r * r <= c * c)
                    gu = np.zeros(selected, dtype=np.float64)
                gradient = J.T @ (weights * r)
                finite_geometry(np, gradient, gu, weights)
                if max(float(np.max(np.abs(gradient))), float(np.max(np.abs(gu)))) <= config['gradient_tolerance']:
                    reason = 'gradient_tolerance'
                    break
                attempts += 1
                trial_cost = None
                dp_norm = du_norm = 0.
                did_accept = False
                bad_reason = None
                try:
                    if mode.startswith('gm_hq'):
                        dp, _ = joint_step(np, r, J, u, c, damping)
                        dp = limited_step(np, dp)
                        raw_D = r * r + c * c
                        D = raw_D + damping * np.maximum(raw_D, 1e-9)
                        B = J * (u * r)[:, None]
                        du = -(gu + B @ dp) / D
                        finite_geometry(np, du)
                        trial_u = np.clip(u + du, 0., 1.)
                    else:
                        A = J.T @ (weights[:, None] * J)
                        A += np.diag(damping * np.maximum(np.diag(A), 1e-9))
                        finite_geometry(np, A)
                        dp = limited_step(np, np.linalg.solve(A, -gradient))
                        trial_u = u.copy()
                    need(trial_u.shape == u.shape, 'Aligned auxiliary trial shape')
                    finite_geometry(np, dp, trial_u)
                    trial_R, trial_t = retract(np, pose_R, pose_t, dp)
                    candidate_full_r, _, _ = linearize(np, x0, x1, trial_R, trial_t, fbar, False)
                    residual_evaluations += 1
                    candidate_r = candidate_full_r[use]
                    trial_cost = _cost(np, candidate_r, trial_u, c, mode, static)
                    dp_norm = float(np.linalg.norm(dp))
                    du_norm = float(np.linalg.norm(trial_u - u))
                    did_accept = math.isfinite(trial_cost) and trial_cost < cost_before
                    if did_accept:
                        pose_R, pose_t, u = trial_R, trial_t, trial_u
                        accepted += 1
                except np.linalg.LinAlgError:
                    solve_rejections += 1
                    bad_reason = 'singular_linear_step'
                except DegenerateGeometry:
                    rejected_geometry += 1
                    bad_reason = 'undefined_trial_geometry'
                    trial_cost = None
                trace.append(dict(attempt=attempt, accepted=did_accept, damping=damping,
                    objective_before=cost_before, objective_trial=trial_cost,
                    pose_increment_norm=dp_norm, auxiliary_increment_norm=du_norm,
                    rejection=bad_reason))
                damping = max(config['lambda_min'], damping * .5) if did_accept else min(config['lambda_max'], damping * 10)
                if damping >= config['lambda_max']:
                    reason = 'damping_limit'
                    break
        geometry_phase = 'final'
        full_r, _, _ = linearize(np, x0, x1, pose_R, pose_t, fbar, False)
        residual_evaluations += 1
        if mode.startswith('ba_continue'):
            current_msac = float(np.sum(np.minimum(full_r * full_r, 1.5 ** 2)))
            msac_selected = current_msac < before_msac
            if not msac_selected:
                pose_R, pose_t = seed_R.copy(), seed_t.copy()
                full_r, _, _ = linearize(np, x0, x1, pose_R, pose_t, fbar, False)
                residual_evaluations += 1
        if mode == 'gm_irls25':
            u = c * c / (c * c + full_r[use] ** 2)
        elif mode.startswith('ba_continue'):
            u = np.sqrt(static)
        final_gm = gm_cost(np, full_r, 1.)
        final_objective = _cost(np, full_r[use], u, c, mode, static)
    except DegenerateGeometry:
        valid = False
        reason = 'invalid_'+geometry_phase+'_geometry'
    auxiliary = np.zeros(n, dtype=np.float64)
    auxiliary[use] = u
    elapsed = clock() - began
    overrun = max(0., elapsed - wall_seconds) if wall_seconds is not None else 0.
    return dict(valid=valid, R=pose_R.copy(), t=pose_t.copy(), E=skew(np, pose_t) @ pose_R,
        mode=mode, reason=reason, attempts=attempts, accepted_steps=accepted, trace=trace,
        selected_rows=selected, final_auxiliary=auxiliary,
        auxiliary_kind='joint_latent' if mode.startswith('gm_hq') else 'irls_sqrtweight' if mode=='gm_irls25' else 'static_seed_sqrtweight',
        initial_gm_cost=initial_gm, final_gm_cost=final_gm,
        initial_objective=initial_objective, final_objective=final_objective,
        residual_evaluations=residual_evaluations, jacobian_evaluations=jacobian_evaluations,
        solve_rejections=solve_rejections, rejected_trial_geometry=rejected_geometry,
        native_style_msac_selected=msac_selected, elapsed_seconds=elapsed,
        time_allowance_seconds=wall_seconds, time_overrun_seconds=overrun,
        rotation_matrix_movement=float(np.linalg.norm(pose_R-seed_R)),
        translation_vector_movement=float(np.linalg.norm(pose_t-seed_t)),
        native_calls=0, model_calls=0, recovery_pending=True)
