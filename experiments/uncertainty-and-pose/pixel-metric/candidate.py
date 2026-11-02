"""Fixed three-attempt pose refinement; NumPy is supplied by the caller."""
import math
import time

ROTATION_ATOL = 1e-5

class GeometryError(ValueError):
 pass

def skew(np, v):
 x, y, z = v
 return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])

def unit(np, v):
 m = float(np.max(np.abs(v)))
 if not math.isfinite(m) or m == 0:
  raise GeometryError("zero_or_nonfinite_direction")
 w = v / m
 return w / math.sqrt(float(w @ w))

def basis(np, t):
 a = np.eye(3)[int(np.argmin(np.abs(t)))]
 b = unit(np, np.cross(t, a))
 return np.column_stack((b, np.cross(t, b)))

def _data(np, p0, p1, K0, K1):
 p0, p1 = np.array(p0, dtype=np.float64, copy=True), np.array(p1, dtype=np.float64, copy=True)
 k0, k1 = np.array(K0, dtype=np.float64, copy=True), np.array(K1, dtype=np.float64, copy=True)
 if p0.ndim != 2 or p0.shape[1:] != (2,) or p1.shape != p0.shape:
  raise ValueError("point_shape")
 for k in (k0, k1):
  if k.shape != (3, 3) or not np.isfinite(k).all():
   raise ValueError("calibration_shape_or_finite")
  if not np.array_equal(k[2], [0., 0., 1.]) or k[0, 0] <= 0 or k[1, 1] <= 0:
   raise ValueError("calibration_structure")
 if not np.isfinite(p0).all() or not np.isfinite(p1).all():
  raise ValueError("nonfinite_points")
 try:
  with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
   i0, i1 = np.linalg.inv(k0), np.linalg.inv(k1)
   h0 = np.column_stack((p0, np.ones(len(p0))))
   h1 = np.column_stack((p1, np.ones(len(p1))))
   x0, x1 = h0 @ i0.T, h1 @ i1.T
 except np.linalg.LinAlgError as e:
  raise ValueError("singular_calibration") from e
 if not all(np.isfinite(a).all() for a in (i0, i1, x0, x1)):
  raise ValueError("invalid_calibrated_points")
 fbar = float(k0[0, 0]/4 + k0[1, 1]/4 + k1[0, 0]/4 + k1[1, 1]/4)
 if not math.isfinite(fbar) or fbar <= 0:
  raise ValueError("invalid_fbar")
 return h0, h1, x0, x1, i0, i1, fbar

def _pose(np, R, t, failed=False):
 R, t = np.array(R, dtype=np.float64, copy=True), np.array(t, dtype=np.float64, copy=True)
 if R.shape != (3, 3) or t.shape != (3,):
  raise ValueError("pose_shape")
 if not np.isfinite(R).all() or not np.isfinite(t).all():
  if failed:
   raise ValueError("failed_seed_nonfinite")
  raise GeometryError("nonfinite_pose")
 if failed:
  return R, t
 with np.errstate(over="ignore", invalid="ignore"):
  proper = np.allclose(R.T @ R, np.eye(3), atol=ROTATION_ATOL, rtol=0) and abs(float(np.linalg.det(R)) - 1) <= ROTATION_ATOL
 if not proper:
  raise GeometryError("improper_rotation")
 return R, unit(np, t)

def residual_matrix(np, v0, v1, M, scale=1., dM=None):
 with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
  a, b = v0 @ M.T, v1 @ M
  q = np.sum(v1 * a, axis=1)
  D = np.sum(a[:, :2]**2 + b[:, :2]**2, axis=1)
  if not np.isfinite(D).all() or np.any(D <= 1e-24):
   raise GeometryError("undefined_denominator")
  root = np.sqrt(D)
  r = scale * q / root
  if not np.isfinite(r).all() or not np.isfinite(r*r).all():
   raise GeometryError("nonfinite_residual")
  J = None
  if dM is not None:
   cols = []
   for m in dM:
    da, db = v0 @ m.T, v1 @ m
    dq = np.sum(v1 * da, axis=1)
    half_dD = np.sum(a[:, :2]*da[:, :2] + b[:, :2]*db[:, :2], axis=1)
    cols.append(scale * (dq/root - q*half_dD/(D*root)))
   J = np.column_stack(cols)
   if not np.isfinite(J).all():
    raise GeometryError("nonfinite_jacobian")
 return r, J

def _linear(np, data, R, t, metric, jacobian):
 h0, h1, x0, x1, i0, i1, fbar = data
 B = basis(np, t)
 with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
  E = skew(np, t) @ R
  dm = None
  if jacobian:
   dm = [skew(np, t) @ skew(np, e) @ R for e in np.eye(3)]
   dm += [skew(np, B[:, j]) @ R for j in range(2)]
  if metric == "pixel":
   M, v0, v1, scale = i1.T @ E @ i0, h0, h1, 1.
   if dm is not None:
    dm = [i1.T @ e @ i0 for e in dm]
  else:
   M, v0, v1, scale = E, x0, x1, fbar
  r, J = residual_matrix(np, v0, v1, M, scale, dm)
 return r, J, B

def geometry(np, p0, p1, K0, K1, R, t, metric, need_jacobian=True):
 if metric not in ("native", "pixel"):
  raise ValueError("metric")
 data = _data(np, p0, p1, K0, K1)
 R, t = _pose(np, R, t)
 return _linear(np, data, R, t, metric, need_jacobian)

def normal_step(np, r, J, damping):
 with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
  H, g = J.T @ J, J.T @ r
  A = H + damping * np.diag(np.maximum(np.diag(H), 1e-9))
  if not np.isfinite(A).all() or not np.isfinite(g).all():
   raise GeometryError("nonfinite_normal_equation")
  d = np.linalg.solve(A, -g)
 if not np.isfinite(d).all():
  raise GeometryError("nonfinite_step")
 return d

def retract(np, R, t, delta, B=None):
 d = np.array(delta, dtype=np.float64, copy=True)
 if d.shape != (5,):
  raise ValueError("step_shape")
 if not np.isfinite(d).all():
  raise GeometryError("nonfinite_step")
 for s in (slice(0, 3), slice(3, 5)):
  n = float(np.linalg.norm(d[s]))
  if not math.isfinite(n):
   raise GeometryError("nonfinite_step_norm")
  if n > .1:
   d[s] *= .1/n
 w = d[:3]
 theta = float(np.linalg.norm(w))
 S = skew(np, w)
 c1 = math.sin(theta)/theta if theta else 1.
 c2 = (1-math.cos(theta))/(theta*theta) if theta >= 1e-8 else .5-theta*theta/24
 Q = np.eye(3) + c1*S + c2*(S @ S)
 B = basis(np, t) if B is None else B
 return Q @ R, unit(np, t + B @ d[3:]), d

def _cost(np, r):
 with np.errstate(over="ignore", invalid="ignore"):
  c = .5*float(r @ r)
 if not math.isfinite(c):
  raise GeometryError("nonfinite_cost")
 return c

def polish(np, p0, p1, K0, K1, R, t, gc_mask, metric, success=True):
 start = time.monotonic()
 if metric not in ("native", "pixel") or type(success) is not bool:
  raise ValueError("metric_or_success")
 data = _data(np, p0, p1, K0, K1)
 mask = np.asarray(gc_mask)
 if mask.shape != (len(data[0]),) or mask.dtype.kind not in "biu" or not np.isin(mask, [0, 1]).all():
  raise ValueError("binary_mask_shape")
 mask = np.array(mask, dtype=bool, copy=True)
 count, trace, accepted = int(mask.sum()), [], 0
 def finish(valid, reason, rr=None, tt=None, initial=None, final=None):
  return dict(valid=valid, reason=reason, R=None if rr is None else rr.copy(),
   t=None if tt is None else tt.copy(), E=None if rr is None else skew(np, tt) @ rr,
   selected_rows=count, attempts=len(trace), accepted_steps=accepted,
   initial_cost=initial, final_cost=final, trace=trace,
   elapsed_seconds=time.monotonic()-start)
 if not success:
  _pose(np, R, t, failed=True)
  return finish(False, "failed_seed")
 try:
  R, t = _pose(np, R, t)
  if count < 5:
   return finish(True, "insufficient_support", R, t)
  data = tuple(a[mask] if j < 4 else a for j, a in enumerate(data))
  r, _, _ = _linear(np, data, R, t, metric, True)
  current = initial = _cost(np, r)
 except GeometryError:
  return finish(False, "invalid_initial_geometry")
 damping = .001
 for attempt in range(3):
  before, trial_cost, trial_R, trial_t, trial_E, limited = current, None, None, None, None, None
  rejection, take, old_lambda = None, False, damping
  try:
   r, J, B = _linear(np, data, R, t, metric, True)
   raw = normal_step(np, r, J, damping)
   trial_R, trial_t, limited = retract(np, R, t, raw, B)
   trial_E = skew(np, trial_t) @ trial_R
   trial_r, _, _ = _linear(np, data, trial_R, trial_t, metric, False)
   trial_cost = _cost(np, trial_r)
   take = trial_cost < current
   if take:
    R, t, current = trial_R, trial_t, trial_cost
    accepted += 1
   else:
    rejection = "no_decrease"
  except np.linalg.LinAlgError:
   rejection = "singular_solve"
  except GeometryError:
   rejection = "invalid_trial_geometry"
  damping = max(1e-12, damping*.5) if take else min(1e12, damping*10)
  trace.append(dict(attempt=attempt, lambda_before=old_lambda, lambda_after=damping,
   cost_before=before, cost_trial=trial_cost, accepted=take, rejection=rejection,
   delta=None if limited is None else limited.copy(),
   R=None if trial_R is None else trial_R.copy(),
   t=None if trial_t is None else trial_t.copy(),
   E=None if trial_E is None else trial_E.copy()))
 return finish(True, "attempt_limit", R, t, initial, current)
