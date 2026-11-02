"""Independent AR reference. Source draft; execution requires later admission."""
import hashlib
import io
import json
import math
import os
from pathlib import Path

ARMS = ('stock', 'fast', 'ar', 'cosine', 'margin', 'spatial_margin')
FILTER_ARMS = ARMS[1:]
RESULT_ROOT = 'results/xrefine/ar-persistence-20261014/'
RESEARCH_ROOT = 'research/ar-persistence-20261014/'
PASSPORT_PIN = {
    'path': RESEARCH_ROOT + 'passport.json', 'bytes': 49939,
    'sha256': 'dba43942b4eae10514b1425021727ee02df6bfcc97923c1527e48921fd0ab857'}


def _json(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'),
                       ensure_ascii=True, allow_nan=False) + '\n').encode('ascii')


def _pin(name, raw):
    return dict(path=name, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def _read(context, pin, cap):
    assert set(pin) == {'path', 'bytes', 'sha256'}
    assert type(pin['bytes']) is int and 0 <= pin['bytes'] <= cap
    root = Path(context['root'])
    rel = Path(pin['path'])
    assert not rel.is_absolute() and '..' not in rel.parts
    current = root
    for part in rel.parts:
        current = current / part
        assert not current.is_symlink()
    before = current.stat()
    assert current.is_file() and before.st_nlink == 1
    assert before.st_size == pin['bytes']
    with current.open('rb') as stream:
        raw = stream.read(cap + 1)
    after = current.stat()
    assert (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) == (
        after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    assert _pin(pin['path'], raw) == pin
    return raw


def _document(context, pin, cap=1048576):
    raw = _read(context, pin, cap)
    value = json.loads(raw.decode('utf8'))
    assert _json(value) == raw
    return value


def _array_hash(value):
    return hashlib.sha256(value.tobytes(order='C')).hexdigest()


def _arrays(context, pin, fingerprints=None, cap=16777216):
    import numpy as np
    import zipfile
    raw = _read(context, pin, cap)
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        members = archive.infolist()
        assert 0 < len(members) <= 65536
        assert len({m.filename for m in members}) == len(members)
        assert all(m.filename.endswith('.npy') and '/' not in m.filename
                   and '\\' not in m.filename and len(m.filename) <= 128 and not m.flag_bits & 1
                   and m.file_size <= cap for m in members)
        assert sum(m.file_size for m in members) <= cap
    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        if fingerprints is not None:
            assert set(archive.files) == set(fingerprints)
        values = {name: archive[name] for name in archive.files}
    for name, value in values.items():
        assert type(value) is np.ndarray and not value.dtype.hasobject
        if fingerprints is not None:
            expected = fingerprints[name]
            assert value.dtype.str == expected['dtype']
            assert list(value.shape) == expected['shape']
            assert _array_hash(value) == expected['sha256']
    return values


def _record(context, pin):
    import numpy as np
    arrays = _arrays(context, pin)
    raw = arrays['record_json']
    assert raw.dtype == np.dtype('u1') and raw.ndim == 1
    assert raw.size <= 1048576
    text = raw.tobytes()
    value = json.loads(text.decode('ascii'))
    assert _json(value) == text
    return value, arrays


def _descriptor_cosines(d0, d1):
    import numpy as np
    a, b = np.asarray(d0), np.asarray(d1)
    assert a.ndim == b.ndim == 2 and a.shape[1] == b.shape[1]
    assert 2 <= len(a) <= 2048 and 2 <= len(b) <= 2048
    assert 1 <= a.shape[1] <= 64
    assert a.dtype.kind == b.dtype.kind == 'f'
    assert np.isfinite(a).all() and np.isfinite(b).all()
    a, b = a.astype(np.float64), b.astype(np.float64)
    na = np.sqrt(np.sum(a * a, axis=1))
    nb = np.sqrt(np.sum(b * b, axis=1))
    assert np.isfinite(na).all() and np.isfinite(nb).all()
    assert (na > 1e-12).all() and (nb > 1e-12).all()
    a /= na[:, None]
    b /= nb[:, None]
    cosine = a @ b.T
    assert np.isfinite(cosine).all()
    return cosine


def _native_inputs(cosine, source, target, raw_p0, hw0):
    import numpy as np
    source, target = np.asarray(source), np.asarray(target)
    points, hw = np.asarray(raw_p0), np.asarray(hw0)
    assert source.dtype.kind == target.dtype.kind == 'i'
    assert source.ndim == target.ndim == 1 and source.shape == target.shape
    assert len(source) <= 2048 and len(set(source.tolist())) == len(source)
    assert len(set(target.tolist())) == len(target)
    assert ((source >= 0) & (source < cosine.shape[0])).all()
    assert ((target >= 0) & (target < cosine.shape[1])).all()
    assert points.shape == (len(source), 2) and points.dtype.kind == 'f'
    assert np.isfinite(points).all()
    assert hw.shape == (2,) and hw.dtype.kind in 'iu'
    height, width = map(int, hw)
    assert 0 < height <= 8192 and 0 < width <= 8192
    assert (points[:, 0] >= 0).all() and (points[:, 0] < width).all()
    assert (points[:, 1] >= 0).all() and (points[:, 1] < height).all()
    return source, target, points, height, width


def _partial_assignment(scores, excluded=None):
    """Min-cost formulation with one inaccessible-to-others null per row."""
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    scores = np.asarray(scores, dtype=np.float64)
    assert scores.ndim == 2 and np.isfinite(scores).all()
    nr, nc = scores.shape
    assert 0 <= nr <= 64 and 0 <= nc <= 288
    if nr == 0:
        return 0.0, []
    cost = np.full((nr, nc + nr), np.inf, dtype=np.float64)
    cost[:, :nc] = -scores
    for row in range(nr):
        cost[row, nc + row] = 0.0
    if excluded is not None:
        row, column = excluded
        assert 0 <= row < nr and 0 <= column < nc
        cost[row, column] = np.inf
    rows, columns = linear_sum_assignment(cost)
    assert rows.tolist() == list(range(nr))
    real = [(int(r), int(c)) for r, c in zip(rows, columns) if c < nc]
    assert excluded not in real
    value = math.fsum(float(scores[r, c]) for r, c in real)
    assert math.isfinite(value)
    return value, real


def _brute_assignment(scores, excluded=None):
    """Small offline oracle; enumerates real choices plus private nulls."""
    nr = len(scores)
    nc = len(scores[0]) if nr else 0
    assert nr <= 5 and nc <= 6
    assert all(len(row) == nc for row in scores)
    assert all(math.isfinite(float(v)) for row in scores for v in row)
    best, witnesses = -math.inf, []

    def walk(row, used, selected):
        nonlocal best, witnesses
        if row == nr:
            value = math.fsum(float(scores[r][c]) for r, c in selected)
            if value > best:
                best, witnesses = value, [tuple(selected)]
            elif value == best:
                witnesses.append(tuple(selected))
            return
        walk(row + 1, used, selected)
        for column in range(nc):
            if column not in used and (row, column) != excluded:
                walk(row + 1, used | {column}, selected + [(row, column)])

    walk(0, set(), [])
    return best, witnesses


def _clamp_delta(value):
    assert math.isfinite(value) and value >= -1e-10
    return max(0.0, value)


def filter_pair(d0, d1, source_indices, target_indices, raw_p0, hw0):
    """Independent masks and objectives; no producer selection code is used."""
    import numpy as np
    cosine = _descriptor_cosines(d0, d1)
    source, target, points, height, width = _native_inputs(
        cosine, source_indices, target_indices, raw_p0, hw0)
    count = len(source)
    similarities, margins = [], []
    for i, j in zip(source, target):
        i, j = int(i), int(j)
        value = float(cosine[i, j])
        row_other = max(float(np.max(cosine[i, :j])) if j else -math.inf,
                        float(np.max(cosine[i, j + 1:])) if j + 1 < cosine.shape[1] else -math.inf)
        col_other = max(float(np.max(cosine[:i, j])) if i else -math.inf,
                        float(np.max(cosine[i + 1:, j])) if i + 1 < cosine.shape[0] else -math.inf)
        similarities.append(value)
        margins.append(min(value - row_other, value - col_other))
    anchors = sorted(range(count), key=lambda k: (
        margins[k], int(source[k]), int(target[k])))[:32]
    rows = set(int(source[k]) for k in anchors)
    for k in anchors:
        i, j = int(source[k]), int(target[k])
        competitors = cosine[:, j].copy()
        competitors[i] = -np.inf
        rows.add(int(np.argmax(competitors)))
    rows = sorted(rows)
    columns = set(int(target[k]) for k in anchors)
    target_ids = np.arange(cosine.shape[1], dtype=np.int64)
    for i in rows:
        columns.update(int(j) for j in np.lexsort((target_ids, -cosine[i]))[:4])
    columns = sorted(columns)
    assert len(rows) <= 64 and len(columns) <= 288
    pooled = cosine[np.ix_(np.asarray(rows, dtype=np.int64),
                           np.asarray(columns, dtype=np.int64))]
    optimum, assignment = _partial_assignment(pooled)
    solves = 1 if rows else 0
    chosen = set(assignment)
    row_ids, column_ids = {v: i for i, v in enumerate(rows)}, {v: i for i, v in enumerate(columns)}
    deltas = []
    for k in anchors:
        edge = (row_ids[int(source[k])], column_ids[int(target[k])])
        if edge not in chosen:
            deltas.append(0.0)
        else:
            without, _ = _partial_assignment(pooled, edge)
            solves += 1
            deltas.append(_clamp_delta(optimum - without))
    assert solves <= 33
    masks = {arm: [True] * count for arm in FILTER_ARMS}
    removed = [k for k, delta in zip(anchors, deltas) if not delta > 0.05 + 1e-10]
    for k in removed:
        masks['ar'][k] = False
    q = len(removed)
    bins = [min(3, int(math.floor(4 * float(p[0]) / width)))
            + 4 * min(3, int(math.floor(4 * float(p[1]) / height))) for p in points]
    q_bins = [sum(bins[k] == b for k in removed) for b in range(16)]
    for arm, key in (('cosine', similarities), ('margin', margins)):
        order = sorted(anchors, key=lambda k: (key[k], int(source[k]), int(target[k])))
        for k in order[:q]:
            masks[arm][k] = False
    for b in range(16):
        order = sorted((k for k in anchors if bins[k] == b), key=lambda k: (
            margins[k], int(source[k]), int(target[k])))
        assert q_bins[b] <= len(order)
        for k in order[:q_bins[b]]:
            masks['spatial_margin'][k] = False
    assert all(sum(not v for v in masks[a]) == q for a in FILTER_ARMS[1:])
    return dict(anchors=anchors, rows=rows, cols=columns, margin=margins,
                cosine=similarities, deltas=deltas, masks=masks, q=q,
                q_bins=q_bins, solves=solves, optimum=optimum,
                assignment=[[rows[r], columns[c]] for r, c in assignment])


def normalize(points, camera):
    import numpy as np
    points, camera = np.asarray(points), np.asarray(camera)
    assert points.ndim == 2 and points.shape[1] == 2 and points.dtype.kind == 'f'
    assert camera.shape == (3, 3) and camera.dtype.kind == 'f'
    assert np.isfinite(points).all() and np.isfinite(camera).all()
    camera = camera.astype(np.float64)
    assert abs(float(np.linalg.det(camera))) > 1e-12
    rays = np.column_stack((points.astype(np.float64), np.ones(len(points)))) @ np.linalg.inv(camera).T
    assert np.isfinite(rays).all() and (np.abs(rays[:, 2]) > 1e-12).all()
    result = rays[:, :2] / rays[:, 2:3]
    assert np.isfinite(result).all()
    return np.ascontiguousarray(result, dtype=np.float64)


def _proper(rotation):
    import numpy as np
    rotation = np.asarray(rotation)
    return (rotation.shape == (3, 3) and rotation.dtype.kind == 'f'
            and bool(np.isfinite(rotation).all())
            and bool(np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5, rtol=0))
            and abs(float(np.linalg.det(rotation)) - 1.0) <= 1e-5)


def pose_error(transform, rotation, translation):
    import numpy as np
    gt = np.asarray(transform)
    assert gt.shape == (4, 4) and np.isfinite(gt).all()
    assert np.array_equal(gt[3], np.array([0., 0., 0., 1.]))
    assert _proper(gt[:3, :3])
    assert _proper(rotation)
    t = np.asarray(translation, dtype=np.float64).reshape(-1)
    assert t.shape == (3,) and np.isfinite(t).all()
    expected = gt[:3, 3]
    nt, ne = float(np.linalg.norm(t)), float(np.linalg.norm(expected))
    relative = np.asarray(rotation) @ gt[:3, :3].T
    angle_r = math.degrees(math.acos(max(-1.0, min(1.0, (float(np.trace(relative)) - 1.0) / 2.0))))
    assert math.isfinite(nt) and math.isfinite(ne)
    if nt * ne <= 2.220446049250313e-16:
        return 180.0, angle_r, 180.0
    cosine = abs(float(np.dot(t / nt, expected / ne)))
    angle_t = math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
    assert math.isfinite(angle_r) and math.isfinite(angle_t)
    return max(angle_r, angle_t), angle_r, angle_t


def auc(errors, thresholds=(5, 10, 20)):
    """Preserved endpoint rule: constant recall from last error to threshold."""
    import bisect
    values = sorted(float(value) for value in errors)
    assert values and all(math.isfinite(v) and 0 <= v <= 180 for v in values)
    errors0 = [0.0] + values
    recall = [0.0] + [(i + 1) / len(values) for i in range(len(values))]
    result = []
    for threshold in thresholds:
        threshold = float(threshold)
        assert math.isfinite(threshold) and threshold > 0
        k = bisect.bisect_left(errors0, threshold)
        x, y = errors0[:k] + [threshold], recall[:k] + [recall[k - 1]]
        area = math.fsum((x[i] - x[i - 1]) * (y[i] + y[i - 1]) / 2
                         for i in range(1, len(x)))
        result.append(100.0 * area / threshold)
    return result


def _close_values(actual, expected, atol=1e-10):
    import numpy as np
    a, b = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
    assert a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
    assert np.allclose(a, b, atol=atol, rtol=0)


def _compare_filter(actual, expected):
    assert type(actual) is dict
    assert set(actual) == {'anchors', 'rows', 'cols', 'cosine', 'margin', 'deltas',
        'optimum', 'assignment_columns', 'masks', 'q', 'q_bins', 'timings_ns', 'solves'}
    for name in ('anchors', 'rows', 'cols', 'assignment_columns', 'q_bins'):
        assert type(actual[name]) is list and all(type(v) is int for v in actual[name])
    for name in ('q', 'solves'):
        assert type(actual[name]) is int
    for name in ('cosine', 'margin', 'deltas'):
        assert type(actual[name]) is list
        assert all(type(v) is float and math.isfinite(v) for v in actual[name])
        _close_values(actual[name], expected[name])
    assert type(actual['optimum']) is float and math.isfinite(actual['optimum'])
    assert type(actual['masks']) is dict and set(actual['masks']) == set(FILTER_ARMS)
    for values in actual['masks'].values():
        assert type(values) is list and len(values) == len(expected['cosine'])
        assert all(type(v) is bool for v in values)
    assert len(actual['q_bins']) == 16 and sum(actual['q_bins']) == actual['q']
    for name in ('anchors', 'rows', 'cols', 'masks', 'q', 'q_bins'):
        assert actual[name] == expected[name], name
    _close_values([actual['optimum']], [expected['optimum']])
    assert 1 <= actual['solves'] <= 33


def _brute_filter_deltas(result, d0, d1, source, target):
    cosine = _descriptor_cosines(d0, d1)
    rows, cols = result['rows'], result['cols']
    matrix = [[float(cosine[i, j]) for j in cols] for i in rows]
    value, _ = _brute_assignment(matrix)
    _close_values([result['optimum']], [value], 1e-12)
    truth = []
    for k in result['anchors']:
        edge = (rows.index(int(source[k])), cols.index(int(target[k])))
        other, _ = _brute_assignment(matrix, edge)
        truth.append(_clamp_delta(value - other))
    _close_values(result['deltas'], truth, 1e-12)


def _bad(call):
    try:
        call()
    except (AssertionError, ValueError, TypeError, IndexError):
        return
    raise AssertionError('Malformed fixture was accepted')


def offline(producer_module, context):
    """Fabricated fixtures, then label-free real loader validation; no real filter/pose."""
    import numpy as np
    checks = []

    def passed(name):
        checks.append(dict(id=name, status='PASS'))

    matrices = [[], [[-1.0, -2.0], [-3.0, -4.0]],
                [[0.0, 0.0], [0.0, 0.0]],
                [[1.0, 1.0], [1.0, 1.0]],
                [[0.7, 0.6, -0.2], [0.6, 0.1, 0.0]],
                [[0.05, -1.0], [-1.0, 1.0]]]
    for fixture in matrices:
        value = np.array(fixture, dtype=np.float64).reshape((0, 0)) if not fixture else np.array(fixture, dtype=np.float64)
        optimum, witness = _partial_assignment(value)
        brute, witnesses = _brute_assignment(fixture)
        assert abs(optimum - brute) <= 1e-12 and tuple(witness) in witnesses
        for row in range(value.shape[0]):
            for column in range(value.shape[1]):
                without, other = _partial_assignment(value, (row, column))
                truth, alternatives = _brute_assignment(fixture, (row, column))
                assert abs(without - truth) <= 1e-12 and tuple(other) in alternatives
    passed('enumerated_partial_assignment_negative_ties_nulls')
    assert not 0.05 > 0.05 + 1e-10
    assert not (0.05 + 0.5e-10) > 0.05 + 1e-10
    assert (0.05 + 2e-10) > 0.05 + 1e-10
    assert _clamp_delta(-1e-10) == 0 and _clamp_delta(0.0) == 0
    _bad(lambda: _clamp_delta(-1.0001e-10))
    passed('epsilon_and_negative_delta_boundaries')
    epsilon_sources = np.array([[1., 0., 0.], [0., 1., 0.]], dtype=np.float64)
    epsilon_ids = np.arange(2, dtype=np.int64)
    epsilon_points = np.array([[10., 10.], [20., 20.]], dtype=np.float32)
    epsilon_hw = np.array([100, 100], dtype=np.int64)
    for a, keep in ((.05, False), (.05 + .5e-10, False), (.05 + 2e-10, True)):
        epsilon_targets = np.array([[a, 0., math.sqrt(1 - a*a)], [0., 1., 0.]])
        result = producer_module.filter_pair(epsilon_sources, epsilon_targets,
            epsilon_ids, epsilon_ids, epsilon_points, epsilon_hw)
        expected = filter_pair(epsilon_sources, epsilon_targets,
            epsilon_ids, epsilon_ids, epsilon_points, epsilon_hw)
        _compare_filter(result, expected)
        _brute_filter_deltas(result, epsilon_sources, epsilon_targets, epsilon_ids, epsilon_ids)
        assert result['masks']['ar'][0] is keep
    passed('producer_realizable_epsilon_threshold_sides')
    d0 = np.array([[1., 0.], [0.8, 0.6], [0., 1.], [-1., 0.]], dtype=np.float32)
    d1 = np.array([[1., 0.], [0.6, 0.8], [0., 1.], [0., -1.]], dtype=np.float32)
    source = np.array([2, 0, 3, 1], dtype=np.int64)
    target = np.array([2, 0, 3, 1], dtype=np.int64)
    points = np.array([[90., 90.], [10., 10.], [10., 90.], [90., 10.]], dtype=np.float32)
    hw = np.array([100, 100], dtype=np.int64)
    reference = filter_pair(d0, d1, source, target, points, hw)
    actual = producer_module.filter_pair(d0.copy(), d1.copy(), source.copy(), target.copy(), points.copy(), hw.copy())
    _compare_filter(actual, reference)
    malformed = json.loads(json.dumps(actual))
    malformed['masks']['fast'][0] = 1
    _bad(lambda: _compare_filter(malformed, reference))
    malformed = json.loads(json.dumps(actual))
    malformed['anchors'][0] = bool(malformed['anchors'][0])
    _bad(lambda: _compare_filter(malformed, reference))
    _brute_filter_deltas(actual, d0, d1, source, target)
    assert reference['rows'] == [0, 1, 2, 3] and reference['cols'] == [0, 1, 2, 3]
    assert set(source[reference['anchors']].tolist()) <= set(reference['rows'])
    assert set(target[reference['anchors']].tolist()) <= set(reference['cols'])
    for arm, mask in reference['masks'].items():
        kept = np.flatnonzero(mask).tolist()
        assert kept == sorted(kept)
        if arm != 'fast':
            assert sum(not v for v in mask) == reference['q']
    bins = [15, 0, 12, 3]
    removed = [i for i, keep in enumerate(reference['masks']['ar']) if not keep]
    spatial = [i for i, keep in enumerate(reference['masks']['spatial_margin']) if not keep]
    assert [sum(bins[i] == b for i in removed) for b in range(16)] == [sum(bins[i] == b for i in spatial) for b in range(16)]
    passed('competitors_pool_mates_equal_counts_bins_native_order')
    # Deliberately non-MNN native edges isolate graph-construction operator
    # coverage; these fabricated edges make no real-correspondence claim.
    outside_sources = np.tile(np.array([[1., 0.]], dtype=np.float32), (5, 1))
    outside_targets = np.array([[1., 0.]] * 5 + [[0., 1.], [-1., 0.]], dtype=np.float32)
    outside_i, outside_j = np.array([3, 4]), np.array([5, 6])
    outside_points = np.array([[10., 10.], [90., 90.]], dtype=np.float32)
    outside = producer_module.filter_pair(outside_sources, outside_targets,
        outside_i, outside_j, outside_points, hw)
    expected_outside = filter_pair(outside_sources, outside_targets,
        outside_i, outside_j, outside_points, hw)
    _compare_filter(outside, expected_outside)
    _brute_filter_deltas(outside, outside_sources, outside_targets, outside_i, outside_j)
    assert outside['rows'] == [0, 3, 4] and outside['cols'] == [0, 1, 2, 3, 5, 6]
    passed('outside_competitor_and_explicit_mates_outside_top4')
    tie0 = np.array([[1., 0.], [1., 0.], [0., 1.], [0., 1.]], dtype=np.float32)
    tie1 = np.array([[1., 0.], [1., 0.], [0., 1.], [0., 1.]], dtype=np.float32)
    tied = filter_pair(tie0, tie1, source, target, points, hw)
    _compare_filter(producer_module.filter_pair(tie0, tie1, source, target, points, hw), tied)
    _brute_filter_deltas(tied, tie0, tie1, source, target)
    assert tied['anchors'] == [1, 3, 0, 2]
    assert tied['deltas'] == [0.0] * 4
    passed('global_ties_alternative_optima_deterministic_ids')
    negative = -np.eye(2, dtype=np.float32)
    identity = np.eye(2, dtype=np.float32)
    ids = np.arange(2, dtype=np.int64)
    small_points = points[:2]
    rejected = producer_module.filter_pair(identity, negative, ids, ids, small_points, hw)
    _compare_filter(rejected, filter_pair(identity, negative, ids, ids, small_points, hw))
    _brute_filter_deltas(rejected, identity, negative, ids, ids)
    assert rejected['q'] == 2 and rejected['deltas'] == [0., 0.]
    passed('negative_native_edges_cannot_borrow_nulls')
    broken = d0.copy(); broken[0] = 0
    nonfinite = d0.copy(); nonfinite[0, 0] = np.nan
    bad_calls = [lambda module: module.filter_pair(broken, d1, source, target, points, hw),
                 lambda module: module.filter_pair(nonfinite, d1, source, target, points, hw),
                 lambda module: module.filter_pair(d0, d1, np.array([0, 0]), np.array([0, 1]), points[:2], hw),
                 lambda module: module.filter_pair(d0, d1, source, target, points[:, :1], hw)]
    for call in bad_calls:
        _bad(lambda: call(producer_module))
    passed('malformed_descriptors_ids_points_rejected')
    camera = np.array([[2., 0.5, 3.], [0., 4., 5.], [0., 0., 1.]])
    pixel = np.array([[3., 5.], [6., 13.]], dtype=np.float64)
    expected = np.array([[0., 0.], [1., 2.]])
    _close_values(normalize(pixel, camera), expected, 1e-12)
    _close_values(producer_module.normalize(pixel, camera), expected, 1e-12)
    _close_values(auc([0., 180.]), [50., 50., 50.], 1e-12)
    _close_values(producer_module.auc([0., 180.]), [50., 50., 50.], 1e-12)
    _close_values(auc([1., 3., 180.]), [50., 175. / 3., 62.5], 1e-12)
    _close_values(producer_module.auc([1., 3., 180.]), auc([1., 3., 180.]), 1e-12)
    _bad(lambda: auc([float('nan')]))
    _bad(lambda: auc([float('inf')]))
    passed('full_camera_normalization_auc_failure_denominator')
    gt = np.eye(4); gt[0, 3] = 1
    _close_values(pose_error(gt, np.eye(3), np.array([-1., 0., 0.])), [0., 0., 0.], 1e-12)
    _close_values(producer_module.pose_error(gt, np.eye(3), np.array([-1., 0., 0.])), [0., 0., 0.], 1e-12)
    assert max(pose_error(gt, np.eye(3), np.zeros(3))) == 180
    passed('directed_pose_sign_symmetry_zero_translation')
    ordinary = (None, np.zeros(7, dtype=bool))
    assert _validate_gc(ordinary, 7)[0] is None
    _bad(lambda: _validate_gc((None, None), 7))
    _bad(lambda: _validate_gc((None, np.zeros(6, dtype=bool)), 7))
    passed('author_none_E_still_has_complete_mask_schema')
    import inspect
    assert tuple(inspect.signature(producer_module.filter_pair).parameters) == (
        'd0', 'd1', 'source_indices', 'target_indices', 'raw_p0', 'hw0')
    assert tuple(inspect.signature(filter_pair).parameters) == (
        'd0', 'd1', 'source_indices', 'target_indices', 'raw_p0', 'hw0')
    passed('API_label_argument_exclusion_source_review_required')
    checks.extend(_fake_native_fixtures(producer_module))
    checks.extend(producer_module.loader_fixtures(load_pair))
    passport = _document(context, PASSPORT_PIN)
    assert len(passport['pairs']) == 8
    rows = []
    for pair in passport['pairs']:
        expected = load_pair(context, pair)
        actual = producer_module.load_pair(context['root'], pair)
        for a, b in zip(actual, expected):
            assert set(a) == set(b)
            assert all(a[k].dtype == b[k].dtype and a[k].shape == b[k].shape
                       and a[k].tobytes() == b[k].tobytes() for k in a)
        counts = {}
        for side in ('0', '1'):
            height, width = map(int, expected[1]['hw' + side])
            for bank, key in ((0, 'keypoints'), (2, 'p'), (3, 'p')):
                p = expected[bank][key + side]
                x = (p[:, 0] < 0) | (p[:, 0] >= width)
                y = (p[:, 1] < 0) | (p[:, 1] >= height)
                counts[f'{bank}:{key}{side}'] = dict(rows=int((x | y).sum()),
                    x=int(x.sum()), y=int(y.sum()))
        rows.append(dict(pair_index=pair['pair_index'], outside_raster=counts))
    return dict(schema='ar-offline-v1', status='PASS', complete=True,
        checks=checks, scientific_execution=False, real_input_validation=dict(
            status='LABEL_FREE_REAL_INPUT_VALIDATION', pairs=rows, bank_loads=64,
            real_filter_calls=0, native_calls=0, gt_read=False))


def _fake_native_fixtures(producer):
    """Python substitutes only; never call real GC or recovery implementations."""
    import numpy as np
    import sys
    import types
    absent = object()
    previous = {name: sys.modules.get(name, absent) for name in ('cv2', 'pygcransac')}
    fake_cv = types.ModuleType('cv2')
    fake_gc = types.ModuleType('pygcransac')
    options = {}
    size = 6
    points = np.array([[1., 1.], [2., 1.], [1., 2.], [2., 2.], [3., 1.], [1., 3.]])
    camera = dict(K0=np.eye(3), K1=np.eye(3),
                  hw0=np.array([100, 100]), hw1=np.array([100, 100]))
    checks = []

    class UnencodableFixture:
        pass

    def gc(*args, **kwargs):
        assert len(args) == 7 and args[0].shape == (size, 4)
        assert kwargs == dict(probabilities=[], threshold=1., conf=.99999,
                              max_iters=1000, min_iters=1000, sampler=0)
        if options['gc_throw']:
            raise RuntimeError('fabricated GC throw')
        return options['gc']

    def recovery(*args, **kwargs):
        assert len(args) == 5 and args[4] == 1e9 and set(kwargs) == {'mask'}
        mask = kwargs['mask']
        if options['recovery_throw']:
            mask[0, 0] = 0
            raise RuntimeError('fabricated recovery throw')
        count = 0 if options['zero_count'] else size
        mask[:] = 0 if count == 0 else options.get('encoding', 255)
        fifth = UnencodableFixture() if options['bad_fifth'] else np.full((4, size), np.nan)
        return count, np.eye(3), np.array([[1.], [0.], [0.]]), mask.copy(), fifth

    def run(raw, gc_throw=False, recovery_throw=False, zero_count=False, bad_fifth=False):
        options.update(gc=raw, gc_throw=gc_throw, recovery_throw=recovery_throw,
                       zero_count=zero_count, bad_fifth=bad_fifth)
        record, arrays, error = dict(draw=0, first_arm_position=0), {}, None
        try:
            producer.fit(points.copy(), points.copy(), camera, arrays, 'offline_fake', record)
        except (ValueError, AssertionError, TypeError, RuntimeError) as failure:
            error = type(failure).__name__
        return record, arrays, error

    def mark(name):
        checks.append(dict(id=name, status='PASS'))

    try:
        fake_gc.findEssentialMatrix = gc
        fake_cv.recoverPose = recovery
        sys.modules['pygcransac'], sys.modules['cv2'] = fake_gc, fake_cv
        none_raw = (None, np.zeros(size, dtype=bool))
        rec, arrays, error = run(none_raw)
        assert error is None and rec['complete'] is True and rec['status'] == 'FAILED'
        assert rec['selected_index'] is None and rec['recoveries'] == []
        assert rec['gc_python_call_attempted'] is True and rec['gc_python_call_returned'] is True
        assert type(rec['gc_ns']) is int and rec['gc_ns'] >= 0
        assert rec['gc_raw_capture'] == 'LOSSLESS' and rec['gc_outer_facts'] == _outer_facts(none_raw)
        restored = _tag(rec['gc_raw'], arrays, set())
        assert restored[0] is None and np.array_equal(restored[1], none_raw[1])
        mark('fake_author_None_E_retains_mask_and_ordinary_failure')
        rec, arrays, error = run((None, None))
        assert error is not None and rec['complete'] is False
        assert rec['gc_raw_capture'] == 'LOSSLESS' and rec['gc_outer_facts']['arity'] == 2
        mark('fake_None_E_malformed_mask_rejected_after_raw_capture')
        for raw in ((UnencodableFixture(), np.ones(size, dtype=bool)),
                    (np.array([[UnencodableFixture()]], dtype=object), np.ones(size, dtype=bool))):
            rec, arrays, error = run(raw)
            assert error is not None and rec['complete'] is False
            assert rec['gc_python_call_returned'] is True
            assert rec['gc_outer_facts'] == _outer_facts(raw)
            assert rec['gc_outer_facts']['arity'] == 2
            assert rec['gc_raw_capture'] == 'UNENCODABLE_CONTENT_UNKNOWN'
        mark('fake_unencodable_GC_outer_facts_and_unknown_content')
        rec, arrays, error = run((None,) * 8)
        assert error is not None and rec['complete'] is False
        assert rec['gc_outer_facts']['arity'] == 8 and rec['gc_raw_capture'] == 'LOSSLESS'
        assert len(rec['gc_raw']['items']) == 8
        mark('fake_malformed_outer_arity_retained')
        rec, arrays, error = run(None, gc_throw=True)
        assert error is not None and rec['complete'] is False
        assert rec['gc_python_call_attempted'] is True and rec['gc_python_call_returned'] is False
        assert type(rec['gc_ns']) is int and rec['gc_ns'] >= 0
        assert rec['gc_raw_capture'] == 'NO_RETURN_OBSERVED'
        mark('fake_GC_throw_records_attempt_and_finally_timing')
        stacked = (np.tile(np.eye(3), (2, 1)), np.ones(size, dtype=bool))
        rec, arrays, error = run(stacked)
        assert error is None and rec['complete'] is True and rec['status'] == 'OK'
        assert rec['selected_index'] == 0 and len(rec['recoveries']) == 2
        for row in rec['recoveries']:
            raw = _tag(row['raw'], arrays, set())
            _validate_recovery(raw, size, stacked[1])
            assert row['complete'] is True and row['python_call_attempted'] is True
            assert row['python_call_returned'] is True and row['raw_capture'] == 'LOSSLESS'
            assert row['outer_facts'] == _outer_facts(raw)
            assert type(row['native_ns']) is int and row['native_ns'] >= 0
            assert raw[4].shape == (4, size) and np.isnan(raw[4]).all()
            assert (_tag(row['mask_after'], arrays, set()) == 255).all()
        mark('fake_first_strict_max_and_whole_nonfinite_unused_fifth')
        audited = _check_fit(rec, arrays, set(), points, points, camera)
        assert audited['selected'] == 0 and len(audited['replays']) == 2
        options['encoding'] = 1
        _bad(lambda: _check_fit(rec, arrays, set(), points, points, camera))
        options['encoding'] = 255
        _bad(lambda: _same_return_schema(np.eye(3, dtype=np.float32), np.eye(3)))
        _bad(lambda: _same_return_schema(np.zeros(3), np.zeros((3, 1))))
        mark('independent_replay_rejects_mask_bytes_dtype_and_shape_changes')
        rec, arrays, error = run(stacked, zero_count=True)
        assert error is None and rec['complete'] is True and rec['status'] == 'FAILED'
        assert rec['selected_index'] is None
        mark('fake_zero_count_is_ordinary_failure')
        rec, arrays, error = run(stacked, bad_fifth=True)
        assert error is not None and rec['complete'] is False
        row = rec['recoveries'][0]
        assert row['python_call_returned'] is True and row['raw_capture'] == 'UNENCODABLE_CONTENT_UNKNOWN'
        assert row['outer_facts']['arity'] == 5
        assert row['outer_facts']['items'][4]['type'] == 'UnencodableFixture'
        assert 'mask_after' in row
        mark('fake_unencodable_recovery_outer_facts_retained')
        rec, arrays, error = run(stacked, recovery_throw=True)
        assert error is not None and rec['complete'] is False
        row = rec['recoveries'][0]
        assert row['python_call_attempted'] is True and row['python_call_returned'] is False
        assert row['raw_capture'] == 'NO_RETURN_OBSERVED'
        assert type(row['native_ns']) is int and row['native_ns'] >= 0
        after = _tag(row['mask_after'], arrays, set())
        assert after[0, 0] == 0 and (after[1:] == 1).all()
        mark('fake_recovery_throw_retains_postmask_and_finally_timing')
    finally:
        for name, value in previous.items():
            if value is absent:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value
    return checks


def _existing(context, name, cap=1048576):
    """Read a fixed output name once, then bind its immutable whole-byte pin."""
    root = Path(context['root'])
    current = root
    rel = Path(name)
    assert not rel.is_absolute() and '..' not in rel.parts
    for component in rel.parts:
        current = current / component
        assert not current.is_symlink()
    stat = current.stat()
    assert current.is_file() and stat.st_nlink == 1 and stat.st_size <= cap
    with current.open('rb') as stream:
        raw = stream.read(cap + 1)
    now = current.stat()
    assert (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns) == (
        now.st_dev, now.st_ino, now.st_size, now.st_mtime_ns)
    assert len(raw) == stat.st_size
    return _pin(name, raw), raw


def load_pair(context, pair):
    import numpy as np
    assert pair['pair_index'] in range(8) and len(pair['banks']) == 4
    values = []
    for bank in pair['banks']:
        pin = {k: bank[k] for k in ('path', 'bytes', 'sha256')}
        value = _arrays(context, pin, bank['array_fingerprints'])
        assert all(np.isfinite(a).all() for a in value.values())
        values.append(value)
    features, cameras, stock, fast = values
    count = (764, 513, 791, 605, 359, 878, 777, 393)[pair['pair_index']]
    assert stock['source_indices'].shape == stock['target_indices'].shape == (count,)
    for key in ('source_indices', 'target_indices', 'matches0', 'descriptors0',
                'descriptors1', 'raw_p0', 'raw_p1'):
        assert np.array_equal(stock[key], fast[key])
    source, target = stock['source_indices'], stock['target_indices']
    assert len(set(source.tolist())) == len(set(target.tolist())) == count
    assert np.array_equal(np.flatnonzero(stock['matches0'] >= 0), source)
    assert np.array_equal(stock['matches0'][source], target)
    for view, indices in ((0, source), (1, target)):
        suffix = str(view)
        descriptor, keypoint = features['descriptors' + suffix], features['keypoints' + suffix]
        assert descriptor.shape == (2048, 64)
        assert np.array_equal(descriptor[indices], stock['descriptors' + suffix])
        assert np.array_equal(keypoint[indices], stock['raw_p' + suffix])
        camera, transform = cameras['K' + suffix], cameras['image_transform' + suffix]
        original = np.asarray(pair['K' + suffix], dtype=np.float64)
        assert np.array_equal(cameras['original_K' + suffix], original)
        assert camera.shape == transform.shape == (3, 3)
        assert abs(float(np.linalg.det(camera))) > 1e-12
        assert np.allclose(camera, transform @ original, atol=1e-9, rtol=1e-12)
        height, width = map(int, cameras['hw' + suffix])
        assert height > 0 and width > 0
        assert keypoint.ndim == 2 and keypoint.shape[1] == 2
        assert ((0 <= keypoint[:, 0]) & (keypoint[:, 0] < width)).all()
        assert ((0 <= keypoint[:, 1]) & (keypoint[:, 1] < height)).all()
        for points in (stock['p' + suffix], fast['p' + suffix]):
            assert points.shape == (count, 2) and points.dtype == np.float32
            assert np.isfinite(points).all()
    return values


def _check_assignment_witness(actual, d0, d1, source_indices, target_indices):
    import numpy as np
    rows, cols = actual['rows'], actual['cols']
    selected = actual['assignment_columns']
    assert len(selected) == len(rows) and all(type(c) is int for c in selected)
    assert len(set(selected)) == len(selected)
    scores = []
    for r, c in enumerate(selected):
        assert 0 <= c < len(cols) + len(rows)
        if c >= len(cols):
            assert c == len(cols) + r
        else:
            a, b = d0[rows[r]].astype(np.float64), d1[cols[c]].astype(np.float64)
            na = math.sqrt(math.fsum(float(x) * float(x) for x in a))
            nb = math.sqrt(math.fsum(float(x) * float(x) for x in b))
            scores.append(math.fsum(float(x / na) * float(y / nb) for x, y in zip(a, b)))
    _close_values([math.fsum(scores)], [actual['optimum']])
    # Count the producer's exclusions from its own optimum, allowing tied
    # independent optima to use a different witness without false rejection.
    assert actual['solves'] == 1 + sum(
        selected[rows.index(int(source_indices[k]))] == cols.index(int(target_indices[k]))
        for k in actual['anchors'])


def _timings(records):
    result = {}
    for arm in ('ar', 'cosine', 'margin', 'spatial_margin'):
        values = []
        for pair in records:
            assert len(pair['timed_ns']) == 5
            for timed in pair['timed_ns']:
                assert set(timed) == {'shared', 'ar_branch', 'cosine_branch',
                                      'margin_branch', 'spatial_margin_branch'}
                assert all(type(v) is int and 0 <= v <= 240000000000 for v in timed.values())
                values.append((timed['shared'] + timed[arm + '_branch']) / 1e6)
        assert len(values) == 40
        ordered = sorted(values)

        def quantile(q):
            position = 39 * q
            i = int(math.floor(position))
            fraction = position - i
            return ordered[i] if i == 39 else ordered[i] + fraction * (ordered[i + 1] - ordered[i])

        result[arm] = dict(all_ms=values, median_ms=quantile(.5), p95_ms=quantile(.95))
    return result


def _mask_reports(records):
    result = []
    for p, record in enumerate(records):
        masks = record['primary']['masks']
        ar = masks['ar']
        overlap = {}
        for arm in ('fast', 'cosine', 'margin', 'spatial_margin'):
            other = masks[arm]
            overlap[arm] = dict(both_kept=sum(a and b for a, b in zip(ar, other)),
                ar_only_kept=sum(a and not b for a, b in zip(ar, other)),
                control_only_kept=sum(b and not a for a, b in zip(ar, other)),
                both_removed=sum(not a and not b for a, b in zip(ar, other)))
        result.append(dict(pair_index=p, q=record['primary']['q'],
            match_counts={arm: len(ar) if arm == 'stock' else sum(masks[arm]) for arm in ARMS},
            overlap=overlap))
    return result


def no_effect(records):
    assert len(records) == 8
    differences = {arm: [] for arm in ('fast', 'cosine', 'margin', 'spatial_margin')}
    for row in records:
        masks = row['primary']['masks']
        for arm in differences:
            differences[arm].append(masks['ar'] != masks[arm])
    scenes = {row['scene'] for row in records}
    assert scenes == {'0015', '0022'}
    diverse = all(any(differences['fast'][i] for i, row in enumerate(records)
                      if row['scene'] == scene) for scene in scenes)
    diverse = diverse and all(sum(differences[arm]) >= 2
                              for arm in ('cosine', 'margin', 'spatial_margin'))
    return not diverse, differences


def _tag(tag, arrays, used):
    import numpy as np
    assert type(tag) is dict and type(tag.get('type')) is str
    kind = tag['type']
    if kind in ('ndarray', 'numpy_scalar'):
        assert set(tag) == {'type', 'key', 'dtype', 'shape'}
        key = tag['key']
        assert key != 'record_json' and key in arrays
        value = arrays[key]
        assert value.dtype.str == tag['dtype'] and list(value.shape) == tag['shape']
        assert value.dtype.kind in 'biuf' and value.ndim <= 3
        used.add(key)
        if kind == 'numpy_scalar':
            assert value.shape == ()
            return value[()]
        return value
    if kind in ('tuple', 'list'):
        assert set(tag) == {'type', 'items'} and type(tag['items']) is list
        assert len(tag['items']) <= 16
        values = [_tag(item, arrays, used) for item in tag['items']]
        return tuple(values) if kind == 'tuple' else values
    if kind == 'NoneType':
        assert set(tag) == {'type'}
        return None
    assert set(tag) == {'type', 'value'}
    assert kind in ('int', 'float', 'bool') and type(tag['value']).__name__ == kind
    assert kind != 'float' or math.isfinite(tag['value'])
    return tag['value']


def _outer_facts(value, depth=0):
    import numpy as np
    kind = type(value)
    result = dict(type=kind.__name__[:80], module=kind.__module__[:120])
    if isinstance(value, (np.ndarray, np.generic)):
        result.update(dtype=value.dtype.str, shape=list(value.shape)[:64])
    elif kind in (tuple, list):
        result['arity'] = len(value)
        if depth == 0:
            result['items'] = [_outer_facts(item, 1) for item in value[:8]]
            result['omitted_items'] = max(0, len(value) - 8)
    return result


def _same_return_schema(actual, expected):
    import numpy as np
    assert type(actual) is type(expected)
    if isinstance(expected, np.ndarray):
        assert actual.dtype == expected.dtype and actual.shape == expected.shape
    elif isinstance(expected, np.generic):
        assert actual.dtype == expected.dtype
    elif type(expected) in (tuple, list):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            _same_return_schema(a, b)
    elif expected is None:
        assert actual is None
    else:
        assert type(expected) in (bool, int, float) and actual == expected


def _return_description(value):
    """Audit replay hashes are distinguished from whole producer raw arrays."""
    import numpy as np
    if isinstance(value, np.ndarray):
        return dict(type='ndarray', dtype=value.dtype.str, shape=list(value.shape),
                    strides=list(value.strides), raw_sha256=_array_hash(value),
                    bytes=value.nbytes)
    if isinstance(value, np.generic):
        result = _return_description(np.asarray(value))
        result['type'] = 'numpy_scalar'
        return result
    if type(value) in (tuple, list):
        return dict(type=type(value).__name__, items=[_return_description(v) for v in value])
    if value is None:
        return dict(type='NoneType')
    assert type(value) in (bool, int, float)
    return dict(type=type(value).__name__, value=value)


def _validate_gc(raw, count):
    import numpy as np
    assert type(raw) is tuple and len(raw) == 2
    essential, mask = raw
    assert type(mask) is np.ndarray and mask.dtype.kind in 'biu'
    assert mask.shape in ((count,), (count, 1))
    assert np.isin(mask, [0, 1]).all()
    if essential is None:
        return None, mask
    assert type(essential) is np.ndarray and essential.dtype.kind == 'f'
    assert essential.ndim == 2 and essential.shape[1] == 3
    assert essential.shape[0] >= 3 and essential.shape[0] % 3 == 0
    assert np.isfinite(essential).all() and mask is not None
    return essential, mask


def _validate_recovery(raw, count, gc_mask):
    import numpy as np
    assert type(raw) is tuple and len(raw) in (4, 5)
    retained, rotation, translation, mask = raw[:4]
    assert type(retained) is int and 0 <= retained <= count
    assert type(rotation) is np.ndarray and _proper(rotation)
    assert type(translation) is np.ndarray and translation.dtype.kind == 'f'
    assert translation.shape in ((3,), (3, 1)) and np.isfinite(translation).all()
    assert type(mask) is np.ndarray and mask.dtype == np.dtype('u1')
    assert mask.shape == (count, 1) and int(np.count_nonzero(mask)) == retained
    assert np.isin(mask, [0, 1, 255]).all()
    assert not ((mask.reshape(-1) != 0) & (gc_mask.reshape(-1) == 0)).any()
    if len(raw) == 5:
        assert type(raw[4]) is np.ndarray and raw[4].dtype.kind == 'f'
        assert raw[4].shape == (4, count)
    return retained, rotation, translation, mask


def _check_fit(fit, arrays, used, p0, p1, camera):
    """Replay saved E once each; there is no GC call in the reference."""
    import numpy as np
    import cv2
    assert fit['complete'] is True and fit['stage'] == 'COMPLETE'
    count = len(p0)
    assert type(fit['draw']) is int and 0 <= fit['draw'] < 5
    assert type(fit['first_arm_position']) is int and 0 <= fit['first_arm_position'] < 6
    assert fit['selected_index'] is None or type(fit['selected_index']) is int
    decoded_inputs = {key: _tag(value, arrays, used) for key, value in fit['inputs'].items()}
    wanted = dict(points=np.ascontiguousarray(np.column_stack((p0, p1)), dtype=np.float64),
                  K0=camera['K0'], K1=camera['K1'], hw0=camera['hw0'], hw1=camera['hw1'])
    assert set(decoded_inputs) in (set(wanted), set(wanted) | {'normalized0', 'normalized1', 'identity'})
    for name, value in wanted.items():
        assert np.array_equal(decoded_inputs[name], value)
    assert type(fit['timing_ns']) is int and fit['timing_ns'] >= 0
    if count < 5:
        assert fit['gc_raw'] is None and not fit['recoveries']
        assert fit['selected_index'] is None and fit['status'] == 'FAILED'
        return dict(attempted=0, replays=[], selected=None)
    raw_gc = _tag(fit['gc_raw'], arrays, used)
    assert fit['gc_python_call_attempted'] is True and fit['gc_python_call_returned'] is True
    assert fit['gc_raw_capture'] == 'LOSSLESS'
    assert type(fit['gc_ns']) is int and fit['gc_ns'] >= 0
    assert fit['gc_outer_facts'] == _outer_facts(raw_gc)
    essential, gc_mask = _validate_gc(raw_gc, count)
    if essential is None:
        assert not fit['recoveries'] and fit['selected_index'] is None
        assert fit['status'] == 'FAILED' and set(decoded_inputs) == set(wanted)
        return dict(attempted=1, replays=[], selected=None)
    n0, n1 = normalize(p0, camera['K0']), normalize(p1, camera['K1'])
    assert set(decoded_inputs) == set(wanted) | {'normalized0', 'normalized1', 'identity'}
    assert np.array_equal(decoded_inputs['normalized0'], n0)
    assert np.array_equal(decoded_inputs['normalized1'], n1)
    assert np.array_equal(decoded_inputs['identity'], np.eye(3))
    assert len(fit['recoveries']) == essential.shape[0] // 3
    selected, best, replay_records = None, 0, []
    for index, record in enumerate(fit['recoveries']):
        assert type(record['E_index']) is int and record['E_index'] == index
        assert record['complete'] is True
        assert record['python_call_attempted'] is True and record['python_call_returned'] is True
        assert record['raw_capture'] == 'LOSSLESS'
        assert type(record['native_ns']) is int and record['native_ns'] >= 0
        original = _tag(record['raw'], arrays, used)
        assert record['outer_facts'] == _outer_facts(original)
        original_count, original_R, original_t, original_mask = _validate_recovery(original, count, gc_mask)
        initial = _tag(record['mask_input'], arrays, used)
        after = _tag(record['mask_after'], arrays, used)
        assert initial.dtype == np.dtype('u1') and initial.shape == (count, 1)
        assert np.array_equal(initial, gc_mask.astype(np.uint8).reshape(-1, 1))
        assert after.dtype == initial.dtype and after.shape == initial.shape
        e = essential[3 * index:3 * index + 3].copy()
        a, b, identity, mask = n0.copy(), n1.copy(), np.eye(3), initial.copy()
        e_before, a_before, b_before, i_before = e.copy(), a.copy(), b.copy(), identity.copy()
        replay = cv2.recoverPose(e, a, b, identity, 1e9, mask=mask)
        assert np.array_equal(e, e_before) and np.array_equal(a, a_before)
        assert np.array_equal(b, b_before) and np.array_equal(identity, i_before)
        replay_count, replay_R, replay_t, replay_mask = _validate_recovery(replay, count, gc_mask)
        _same_return_schema(replay, original)
        assert replay_count == original_count
        _close_values(replay_R, original_R, 1e-9)
        _close_values(replay_t, original_t, 1e-9)
        assert replay_mask.tobytes() == original_mask.tobytes()
        assert mask.dtype == after.dtype and mask.shape == after.shape
        assert mask.tobytes() == after.tobytes()
        replay_records.append(dict(E_index=index, count=replay_count,
            rotation=replay_R.tolist(), translation=replay_t.tolist(),
            original_raw=_return_description(original), replay_raw=_return_description(replay),
            consumed_match=True, schema_match=True,
            rotation_raw_hash_match=_array_hash(replay_R) == _array_hash(original_R),
            translation_raw_hash_match=_array_hash(replay_t) == _array_hash(original_t),
            rotation_max_abs_difference=float(np.max(np.abs(replay_R - original_R))),
            translation_max_abs_difference=float(np.max(np.abs(replay_t - original_t))),
            mask_raw_hash_match=True, postmask_raw_hash_match=True,
            unused_fifth_raw_hash_match=(_array_hash(replay[4]) == _array_hash(original[4])) if len(replay) == 5 else None,
            unused_fifth_policy='schema exact; raw hash comparison reported; values are not consumed',
            unused_fifth_nonfinite_allowed=True,
            whole_raw_evidence='producer arrays; replay schemas and byte hashes'))
        if original_count > best:
            selected, best = index, original_count
    assert fit['selected_index'] == selected
    assert fit['status'] == ('OK' if selected is not None else 'FAILED')
    return dict(attempted=1, replays=replay_records, selected=selected)


def _directed_truths(context, pairs):
    import numpy as np
    pin = dict(path='vendor/XFeat/assets/megadepth_1500.json', bytes=4024939,
               sha256='6aa01108aa26b1d5f5f1aad58a35514a173febd525f71cdab312ef72edbcca19')
    raw = _read(context, pin, 4194304)
    data = json.loads(raw.decode('utf8'))
    assert type(data) is list and len(data) == 1500
    truths = []
    for pair in pairs:
        locations = [i for i, item in enumerate(data) if item['pair_names'] == pair['pair_names']]
        assert locations == [pair['legacy_metadata_pair_id']]
        record = data[locations[0]]
        assert np.array_equal(np.asarray(record['K0']), np.asarray(pair['K0']))
        assert np.array_equal(np.asarray(record['K1']), np.asarray(pair['K1']))
        transform = np.asarray(record['T_0to1'], dtype=np.float64)
        assert transform.shape == (4, 4) and np.isfinite(transform).all()
        assert np.array_equal(transform[3], np.array([0., 0., 0., 1.]))
        assert _proper(transform[:3, :3])
        truths.append(transform)
    return truths


def _expected_groups(pair_index, draw, banks, masks):
    import numpy as np
    stock, fast = banks[2:]
    offset = (pair_index + draw) % len(ARMS)
    ordered = ARMS[offset:] + ARMS[:offset]
    groups, identities = [], {}
    for position, arm in enumerate(ordered):
        bank = stock if arm == 'stock' else fast
        keep = np.ones(len(bank['p0']), dtype=bool) if arm in ('stock', 'fast') else np.asarray(masks[arm], dtype=bool)
        assert keep.shape == (len(bank['p0']),)
        p0, p1 = bank['p0'][keep], bank['p1'][keep]
        identity = (p0.dtype.str, tuple(p0.shape), p0.tobytes(),
                    p1.dtype.str, tuple(p1.shape), p1.tobytes())
        if identity in identities:
            groups[identities[identity]]['arms'].append(arm)
        else:
            index = len(groups)
            identities[identity] = index
            groups.append(dict(id=f'd{draw}g{index}', draw=draw,
                first_arm_position=position, arms=[arm], p0=p0, p1=p1,
                coords_hash=hashlib.sha256(p0.tobytes() + p1.tobytes()).hexdigest()))
    return groups


def _nested_match(actual, expected, atol=1e-8):
    """Fixed audit tolerance for arithmetic; boolean decisions remain exact."""
    if type(expected) is dict:
        assert type(actual) is dict and set(actual) == set(expected)
        for name, value in expected.items():
            _nested_match(actual[name], value, atol)
    elif type(expected) is list:
        assert type(actual) is list and len(actual) == len(expected)
        for a, b in zip(actual, expected):
            _nested_match(a, b, atol)
    elif type(expected) is float:
        assert type(actual) in (float, int) and math.isfinite(actual)
        assert abs(actual - expected) <= atol
    else:
        assert type(actual) is type(expected) and actual == expected


def _score(poses, truths, records, timing):
    import numpy as np
    errors = [[[180.0 for _ in ARMS] for _ in range(5)] for _ in range(8)]
    success = [[[False for _ in ARMS] for _ in range(5)] for _ in range(8)]
    detail = []
    for pair, (bank, arrays) in enumerate(poses):
        seen = set()
        for fit in bank['fits']:
            value = (180.0, 180.0, 180.0)
            ok = fit['status'] == 'OK'
            if ok:
                decoded = _tag(fit['recoveries'][fit['selected_index']]['raw'], arrays, set())
                value = pose_error(truths[pair], decoded[1], decoded[2])
            draw = fit['draw']
            for arm in fit['arms']:
                a = ARMS.index(arm)
                assert (draw, a) not in seen
                seen.add((draw, a))
                errors[pair][draw][a], success[pair][draw][a] = value[0], ok
                detail.append(dict(pair_index=pair, draw=draw, arm=arm,
                                   fit_id=fit['id'], error=list(value), success=ok))
        assert len(seen) == 30
    aucs = {arm: [auc([errors[p][r][a] for p in range(8)]) for r in range(5)]
            for a, arm in enumerate(ARMS)}
    means = {arm: [math.fsum(aucs[arm][r][t] for r in range(5)) / 5
                   for t in range(3)] for arm in ARMS}
    scenes = {}
    for scene in ('0015', '0022'):
        rows = [i for i, record in enumerate(records) if record['scene'] == scene]
        assert rows == ([0, 6] if scene == '0015' else [1, 2, 3, 4, 5, 7])
        scenes[scene] = {}
        for a, arm in enumerate(ARMS):
            draws = [auc([errors[p][r][a] for p in rows]) for r in range(5)]
            scenes[scene][arm] = [math.fsum(v[t] for v in draws) / 5 for t in range(3)]
    counts = {}
    for a, arm in enumerate(ARMS):
        counts[arm] = dict(
            failures=sum(not success[p][r][a] for p in range(8) for r in range(5)),
            catastrophes_over90=sum(errors[p][r][a] > 90 for p in range(8) for r in range(5)),
            rescues_vs_fast=sum(errors[p][r][a] <= 5 < errors[p][r][1] for p in range(8) for r in range(5)),
            damages_vs_fast=sum(errors[p][r][1] <= 5 < errors[p][r][a] for p in range(8) for r in range(5)))
    gates = dict(
        auc_each_control=all(means['ar'][0] - means[arm][0] >= .5 for arm in ARMS if arm != 'ar'),
        four_positive_draws=sum(aucs['ar'][r][0] > aucs['fast'][r][0] for r in range(5)) >= 4,
        each_scene_positive=all(v['ar'][0] > v['fast'][0] for v in scenes.values()),
        failures=counts['ar']['failures'] <= counts['fast']['failures'],
        catastrophes=counts['ar']['catastrophes_over90'] <= counts['fast']['catastrophes_over90'],
        median_filter=timing['ar']['median_ms'] <= 10,
        p95_filter=timing['ar']['p95_ms'] <= 25)
    return dict(outcome='PILOT_PASS_PENDING_AUDIT' if all(gates.values()) else 'SCIENTIFIC_NEGATIVE_PENDING_AUDIT',
                gates=gates, auc_by_draw=aucs, mean_auc=means, scene_mean_auc=scenes,
                counts=counts, pair_draw_results=detail)


def audit(context):
    """Independent cached-input, native-recovery and directed-score replay."""
    import numpy as np
    version = context['version']
    assert type(version) is int and version in (1, 2, 3, 4, 5, 6, 7, 8)
    passport = _document(context, PASSPORT_PIN)
    assert len(passport['pairs']) == 8
    seal_name = RESULT_ROOT + f'prediction-seal-v{version}.json'
    seal_pin, seal_raw = _existing(context, seal_name)
    seal = json.loads(seal_raw.decode('ascii'))
    assert _json(seal) == seal_raw
    assert set(seal) == {'schema', 'prediction', 'pose_pins', 'gt_read', 'status'}
    assert seal['schema'] == 'ar-prediction-seal-v1' and seal['gt_read'] is False
    assert seal['prediction']['path'] == RESULT_ROOT + f'predictions-v{version}.npz'
    prediction, prediction_arrays = _record(context, seal['prediction'])
    assert set(prediction_arrays) == {'record_json'}
    assert prediction['schema'] == 'ar-predictions-v1' and type(prediction['version']) is int
    assert prediction['version'] == version and type(prediction['native_calls']) is int
    assert prediction['labels_read'] is False and prediction['native_calls'] == 0
    assert prediction['input_pins'] == [bank for pair in passport['pairs'] for bank in pair['banks']]
    records = prediction['pairs']
    assert len(records) == 8
    all_banks, input_replay = [], []
    for p, (pair, record) in enumerate(zip(passport['pairs'], records)):
        assert type(pair['pair_index']) is type(record['pair_index']) is int
        assert pair['pair_index'] == record['pair_index'] == p
        assert record['scene'] == pair['scene']
        banks = load_pair(context, pair)
        features, camera, _, fast = banks
        expected = filter_pair(features['descriptors0'], features['descriptors1'],
            fast['source_indices'], fast['target_indices'], fast['raw_p0'], camera['hw0'])
        _compare_filter(record['primary'], expected)
        _check_assignment_witness(record['primary'], features['descriptors0'], features['descriptors1'],
                                  fast['source_indices'], fast['target_indices'])
        assert len(record['solve_counts']) == 6
        assert all(type(v) is int and 1 <= v <= 33 for v in record['solve_counts'])
        assert record['solve_counts'] == [record['primary']['solves']] * 6
        assert type(record['input_load_and_integrity_ns']) is int and record['input_load_and_integrity_ns'] >= 0
        input_replay.append(dict(pair_index=p, q=expected['q'], rows=len(expected['rows']),
            cols=len(expected['cols']), solves=expected['solves'], masks_match=True,
            objective_match=True, reference_uses_producer_math=False))
        all_banks.append(banks)
    assert sum(sum(record['solve_counts']) for record in records) <= 1584
    assert sum(row['solves'] for row in input_replay) <= 264
    ineffective, differences = no_effect(records)
    timing = _timings(records)
    summary_pin, summary_raw = _existing(context, RESULT_ROOT + f'summary-v{version}.json')
    summary = json.loads(summary_raw.decode('ascii'))
    assert _json(summary) == summary_raw
    assert summary['schema'] == 'ar-summary-v1' and type(summary['version']) is int
    assert summary['version'] == version and type(summary['native_calls']) is int
    assert summary['prediction_pin'] == seal['prediction'] and summary['seal_pin'] == seal_pin
    assert summary['full_pipeline_latency'] == 'UNMEASURED'
    assert summary['physical_correspondence_truth'] == 'UNKNOWN'
    assert summary['scientific_family_increment'] == 'ONLY_AFTER_INDEPENDENT_AUDIT'
    _nested_match(summary['filter_timing'], timing, 1e-9)
    assert summary['mask_differences'] == differences
    assert type(summary['producer_elapsed_ns']) is int and summary['producer_elapsed_ns'] >= 0
    poses, native_replay, attempts = [], [], 0
    if ineffective:
        assert seal['status'] == 'NO_EFFECT' and seal['pose_pins'] == []
        assert summary['outcome'] == 'NO_EFFECT_PENDING_AUDIT'
        assert summary['gt_read'] is False and summary['native_calls'] == 0
        assert not any(name in summary for name in ('mean_auc', 'auc_by_draw', 'pair_draw_results'))
        for p in range(8):
            assert not (Path(context['root']) / (RESULT_ROOT + f'poses-v{version}-{p:04d}.npz')).exists()
    else:
        assert seal['status'] == 'ALL_NATIVE_OUTPUTS_SEALED' and len(seal['pose_pins']) == 8
        for p, pose_pin in enumerate(seal['pose_pins']):
            assert pose_pin['path'] == RESULT_ROOT + f'poses-v{version}-{p:04d}.npz'
            pose, arrays = _record(context, pose_pin)
            assert pose['schema'] == 'ar-native-pose-v1' and type(pose['pair_index']) is int
            assert pose['pair_index'] == p and pose['complete'] is True and 'failure' not in pose
            assert pose['prediction'] == seal['prediction']
            groups = [group for draw in range(5) for group in
                      _expected_groups(p, draw, all_banks[p], records[p]['primary']['masks'])]
            assert len(groups) == len(pose['fits'])
            used = set()
            for fit, group in zip(pose['fits'], groups):
                for name in ('id', 'draw', 'first_arm_position', 'arms', 'coords_hash'):
                    assert fit[name] == group[name]
                result = _check_fit(fit, arrays, used, group['p0'], group['p1'], all_banks[p][1])
                attempts += result['attempted']
                native_replay.append(dict(pair_index=p, fit_id=fit['id'],
                    producer_pose_bank=pose_pin, selected_index=result['selected'],
                    replays=result['replays'], complete=True))
            assert used == set(arrays) - {'record_json'}
            poses.append((pose, arrays))
        assert attempts <= 240 and summary['native_calls'] == attempts
        assert summary['gt_read'] is True
        # This is the first GT read in the audit, after all sealed banks validate.
        truths = _directed_truths(context, passport['pairs'])
        scored = _score(poses, truths, records, timing)
        for key, value in scored.items():
            _nested_match(summary[key], value)
    # Re-read all immutable seal operands after replay before accepting audit.
    _read(context, seal_pin, 1048576)
    _read(context, seal['prediction'], 16777216)
    _read(context, summary_pin, 1048576)
    for pin in seal['pose_pins']:
        _read(context, pin, 16777216)
    report = dict(schema='ar-audit-v1', status='PASS', complete=True, version=version,
        prediction_seal=seal_pin, prediction=seal['prediction'], summary=summary_pin,
        mask_pairs=8, input_replay=input_replay, native_fits=attempts,
        recovered_replays=sum(len(row['replays']) for row in native_replay),
        native_replay=native_replay, summary_verified=True,
        no_effect=ineffective, outcome=summary['outcome'],
        pair_match_counts_and_overlap=_mask_reports(records),
        tolerance=dict(objective=1e-10, recovery=1e-9, score=1e-8, rtol=0),
        raw_evidence_scope='whole producer returns; independent replay schemas/hash evidence',
        physical_correspondence_truth='UNKNOWN', full_pipeline_latency='UNMEASURED')
    raw = _json(report)
    assert len(raw) <= 1048576
    artifact = context['write'](RESULT_ROOT + f'audit-v{version}.json', raw)
    assert artifact == _pin(RESULT_ROOT + f'audit-v{version}.json', raw)
    return dict(schema='ar-audit-v1', status='PASS', complete=True, version=version,
                prediction_seal=seal_pin, mask_pairs=8, native_fits=attempts,
                recovered_replays=report['recovered_replays'], summary_verified=True,
                artifact=artifact)
