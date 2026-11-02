"""AR fixed-protocol producer. Importing this file performs no numerical work."""
import hashlib
import io
import json
import math
import time
import zipfile
from pathlib import Path

AR = 'research/ar-persistence-20261014/'
OUT = 'results/xrefine/ar-persistence-20261014/'
ARMS = ('stock', 'fast', 'ar', 'cosine', 'margin', 'spatial_margin')
GT = 'vendor/XFeat/assets/megadepth_1500.json'
GT_SHA = '6aa01108aa26b1d5f5f1aad58a35514a173febd525f71cdab312ef72edbcca19'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'),
                       ensure_ascii=True, allow_nan=False) + '\n').encode()


def sha(value):
    return hashlib.sha256(value).hexdigest()


def checked(root, pin):
    path = Path(root) / pin['path']
    require(not Path(pin['path']).is_absolute() and '..' not in Path(pin['path']).parts,
            'relative input path')
    require(path.is_file() and not path.is_symlink() and path.stat().st_nlink == 1,
            'regular unique input')
    require(path.stat().st_size == pin['bytes'], 'input size')
    raw = path.read_bytes()
    require(sha(raw) == pin['sha256'], 'input checksum')
    return raw


def load_bank(root, record):
    import numpy as np
    raw = checked(root, record)
    pins = record['array_fingerprints']
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        names = [k + '.npy' for k in pins]
        require(sorted(z.namelist()) == sorted(names), 'exact unique NPZ members')
        for info in z.infolist():
            p = pins[info.filename[:-4]]
            size = math.prod(p['shape']) * np.dtype(p['dtype']).itemsize
            require(size <= info.file_size <= size + 4096, 'NPY size bound')
    arrays = {}
    with np.load(io.BytesIO(raw), allow_pickle=False) as bank:
        for key, pin in pins.items():
            value = bank[key]
            require(value.dtype.str == pin['dtype'] and list(value.shape) == pin['shape'],
                    'array schema')
            require(sha(value.tobytes(order='C')) == pin['sha256'], 'array checksum')
            require(np.isfinite(value).all(), 'finite input')
            arrays[key] = value
    return arrays


def load_pair(root, pair):
    import numpy as np
    require(len(pair['banks']) == 4, 'four input banks')
    features, camera, stock, fast = [load_bank(root, r) for r in pair['banks']]
    n = len(stock['p0'])
    require(n == (764, 513, 791, 605, 359, 878, 777, 393)[pair['pair_index']],
            'frozen match count')
    ids0, ids1 = stock['source_indices'], stock['target_indices']
    require(ids0.dtype.kind == ids1.dtype.kind == 'i', 'integer native IDs')
    require(ids0.shape == ids1.shape == (n,), 'native ID shape')
    require(((0 <= ids0) & (ids0 < 2048)).all() and
            ((0 <= ids1) & (ids1 < 2048)).all(), 'native ID bounds')
    require(len(np.unique(ids0)) == len(np.unique(ids1)) == n, 'native one-to-one IDs')
    require(np.array_equal(np.flatnonzero(stock['matches0'] >= 0), ids0),
            'native source order')
    require(np.array_equal(stock['matches0'][ids0], ids1) and
            ((stock['matches0'] == -1) | ((stock['matches0'] >= 0) &
             (stock['matches0'] < 2048))).all(), 'native target join')
    for key in ('source_indices', 'target_indices', 'matches0', 'descriptors0',
                'descriptors1', 'raw_p0', 'raw_p1'):
        require(np.array_equal(stock[key], fast[key]), 'stock/fast invariant ' + key)
    for view, ids in ((0, ids0), (1, ids1)):
        k = str(view)
        require(features['descriptors' + k].shape == (2048, 64), 'full descriptor shape')
        require(np.array_equal(features['descriptors' + k][ids], stock['descriptors' + k]),
                'gathered descriptor join')
        require(np.array_equal(features['keypoints' + k][ids], stock['raw_p' + k]),
                'raw point join')
        K, H, hw = camera['K' + k], camera['image_transform' + k], camera['hw' + k]
        raw_K = np.asarray(pair['K' + k], dtype=np.float64)
        require(np.array_equal(camera['original_K' + k], raw_K), 'original calibration join')
        require(K.shape == H.shape == (3, 3) and hw.shape == (2,) and (hw > 0).all(),
                'processed camera schema')
        require(abs(float(np.linalg.det(K))) > 1e-12 and
                np.allclose(K, H @ raw_K, rtol=1e-12, atol=1e-9), 'camera transform')
        height, width = map(int, hw)
        points = features['keypoints' + k]
        require(points.ndim == 2 and points.shape[1] == 2 and
                ((points[:, 0] >= 0) & (points[:, 0] < width) &
                 (points[:, 1] >= 0) & (points[:, 1] < height)).all(), 'raw pixel bounds')
        for points in (stock['p' + k], fast['p' + k]):
            require(points.shape == (n, 2) and points.dtype == np.float32 and
                    np.isfinite(points).all(), 'finite native refined coordinate schema')
    return features, camera, stock, fast


def loader_fixtures(reference_load):
    """Fabricated loader contracts; restore both bank readers even on failure."""
    import numpy as np
    reference_globals = reference_load.__globals__
    old_bank, old_arrays = load_bank, reference_globals['_arrays']
    banks, checks = [], []
    pair = dict(pair_index=0, banks=[dict(path=str(i), bytes=0, sha256='0' * 64,
        array_fingerprints={}) for i in range(4)], K0=np.eye(3).tolist(), K1=np.eye(3).tolist())

    def fabricated():
        features, camera, stock = {}, {}, {}
        ids = np.arange(764, dtype=np.int64)
        matches = np.full(2048, -1, dtype=np.int64); matches[:764] = ids
        stock.update(source_indices=ids.copy(), target_indices=ids.copy(), matches0=matches)
        for side in ('0', '1'):
            points = np.tile(np.array([[.1, 9.9]], dtype=np.float32), (2048, 1))
            features['keypoints' + side] = points
            features['descriptors' + side] = np.ones((2048, 64), dtype=np.float32)
            for key in ('K', 'image_transform', 'original_K'):
                camera[key + side] = np.eye(3, dtype=np.float64)
            camera['hw' + side] = np.array([10, 10], dtype=np.int64)
            stock['raw_p' + side] = points[ids].copy()
            stock['descriptors' + side] = features['descriptors' + side][ids].copy()
            stock['p' + side] = stock['raw_p' + side].copy()
            stock['p' + side][0] = [-.5, 10.5]
        return [features, camera, stock, {k: v.copy() for k, v in stock.items()}]

    try:
        globals()['load_bank'] = lambda root, record: banks[int(record['path'])]
        reference_globals['_arrays'] = lambda context, pin, *args, **kwargs: banks[int(pin['path'])]
        cases = ['finite_outside', 'raw_outside'] + [
            (bank, side, defect) for bank in (2, 3) for side in ('0', '1')
            for defect in ('nan', 'inf', 'shape', 'dtype')]
        for case in cases:
            banks = fabricated()
            if case == 'raw_outside':
                for side in ('0', '1'):
                    banks[0]['keypoints' + side][0, 0] = -.1
                    for bank in (2, 3):
                        banks[bank]['raw_p' + side][0, 0] = -.1
            elif case != 'finite_outside':
                bank, side, defect = case
                name = 'p' + side
                if defect in ('nan', 'inf'):
                    banks[bank][name][0, 0] = float(defect)
                elif defect == 'shape':
                    banks[bank][name] = banks[bank][name][:, :1].copy()
                else:
                    banks[bank][name] = banks[bank][name].astype(np.float64)
            before = {(i, k): (v.dtype.str, v.shape, v.tobytes())
                      for i, bank in enumerate(banks) for k, v in bank.items()}
            for name, call in (('producer', lambda: load_pair('', pair)),
                               ('reference', lambda: reference_load({}, pair))):
                if case == 'finite_outside':
                    result = call()
                    require(all(result[i] is banks[i] for i in range(4)), 'fixture native bank identity')
                else:
                    try:
                        call()
                    except (AssertionError, ValueError, TypeError, IndexError):
                        pass
                    else:
                        raise AssertionError('Loader accepted malformed fixture: ' + name + str(case))
                require(before == {(i, k): (v.dtype.str, v.shape, v.tobytes())
                    for i, bank in enumerate(banks) for k, v in bank.items()}, 'fixture input mutation')
            checks.append(dict(id='both_loaders_' + str(case), status='PASS'))
    finally:
        globals()['load_bank'] = old_bank
        reference_globals['_arrays'] = old_arrays
    require(load_bank is old_bank and reference_globals['_arrays'] is old_arrays,
            'loader fixture reader restoration')
    return checks


def filter_pair(d0, d1, source_indices, target_indices, raw_p0, hw0):
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    begin = time.perf_counter_ns()
    x, y = np.asarray(d0, dtype=np.float64), np.asarray(d1, dtype=np.float64)
    ids0, ids1 = np.asarray(source_indices), np.asarray(target_indices)
    require(x.ndim == y.ndim == 2 and x.shape[1] == y.shape[1], 'descriptor dimensions')
    require(np.isfinite(x).all() and np.isfinite(y).all(), 'finite descriptors')
    nx, ny = np.linalg.norm(x, axis=1), np.linalg.norm(y, axis=1)
    require((nx > 1e-12).all() and (ny > 1e-12).all(), 'descriptor norms')
    require(ids0.ndim == ids1.ndim == 1 and ids0.shape == ids1.shape and len(ids0) > 0,
            'nonempty ID vectors')
    require(ids0.dtype.kind in 'iu' and ids1.dtype.kind in 'iu' and
            ((ids0 >= 0) & (ids0 < len(x))).all() and
            ((ids1 >= 0) & (ids1 < len(y))).all(), 'ID bounds')
    require(len(np.unique(ids0)) == len(ids0) and len(np.unique(ids1)) == len(ids1),
            'unique IDs')
    require(min(len(x), len(y)) >= 2, 'competitors exist')
    C = (x / nx[:, None]) @ (y / ny[:, None]).T
    row = C[ids0].copy()
    row[np.arange(len(ids0)), ids1] = -np.inf
    col = C[:, ids1].T.copy()
    col[np.arange(len(ids0)), ids0] = -np.inf
    competitors = np.argmax(col, axis=1)
    cosine = C[ids0, ids1]
    margin = np.minimum(cosine - row.max(axis=1), cosine - col.max(axis=1))
    anchors = np.lexsort((ids1, ids0, margin))[:min(32, len(ids0))]
    shared_end = time.perf_counter_ns()
    S = np.unique(np.r_[ids0[anchors], competitors[anchors]])
    columns = [int(j) for j in ids1[anchors]]
    all_targets = np.arange(len(y))
    for i in S:
        columns.extend(np.lexsort((all_targets, -C[i]))[:min(4, len(y))].tolist())
    T = np.asarray(sorted(set(columns)), dtype=np.int64)
    require(len(S) <= 64 and len(T) <= 288, 'pool bounds')
    weights = np.full((len(S), len(T) + len(S)), -np.inf, dtype=np.float64)
    weights[:, :len(T)] = C[np.ix_(S, T)]
    weights[np.arange(len(S)), len(T) + np.arange(len(S))] = 0.

    def solve(w):
        rr, cc = linear_sum_assignment(w, maximize=True)
        require(np.array_equal(rr, np.arange(len(S))) and np.isfinite(w[rr, cc]).all(),
                'complete feasible assignment')
        return cc, math.fsum(float(w[r, c]) for r, c in zip(rr, cc))

    chosen, optimum = solve(weights)
    sr, tc = {int(v): i for i, v in enumerate(S)}, {int(v): j for j, v in enumerate(T)}
    deltas, solves = [], 1
    for edge in anchors:
        i, j = sr[int(ids0[edge])], tc[int(ids1[edge])]
        delta = 0.
        if chosen[i] == j:
            excluded = weights.copy()
            excluded[i, j] = -np.inf
            _, alternative = solve(excluded)
            solves += 1
            delta = optimum - alternative
            require(delta >= -1e-10, 'negative exclusion delta')
            delta = max(0., delta)
        deltas.append(delta)
    masks = {'fast': np.ones(len(ids0), dtype=bool), 'ar': np.ones(len(ids0), dtype=bool)}
    rejected = anchors[np.asarray(deltas) <= .05 + 1e-10]
    masks['ar'][rejected] = False
    ar_end = time.perf_counter_ns()
    q = len(rejected)
    timings = {'shared': shared_end - begin, 'ar_branch': ar_end - shared_end}
    for arm, values in (('cosine', cosine), ('margin', margin)):
        start = time.perf_counter_ns()
        selected = anchors[np.lexsort((ids1[anchors], ids0[anchors], values[anchors]))[:q]]
        keep = np.ones(len(ids0), dtype=bool)
        keep[selected] = False
        masks[arm] = keep
        timings[arm + '_branch'] = time.perf_counter_ns() - start
    start = time.perf_counter_ns()
    xy, hw = np.asarray(raw_p0), np.asarray(hw0)
    require(xy.shape == (len(ids0), 2) and hw.shape == (2,) and
            np.isfinite(xy).all() and (hw > 0).all(), 'bin inputs')
    bins = np.clip(np.floor(xy / hw[::-1] * 4), 0, 3).astype(np.int64)
    bins = bins[:, 1] * 4 + bins[:, 0]
    q_bins = np.bincount(bins[rejected], minlength=16)
    keep = np.ones(len(ids0), dtype=bool)
    for b in range(16):
        a = anchors[bins[anchors] == b]
        selected = a[np.lexsort((ids1[a], ids0[a], margin[a]))[:q_bins[b]]]
        keep[selected] = False
    masks['spatial_margin'] = keep
    timings['spatial_margin_branch'] = time.perf_counter_ns() - start
    require(solves <= 33 and all(int((~m).sum()) == q for k, m in masks.items()
                                if k != 'fast'), 'matched removal counts')
    return {'anchors': anchors.tolist(), 'rows': S.tolist(), 'cols': T.tolist(),
            'cosine': cosine.tolist(), 'margin': margin.tolist(), 'deltas': deltas,
            'optimum': optimum, 'assignment_columns': chosen.tolist(),
            'masks': {k: v.tolist() for k, v in masks.items()}, 'q': q,
            'q_bins': q_bins.tolist(), 'timings_ns': timings, 'solves': solves}


def normalize(points, K):
    import numpy as np
    p, K = np.asarray(points, dtype=np.float64), np.asarray(K, dtype=np.float64)
    require(p.ndim == 2 and p.shape[1] == 2 and K.shape == (3, 3) and
            np.isfinite(p).all() and np.isfinite(K).all(), 'normalization schema')
    h = np.c_[p, np.ones(len(p))] @ np.linalg.inv(K).T
    require(np.isfinite(h).all() and (np.abs(h[:, 2]) > 1e-12).all(), 'homogeneous denominator')
    result = np.ascontiguousarray(h[:, :2] / h[:, 2:], dtype=np.float64)
    require(np.isfinite(result).all(), 'normalized finite')
    return result


def auc(errors, thresholds=(5, 10, 20)):
    import numpy as np
    e = np.asarray(errors, dtype=np.float64)
    require(e.ndim == 1 and len(e) > 0 and np.isfinite(e).all() and (e >= 0).all(), 'AUC input')
    e = np.r_[0., np.sort(e)]
    recall = np.r_[0., (np.arange(len(errors)) + 1) / len(errors)]
    output = []
    for threshold in thresholds:
        require(threshold > 0, 'positive AUC threshold')
        last = np.searchsorted(e, threshold)
        output.append(100. * float(np.trapz(np.r_[recall[:last], recall[last - 1]],
                                          x=np.r_[e[:last], threshold]) / threshold))
    return output


def pose_error(T, R, t):
    import numpy as np
    rotation = float(np.degrees(np.arccos(np.clip((np.trace(R.T @ T[:3, :3]) - 1) / 2, -1, 1))))
    t, truth = np.asarray(t).reshape(3), T[:3, 3]
    norm = np.linalg.norm(t) * np.linalg.norm(truth)
    if norm <= np.finfo(float).eps:
        return 180., rotation, 180.
    translation = float(np.degrees(np.arccos(np.clip(np.dot(t, truth) / norm, -1, 1))))
    translation = min(translation, 180. - translation)
    return max(rotation, translation), rotation, translation


def native_facts(value, depth=0):
    """Bounded outer facts only; unsupported content remains explicitly UNKNOWN."""
    import numpy as np
    kind = type(value)
    facts = {'type': kind.__name__[:80], 'module': kind.__module__[:120]}
    if isinstance(value, (np.ndarray, np.generic)):
        facts.update(dtype=value.dtype.str, shape=list(value.shape)[:64])
    elif kind in (tuple, list):
        facts['arity'] = len(value)
        if depth == 0:
            facts['items'] = [native_facts(v, 1) for v in value[:8]]
            facts['omitted_items'] = max(0, len(value) - 8)
    return facts


def tagged(value, arrays, prefix):
    import numpy as np
    if isinstance(value, np.ndarray):
        require(value.dtype.kind in 'biuf', 'native numeric array')
        require(prefix not in arrays, 'unique evidence array')
        arrays[prefix] = value.copy()
        return {'type': 'ndarray', 'key': prefix, 'dtype': value.dtype.str, 'shape': list(value.shape)}
    if isinstance(value, np.generic):
        result = tagged(np.asarray(value), arrays, prefix)
        result['type'] = 'numpy_scalar'
        return result
    if type(value) in (tuple, list):
        return {'type': type(value).__name__, 'items':
                [tagged(v, arrays, prefix + '_' + str(i)) for i, v in enumerate(value)]}
    if value is None:
        return {'type': 'NoneType'}
    require(type(value) in (bool, int, float) and
            (type(value) is not float or math.isfinite(value)), 'native scalar')
    return {'type': type(value).__name__, 'value': value}


def decoded(value, arrays):
    kind = value['type']
    if kind in ('ndarray', 'numpy_scalar'):
        array = arrays[value['key']]
        require(array.dtype.str == value['dtype'] and list(array.shape) == value['shape'], 'tag schema')
        return array[()] if kind == 'numpy_scalar' else array
    if kind in ('tuple', 'list'):
        items = [decoded(v, arrays) for v in value['items']]
        return tuple(items) if kind == 'tuple' else items
    if kind == 'NoneType':
        return None
    require(kind in ('int', 'float', 'bool') and type(value['value']).__name__ == kind,
            'scalar tag')
    return value['value']


def archive(record, arrays=None):
    import numpy as np
    arrays = dict(arrays or {})
    require('record_json' not in arrays, 'reserved archive member')
    arrays['record_json'] = np.frombuffer(canonical(record), dtype=np.uint8)
    output = io.BytesIO()
    np.savez(output, **arrays)
    return output.getvalue()


def fit(p0, p1, camera, arrays, ident, rec):
    import numpy as np
    import cv2
    import pygcransac
    start = time.perf_counter_ns()
    points = np.ascontiguousarray(np.c_[p0, p1], dtype=np.float64)
    K0, K1 = [np.ascontiguousarray(camera[k], dtype=np.float64).copy() for k in ('K0', 'K1')]
    h0, w0 = map(int, camera['hw0'])
    h1, w1 = map(int, camera['hw1'])
    inputs = {'points': points.copy(), 'K0': K0.copy(), 'K1': K1.copy(),
              'hw0': np.asarray([h0, w0]), 'hw1': np.asarray([h1, w1])}
    rec.update(id=ident, recoveries=[], selected_index=None, status='INCOMPLETE',
               gc_raw=None, complete=False, stage='INPUTS_READY')
    rec['inputs'] = {k: tagged(v, arrays, ident + '_input_' + k) for k, v in inputs.items()}
    if len(points) >= 5:
        rec.update(gc_python_call_attempted=True, gc_python_call_returned=False,
                   gc_raw_capture='NO_RETURN_OBSERVED', stage='GC_PYTHON_CALL_ATTEMPTED')
        native_start = time.perf_counter_ns()
        try:
            raw = pygcransac.findEssentialMatrix(points, K0, K1, h0, w0, h1, w1,
                                                probabilities=[], threshold=1., conf=.99999,
                                                max_iters=1000, min_iters=1000, sampler=0)
            rec['gc_python_call_returned'] = True
        finally:
            rec['gc_ns'] = time.perf_counter_ns() - native_start
        rec['stage'] = 'GC_RETURNED'
        rec['gc_outer_facts'] = native_facts(raw)
        rec['gc_raw_capture'] = 'UNENCODABLE_CONTENT_UNKNOWN'
        rec['gc_raw'] = tagged(raw, arrays, ident + '_gc')
        rec['gc_raw_capture'] = 'LOSSLESS'
        require(type(raw) is tuple and len(raw) == 2, 'GC return tuple')
        require(np.array_equal(points, inputs['points']) and np.array_equal(K0, inputs['K0'])
                and np.array_equal(K1, inputs['K1']), 'GC inputs unmutated')
        E, mask = raw
        require(isinstance(mask, np.ndarray) and mask.dtype.kind in 'biu' and
                mask.shape in ((len(points),), (len(points), 1)) and
                np.isin(mask, [0, 1]).all(), 'GC mask schema')
        if E is not None:
            require(isinstance(E, np.ndarray) and E.dtype.kind == 'f' and E.ndim == 2
                    and E.shape[1] == 3 and E.shape[0] >= 3 and E.shape[0] % 3 == 0
                    and np.isfinite(E).all(), 'essential schema')
            n0, n1 = normalize(p0, K0), normalize(p1, K1)
            inputs.update(normalized0=n0.copy(), normalized1=n1.copy(), identity=np.eye(3))
            for key in ('normalized0', 'normalized1', 'identity'):
                rec['inputs'][key] = tagged(inputs[key], arrays, ident + '_input_' + key)
            best = 0
            for index, essential in enumerate(np.split(E, len(E) // 3)):
                supplied = np.asarray(mask, dtype=np.uint8).reshape(-1, 1).copy()
                before = supplied.copy()
                essential_arg = essential.copy()
                prefix = ident + '_rec' + str(index)
                recovery = {'E_index': index, 'complete': False,
                            'mask_input': tagged(before, arrays, prefix + '_mask_before')}
                rec['recoveries'].append(recovery)
                recovery.update(python_call_attempted=True, python_call_returned=False,
                                raw_capture='NO_RETURN_OBSERVED')
                rec['stage'] = 'RECOVERY_PYTHON_CALL_ATTEMPTED'
                native_start = time.perf_counter_ns()
                try:
                    recovered = cv2.recoverPose(essential_arg, n0, n1, np.eye(3), 1e9, mask=supplied)
                    recovery['python_call_returned'] = True
                finally:
                    recovery['native_ns'] = time.perf_counter_ns() - native_start
                    recovery['mask_after'] = tagged(supplied, arrays, prefix + '_mask_after')
                rec['stage'] = 'RECOVERY_RETURNED'
                recovery['outer_facts'] = native_facts(recovered)
                recovery['raw_capture'] = 'UNENCODABLE_CONTENT_UNKNOWN'
                recovery['raw'] = tagged(recovered, arrays, prefix + '_raw')
                recovery['raw_capture'] = 'LOSSLESS'
                require(type(recovered) is tuple and len(recovered) in (4, 5), 'recovery tuple')
                require(np.array_equal(essential_arg, essential) and
                        np.array_equal(n0, inputs['normalized0']) and
                        np.array_equal(n1, inputs['normalized1']), 'recovery operands unmutated')
                n, R, t, mout = recovered[:4]
                require(type(n) is int and 0 <= n <= len(points), 'native cheirality count')
                require(isinstance(R, np.ndarray) and R.shape == (3, 3) and R.dtype.kind == 'f'
                        and np.isfinite(R).all() and np.allclose(R.T @ R, np.eye(3), atol=1e-5, rtol=0)
                        and abs(float(np.linalg.det(R)) - 1.) <= 1e-5, 'native proper rotation')
                require(isinstance(t, np.ndarray) and t.shape in ((3,), (3, 1)) and
                        t.dtype.kind == 'f' and np.isfinite(t).all(), 'native translation')
                require(isinstance(mout, np.ndarray) and mout.dtype == np.uint8 and
                        mout.shape == (len(points), 1) and np.isin(mout, [0, 1, 255]).all()
                        and int(np.count_nonzero(mout)) == n, 'native recovered mask')
                require(((mout.reshape(-1) == 0) | (mask.reshape(-1) != 0)).all(),
                        'recovered mask subset of GC mask')
                if len(recovered) == 5:
                    require(isinstance(recovered[4], np.ndarray) and recovered[4].dtype.kind == 'f'
                            and recovered[4].shape == (4, len(points)), 'unused triangulated schema')
                recovery['complete'] = True
                if n > best:
                    best = n
                    rec['selected_index'], rec['status'] = index, 'OK'
    if rec['status'] == 'INCOMPLETE':
        rec['status'] = 'FAILED'
    rec.update(complete=True, stage='COMPLETE')
    rec['timing_ns'] = time.perf_counter_ns() - start
    return rec


def no_effect(pair_records):
    differences = {arm: [] for arm in ('fast', 'cosine', 'margin', 'spatial_margin')}
    for pair in pair_records:
        masks = pair['primary']['masks']
        for arm in differences:
            differences[arm].append(masks['ar'] != masks[arm])
    scenes = sorted({p['scene'] for p in pair_records})
    eligible = (all(any(differences['fast'][i] for i, p in enumerate(pair_records)
                        if p['scene'] == scene) for scene in scenes)
                and all(sum(differences[arm]) >= 2 for arm in ('cosine', 'margin', 'spatial_margin')))
    return not eligible, differences


def timing_summary(pair_records):
    import numpy as np
    output = {}
    for arm in ('ar', 'cosine', 'margin', 'spatial_margin'):
        times = [(r['shared'] + r[arm + '_branch']) / 1e6 for p in pair_records for r in p['timed_ns']]
        require(len(times) == 40 and all(t >= 0 for t in times), 'forty timings')
        output[arm] = {'all_ms': times, 'median_ms': float(np.quantile(times, .5, method='linear')),
                       'p95_ms': float(np.quantile(times, .95, method='linear'))}
    return output


def ground_truth(root, pairs):
    import numpy as np
    raw = (Path(root) / GT).read_bytes()
    require(sha(raw) == GT_SHA, 'GT authority hash')
    data = json.loads(raw)
    require(type(data) is list and len(data) == 1500, 'GT master schema')
    output = []
    for pair in pairs:
        indices = [i for i, record in enumerate(data) if record['pair_names'] == pair['pair_names']]
        require(len(indices) == 1 and indices[0] == pair['legacy_metadata_pair_id'], 'unique directed GT join')
        i = indices[0]
        for key in ('K0', 'K1'):
            require(np.array_equal(np.asarray(data[i][key]), np.asarray(pair[key])), 'GT calibration join')
        T = np.asarray(data[i]['T_0to1'], dtype=np.float64)
        require(T.shape == (4, 4) and np.isfinite(T).all() and
                np.array_equal(T[3], [0., 0., 0., 1.]), 'directed transform schema')
        output.append(T)
    return output


def score(native, truths, pair_records, timing):
    import numpy as np
    errors = np.full((8, 5, 6), 180., dtype=np.float64)
    success = np.zeros((8, 5, 6), dtype=bool)
    detailed = []
    for p, ((record, arrays), T) in enumerate(zip(native, truths)):
        seen = set()
        for fit_record in record['fits']:
            value = (180., 180., 180.)
            if fit_record['status'] == 'OK':
                recovered = decoded(fit_record['recoveries'][fit_record['selected_index']]['raw'], arrays)
                value = pose_error(T, recovered[1], recovered[2])
            for arm in fit_record['arms']:
                r, a = fit_record['draw'], ARMS.index(arm)
                require((r, a) not in seen, 'one fit per arm/draw')
                seen.add((r, a))
                errors[p, r, a] = value[0]
                success[p, r, a] = fit_record['status'] == 'OK'
                detailed.append({'pair_index': p, 'draw': r, 'arm': arm, 'fit_id': fit_record['id'],
                                 'error': list(value), 'success': bool(success[p, r, a])})
        require(len(seen) == 30, 'complete six-arm five-draw scoring')
    aucs = {arm: [auc(errors[:, r, a]) for r in range(5)] for a, arm in enumerate(ARMS)}
    means = {arm: np.mean(v, axis=0).tolist() for arm, v in aucs.items()}
    scenes = {}
    for scene in sorted({p['scene'] for p in pair_records}):
        rows = [i for i, p in enumerate(pair_records) if p['scene'] == scene]
        scenes[scene] = {arm: np.mean([auc(errors[rows, r, a]) for r in range(5)], axis=0).tolist()
                         for a, arm in enumerate(ARMS)}
    counts = {arm: {'failures': int((~success[:, :, a]).sum()),
                    'catastrophes_over90': int((errors[:, :, a] > 90).sum()),
                    'rescues_vs_fast': int(((errors[:, :, a] <= 5) & (errors[:, :, 1] > 5)).sum()),
                    'damages_vs_fast': int(((errors[:, :, a] > 5) & (errors[:, :, 1] <= 5)).sum())}
              for a, arm in enumerate(ARMS)}
    gates = {'auc_each_control': all(means['ar'][0] - means[arm][0] >= .5 for arm in ARMS if arm != 'ar'),
             'four_positive_draws': sum(aucs['ar'][r][0] > aucs['fast'][r][0] for r in range(5)) >= 4,
             'each_scene_positive': all(s['ar'][0] > s['fast'][0] for s in scenes.values()),
             'failures': counts['ar']['failures'] <= counts['fast']['failures'],
             'catastrophes': counts['ar']['catastrophes_over90'] <= counts['fast']['catastrophes_over90'],
             'median_filter': timing['ar']['median_ms'] <= 10,
             'p95_filter': timing['ar']['p95_ms'] <= 25}
    return {'outcome': 'PILOT_PASS_PENDING_AUDIT' if all(gates.values()) else 'SCIENTIFIC_NEGATIVE_PENDING_AUDIT',
            'gates': gates, 'auc_by_draw': aucs, 'mean_auc': means, 'scene_mean_auc': scenes,
            'counts': counts, 'pair_draw_results': detailed}


def pilot(ctx):
    import_begin = time.perf_counter_ns()
    import numpy as np
    import scipy
    from scipy.optimize import linear_sum_assignment
    import cv2
    import pygcransac
    import_ns = time.perf_counter_ns() - import_begin
    require(np.__version__ == '1.26.4' and scipy.__version__ == '1.15.2', 'pinned numerical versions')
    root, version, write = ctx['root'], ctx['version'], ctx['write']
    io_spans = []

    def save(name, record, arrays=None, as_json=False):
        start = time.perf_counter_ns()
        raw = canonical(record) if as_json else archive(record, arrays)
        serialized = time.perf_counter_ns()
        pin = write(name, raw)
        written = time.perf_counter_ns()
        checked(root, pin)
        readback = time.perf_counter_ns()
        io_spans.append({'path': name, 'serialize_ns': serialized - start,
                         'write_ns': written - serialized, 'readback_ns': readback - written})
        return pin

    passport_begin = time.perf_counter_ns()
    passport = json.loads((Path(root) / (AR + 'passport.json')).read_bytes())
    passport_load_ns = time.perf_counter_ns() - passport_begin
    begin = import_begin
    loaded, records = [], []
    for pair in passport['pairs']:
        start = time.perf_counter_ns()
        banks = load_pair(root, pair)
        load_ns = time.perf_counter_ns() - start
        f, c, stock, fast = banks
        args = (f['descriptors0'], f['descriptors1'], fast['source_indices'],
                fast['target_indices'], fast['raw_p0'], c['hw0'])
        primary = filter_pair(*args)
        timed, solve_counts = [], [primary['solves']]
        for _ in range(5):
            repeated = filter_pair(*args)
            require({k: v for k, v in primary.items() if k != 'timings_ns'} ==
                    {k: v for k, v in repeated.items() if k != 'timings_ns'}, 'filter repeat identity')
            timed.append(repeated['timings_ns'])
            solve_counts.append(repeated['solves'])
        records.append({'pair_index': pair['pair_index'], 'scene': pair['scene'],
                        'primary': primary, 'timed_ns': timed, 'solve_counts': solve_counts,
                        'input_load_and_integrity_ns': load_ns})
        loaded.append(banks)
    require(sum(sum(p['solve_counts']) for p in records) <= 1584, 'pilot assignment solve bound')
    prediction = {'schema': 'ar-predictions-v1', 'version': version, 'pairs': records,
                  'input_pins': [r for p in passport['pairs'] for r in p['banks']],
                  'labels_read': False, 'native_calls': 0}
    prediction_pin = save(OUT + f'predictions-v{version}.npz', prediction)
    ineffective, differences = no_effect(records)
    timing = timing_summary(records)
    common = {'schema': 'ar-summary-v1', 'version': version, 'mask_differences': differences,
              'filter_timing': timing, 'numerical_import_ns': import_ns,
              'passport_load_ns': passport_load_ns,
              'full_pipeline_latency': 'UNMEASURED',
              'physical_correspondence_truth': 'UNKNOWN', 'prediction_pin': prediction_pin,
              'scientific_family_increment': 'ONLY_AFTER_INDEPENDENT_AUDIT',
              'runtime_integrity': 'PENDING_CONTROLLER_AND_AUDIT'}
    if ineffective:
        seal = {'schema': 'ar-prediction-seal-v1', 'prediction': prediction_pin, 'pose_pins': [],
                'gt_read': False, 'status': 'NO_EFFECT'}
        common.update(outcome='NO_EFFECT_PENDING_AUDIT', gt_read=False, native_calls=0)
    else:
        native, pose_pins, attempts = [], [], 0
        for p, (banks, record) in enumerate(zip(loaded, records)):
            _, camera, stock, fast = banks
            data, fits = {}, []
            rec = {'schema': 'ar-native-pose-v1', 'pair_index': p, 'prediction': prediction_pin,
                   'fits': fits, 'complete': False}
            try:
                for draw in range(5):
                    offset = (p + draw) % 6
                    order = ARMS[offset:] + ARMS[:offset]
                    groups = {}
                    for arm in order:
                        bank = stock if arm == 'stock' else fast
                        mask = np.ones(len(bank['p0']), dtype=bool) if arm in ('stock', 'fast') else np.asarray(record['primary']['masks'][arm])
                        p0, p1 = bank['p0'][mask], bank['p1'][mask]
                        identity = (p0.dtype.str, p0.shape, p0.tobytes(), p1.dtype.str, p1.shape, p1.tobytes())
                        if identity in groups:
                            groups[identity]['arms'].append(arm)
                            continue
                        ident = f'd{draw}g{len(groups)}'
                        result = {'id': ident, 'draw': draw, 'arms': [arm],
                                  'first_arm_position': order.index(arm),
                                  'coords_hash': sha(p0.tobytes() + p1.tobytes())}
                        fits.append(result)
                        groups[identity] = result
                        attempts += int(len(p0) >= 5)
                        fit(p0, p1, camera, data, ident, result)
            except BaseException as failure:
                rec['failure'] = {'type': type(failure).__name__, 'message': str(failure)[:1024],
                                  'scientific_status': 'INVALID_UNSCORED'}
                try:
                    save(OUT + f'poses-v{version}-{p:04d}.npz', rec, data)
                except BaseException as evidence_failure:
                    print(canonical({'event': 'INCOMPLETE_EVIDENCE_SAVE_FAILED', 'pair_index': p,
                                     'error_type': type(evidence_failure).__name__,
                                     'original_error_type': type(failure).__name__}).decode(), flush=True)
                raise
            rec['complete'] = True
            pin = save(OUT + f'poses-v{version}-{p:04d}.npz', rec, data)
            pose_pins.append(pin)
            native.append((rec, data))
        require(attempts <= 240, 'GC attempt bound')
        seal = {'schema': 'ar-prediction-seal-v1', 'prediction': prediction_pin, 'pose_pins': pose_pins,
                'gt_read': False, 'status': 'ALL_NATIVE_OUTPUTS_SEALED'}
        common['native_calls'] = attempts
    seal_pin = save(OUT + f'prediction-seal-v{version}.json', seal, as_json=True)
    readback_begin = time.perf_counter_ns()
    for pin in [prediction_pin, seal_pin] + seal['pose_pins']:
        checked(root, pin)
    common['prelabel_readback_ns'] = time.perf_counter_ns() - readback_begin
    if not ineffective:
        start = time.perf_counter_ns()
        truths = ground_truth(root, passport['pairs'])
        common.update(score(native, truths, records, timing), gt_read=True,
                      scoring_ns=time.perf_counter_ns() - start)
    common.update(seal_pin=seal_pin, producer_elapsed_ns=time.perf_counter_ns() - begin,
                  io_spans_before_summary=list(io_spans),
                  summary_io_location='controller job result: all_io_spans')
    summary_pin = save(OUT + f'summary-v{version}.json', common, as_json=True)
    return {'status': 'PRODUCER_FINISHED_PENDING_AUDIT', 'outcome': common['outcome'],
            'summary': summary_pin, 'gt_read': common['gt_read'], 'native_calls': common['native_calls'],
            'all_io_spans': io_spans, 'total_producer_ns': time.perf_counter_ns() - begin}


def main(stage, ctx):
    require(stage == 'pilot', 'producer stage')
    return pilot(ctx)
