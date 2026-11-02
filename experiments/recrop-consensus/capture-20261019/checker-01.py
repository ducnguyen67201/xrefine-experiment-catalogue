"""Independent identity-capture audit; no model, filesystem or scientific imports.

NumPy is supplied by the admitted caller.  Defining this module performs no
numerical work.  `validate` examines every saved observation and recomputes the
fixed gates; it does not confer resource, source or experiment admission.
"""
import math


ARMS = ('raw', 'one', 'sham', 'recrop', 'extrap', 'phase')
CONTROLS = ('one', 'extrap', 'phase')
FIELDS = ('ids', 'anchors', 'offsets') + tuple(
    arm + str(endpoint) for arm in ARMS for endpoint in (0, 1))
OFFSETS = ((0, 0), (2, 0), (-2, 0), (0, 2), (0, -2),
           (4, 0), (-4, 0), (0, 4), (0, -4),
           (8, 0), (-8, 0), (0, 8), (0, -8))
RADIUS_ROWS = {'0': (0,), '2': (1, 2, 3, 4),
               '4': (5, 6, 7, 8), '8': (9, 10, 11, 12)}
AXIS_BOUND = 3.403
BOUND_TOLERANCE = 1e-4
PARITY_TOLERANCE = 1e-5


def _require(ok, message):
    if not bool(ok):
        raise ValueError(message)


def _array(np, value, dtype, shape, label):
    _require(isinstance(value, np.ndarray), label + ': ndarray required')
    _require(value.dtype == np.dtype(dtype), label + ': wrong dtype')
    _require(value.shape == shape, label + ': wrong shape')
    _require(np.isfinite(value).all(), label + ': nonfinite value')
    return value


def _same(np, actual, expected, label):
    _require(np.array_equal(actual, expected), label + ': exact relation failed')


def _center(np, value):
    return np.floor(value.astype(np.float64)) + .5


def _bound(np, output, input_points, label):
    maximum = float(np.max(np.abs(output.astype(np.float64) -
                                 _center(np, input_points))))
    _require(maximum <= AXIS_BOUND + BOUND_TOLERANCE,
             label + ': native axis bound failed')
    return maximum


def _norm(np, value):
    return np.sqrt(np.sum(value * value, axis=-1, dtype=np.float64))


def _case_metrics(np, p0, p1, anchors):
    a = anchors.astype(np.float64)
    x = p0.astype(np.float64)
    y = p1.astype(np.float64)
    return {
        'error': _norm(np, x - y),
        'midpoint_drift': _norm(np, (x + y) / 2 - a),
        'endpoint_drift': (_norm(np, x - a) + _norm(np, y - a)) / 2,
    }


def _summary(error, midpoint, endpoint):
    """Scalar accumulation independent of the actor's NumPy/sum producer."""
    count = len(error)
    _require(count > 0 and count == len(midpoint) == len(endpoint),
             'metric denominator mismatch')
    ordered = sorted(error)
    middle = count // 2
    median = (ordered[middle] if count % 2 else
              (ordered[middle - 1] + ordered[middle]) / 2)
    return {
        'count': count,
        'mean_error': math.fsum(error) / count,
        'median_error': median,
        'recall_le_0_5': sum(v <= .5 for v in error) / count,
        'recall_le_1': sum(v <= 1 for v in error) / count,
        'recall_le_3': sum(v <= 3 for v in error) / count,
        'mean_midpoint_drift': math.fsum(midpoint) / count,
        'mean_endpoint_drift': math.fsum(endpoint) / count,
    }


def _table(metrics, rows, image_ids=range(8)):
    table = {}
    for arm in ARMS:
        values = {key: [] for key in ('error', 'midpoint_drift', 'endpoint_drift')}
        for image in image_ids:
            for key in values:
                values[key].extend(metrics[image][arm][key][list(rows)].ravel().tolist())
        table[arm] = _summary(values['error'], values['midpoint_drift'],
                              values['endpoint_drift'])
    return table


def validate(np, arrays, image_hws, cached_keypoints):
    """Validate the exact 120-array archive and return a compact scalar report.

    Coordinates and all three error/drift arrays are recomputed for every case.
    The sealed point archive retains the individual observations; this report
    avoids duplicating those arrays as large JSON lists.  `advance` means only
    that the fixed numerical gates passed, conditional on external admission,
    resource closure and the independent fresh-read replay also passing.
    """
    expected_keys = {f'i{i}_{field}' for i in range(8) for field in FIELDS}
    _require(set(arrays.keys()) == expected_keys, 'archive key set mismatch')
    _require(len(image_hws) == len(cached_keypoints) == 8,
             'exactly eight image entries required')
    offsets = np.asarray(OFFSETS, dtype=np.float32)
    metrics = []
    bounds = {'one': 0.0, 'sham': 0.0, 'recrop': 0.0}
    parity = 0.0
    lower_bound = 8 - 2 * (AXIS_BOUND + BOUND_TOLERANCE)
    stress_min = {'one': math.inf, 'sham': math.inf}
    selection = []
    mechanisms = []
    for image in range(8):
        prefix = f'i{image}_'
        dims = image_hws[image]
        _require(len(dims) == 2 and all(
            isinstance(v, (int, np.integer)) and not isinstance(v, bool)
            for v in dims), prefix + 'invalid raster dimensions')
        height, width = map(int, dims)
        _require(height > 48 and width > 48,
                 prefix + 'raster has no eligible interior')
        bank = cached_keypoints[image]
        _require(isinstance(bank, np.ndarray) and bank.ndim == 2 and
                 bank.shape[1] == 2 and bank.shape[0] >= 32,
                 prefix + 'invalid full cached bank shape')
        _array(np, bank, 'float32', bank.shape, prefix + 'cached keypoints')
        all_anchors = np.floor(bank) + np.float32(.5)
        eligible = ((all_anchors[:, 0] >= 24) &
                    (all_anchors[:, 0] < width - 24) &
                    (all_anchors[:, 1] >= 24) &
                    (all_anchors[:, 1] < height - 24))
        selected = np.flatnonzero(eligible)[:32].astype(np.int64)
        _require(len(selected) == 32, prefix + 'fewer than 32 eligible anchors')
        ids = _array(np, arrays[prefix + 'ids'], 'int64', (32,), prefix + 'ids')
        anchors = _array(np, arrays[prefix + 'anchors'], 'float32', (32, 2),
                         prefix + 'anchors')
        stored_offsets = _array(np, arrays[prefix + 'offsets'], 'float32',
                                (13, 2), prefix + 'offsets')
        _same(np, ids, selected, prefix + 'first eligible native IDs')
        _same(np, anchors, all_anchors[selected], prefix + 'cached anchors')
        _same(np, stored_offsets, offsets, prefix + 'fixed offset ordering')
        points = {}
        for arm in ARMS:
            for end in (0, 1):
                name = arm + str(end)
                points[name] = _array(np, arrays[prefix + name], 'float32',
                                      (13, 32, 2), prefix + name)
        raw0 = np.broadcast_to(anchors, (13, 32, 2))
        raw1 = (raw0 + offsets[:, None, :]).astype(np.float32)
        _same(np, points['raw0'], raw0, prefix + 'raw source identity')
        _same(np, points['raw1'], raw1, prefix + 'raw target offset identity')
        for end in (0, 1):
            raw = points['raw' + str(end)]
            one = points['one' + str(end)]
            sham = points['sham' + str(end)]
            recrop = points['recrop' + str(end)]
            maximum = float(np.max(np.abs(one.astype(np.float64) -
                                         sham.astype(np.float64))))
            parity = max(parity, maximum)
            _require(maximum <= PARITY_TOLERANCE,
                     prefix + 'one/sham parity failed')
            bounds['one'] = max(bounds['one'], _bound(np, one, raw, prefix + 'one'))
            bounds['sham'] = max(bounds['sham'], _bound(np, sham, raw, prefix + 'sham'))
            bounds['recrop'] = max(bounds['recrop'],
                                    _bound(np, recrop, one, prefix + 'recrop'))
            raw64 = raw.astype(np.float64)
            one64 = one.astype(np.float64)
            extrap = (raw64 + 2 * (one64 - raw64)).astype(np.float32)
            phase = (np.floor(one64) + .5 + (one64 - raw64)).astype(np.float32)
            _same(np, points['extrap' + str(end)], extrap,
                  prefix + 'FP64 extrapolation/FP32 cast')
            _same(np, points['phase' + str(end)], phase,
                  prefix + 'FP64 phase extrapolation/FP32 cast')
        image_metrics = {arm: _case_metrics(np, points[arm + '0'],
                                            points[arm + '1'], anchors)
                         for arm in ARMS}
        expected_raw_error = _norm(np, offsets.astype(np.float64))
        _same(np, image_metrics['raw']['error'],
              np.broadcast_to(expected_raw_error[:, None], (13, 32)),
              prefix + 'known raw error')
        _require(float(np.max(image_metrics['raw']['error'][0])) == 0.0,
                 prefix + 'zero-offset raw identity failed')
        for arm in stress_min:
            minimum = float(np.min(image_metrics[arm]['error'][9:13]))
            stress_min[arm] = min(stress_min[arm], minimum)
            _require(minimum + 1e-10 >= lower_bound,
                     prefix + arm + ': analytic 8px lower bound failed')
        changed0 = np.any(np.floor(points['one0']) != np.floor(points['raw0']), axis=-1)
        changed1 = np.any(np.floor(points['one1']) != np.floor(points['raw1']), axis=-1)
        mechanisms.append({
            'image': image,
            'changed_source_crop_count': int(np.count_nonzero(changed0)),
            'changed_target_crop_count': int(np.count_nonzero(changed1)),
            'changed_either_crop_count': int(np.count_nonzero(changed0 | changed1)),
            'changed_both_crops_count': int(np.count_nonzero(changed0 & changed1)),
            'by_condition_either': [int(np.count_nonzero(row))
                                    for row in changed0 | changed1],
        })
        metrics.append(image_metrics)
        selection.append({'image': image, 'height': height, 'width': width,
                          'eligible_count': int(np.count_nonzero(eligible)),
                          'selected_ids': ids.tolist()})
    radius_metrics = {radius: _table(metrics, rows)
                      for radius, rows in RADIUS_ROWS.items()}
    primary_rows = tuple(range(1, 9))
    primary = _table(metrics, primary_rows)
    all_cases = _table(metrics, tuple(range(13)))
    conditions = [{'offset': list(offset), 'metrics': _table(metrics, (row,))}
                  for row, offset in enumerate(OFFSETS)]
    per_image = [{'image': image,
                  'primary': _table(metrics, primary_rows, (image,)),
                  'radius_mean_errors': {
                      radius: {arm: math.fsum(
                          metrics[image][arm]['error'][list(rows)].ravel().tolist()) /
                          (32 * len(rows)) for arm in ARMS}
                      for radius, rows in RADIUS_ROWS.items()}}
                 for image in range(8)]
    checks = {}
    comparisons = {}
    for control in CONTROLS:
        gain = primary[control]['mean_error'] - primary['recrop']['mean_error']
        positives = sum(row['primary']['recrop']['mean_error'] <
                        row['primary'][control]['mean_error'] for row in per_image)
        checks['mean_gain_vs_' + control] = gain >= .10
        checks['positive_images_vs_' + control] = positives >= 6
        comparisons[control] = {'mean_error_gain_px': gain,
                                'positive_image_entries': positives,
                                'required_mean_gain_px': .10,
                                'required_positive_image_entries': 6}
    zero = radius_metrics['0']
    zero_error_increase = zero['recrop']['mean_error'] - zero['one']['mean_error']
    zero_drift_increase = (zero['recrop']['mean_endpoint_drift'] -
                          zero['one']['mean_endpoint_drift'])
    checks['zero_error_noninferiority'] = zero_error_increase <= .05
    checks['zero_endpoint_drift_noninferiority'] = zero_drift_increase <= .05
    numerical_pass = all(checks.values())
    return {
        'schema': 'prof-capture-independent-audit-v1',
        'status': 'PASS', 'complete': True, 'advance': numerical_pass,
        'advance_scope': 'Numerical gates only; external source/resource/closure and fresh-read audit required. A pass admits only separately designed real-cross-view research.',
        'cases_checked': 3328, 'primary_cases': 2048, 'arrays_checked': 120,
        'selection': selection,
        'integrity': {'one_sham_max_abs_px': parity,
                      'native_max_axis_shift_px': bounds,
                      'safe_axis_bound_px': AXIS_BOUND,
                      'bound_tolerance_px': BOUND_TOLERANCE,
                      'stress8_onepass_lower_bound_px': lower_bound,
                      'stress8_min_error_px': stress_min},
        'radius_metrics': radius_metrics, 'primary_metrics': primary,
        'all_case_metrics': all_cases, 'condition_metrics': conditions,
        'per_image_metrics': per_image, 'mechanism': mechanisms,
        'gates': {'checks': checks, 'all_numerical_pass': numerical_pass,
                  'primary_comparisons': comparisons,
                  'zero_error_increase_px': zero_error_increase,
                  'zero_mean_endpoint_drift_increase_px': zero_drift_increase,
                  'allowed_zero_increase_px': .05,
                  'stress8_excluded_from_promotion': True},
    }


def fixtures(np):
    """Run real hand checks when called by the admitted actor, never on import."""
    origin = np.asarray([[0., 0.]], dtype=np.float32)
    identity = _case_metrics(np, origin, origin, origin)
    _require(float(identity['error'][0]) == 0 and
             float(identity['midpoint_drift'][0]) == 0 and
             float(identity['endpoint_drift'][0]) == 0, 'identity fixture')
    point = np.asarray([[3., 4.]], dtype=np.float32)
    triangle = _case_metrics(np, origin, point, origin)
    _require(float(triangle['error'][0]) == 5, '3-4-5 fixture')
    moving = _case_metrics(np, point, point, origin)
    _require(float(moving['error'][0]) == 0 and
             float(moving['midpoint_drift'][0]) == 5 and
             float(moving['endpoint_drift'][0]) == 5,
             'co-movement drift fixture')
    opposite = _case_metrics(np, point, -point, origin)
    _require(float(opposite['error'][0]) == 10 and
             float(opposite['midpoint_drift'][0]) == 0 and
             float(opposite['endpoint_drift'][0]) == 5,
             'opposite endpoint drift fixture')
    epsilon = 1e-6
    q = math.exp(-2)
    minimum = -5 * (3 - 3 * q + epsilon) / (3 + 6 * q + epsilon)
    maximum = 5 * (3 - 3 * q - epsilon) / (3 + 6 * q + epsilon)
    for high_column, expected in ((0, minimum), (2, maximum)):
        weights = [[1.0 if column == high_column else q for column in range(3)]
                   for _ in range(3)]
        denominator = math.fsum(v for row in weights for v in row) + epsilon
        actual = 5 * (math.fsum(column * weights[row][column]
                               for row in range(3) for column in range(3)) /
                      denominator - 1)
        _require(abs(actual - expected) < 1e-12, 'epsilon-bound fixture')
    uniform = 5 * (9 / (9 + epsilon) - 1)
    _require(abs(uniform - (-5 * epsilon / (9 + epsilon))) < 1e-12 and
             max(abs(minimum), abs(maximum)) < AXIS_BOUND,
             'epsilon asymmetry/safe-bound fixture')
    # Exercise the full verifier on a legitimate non-improving synthetic archive.
    bank = np.zeros((2048, 2), dtype=np.float32)
    bank[:32, 0] = np.arange(32, dtype=np.float32) + np.float32(32)
    bank[:32, 1] = np.float32(64)
    anchor = np.floor(bank[:32]) + np.float32(.5)
    offsets = np.asarray(OFFSETS, dtype=np.float32)
    raw0 = np.broadcast_to(anchor, (13, 32, 2)).copy()
    raw1 = (raw0 + offsets[:, None, :]).astype(np.float32)
    archive = {}
    for image in range(8):
        archive[f'i{image}_ids'] = np.arange(32, dtype=np.int64)
        archive[f'i{image}_anchors'] = anchor.copy()
        archive[f'i{image}_offsets'] = offsets.copy()
        for arm in ARMS:
            archive[f'i{image}_{arm}0'] = raw0.copy()
            archive[f'i{image}_{arm}1'] = raw1.copy()
    valid = validate(np, archive, [(128, 128)] * 8, [bank] * 8)
    _require(valid['status'] == 'PASS' and valid['complete'] and
             not valid['advance'], 'full valid-negative fixture')
    archive['i0_raw1'][0, 0, 0] += np.float32(1)
    rejected = False
    try:
        validate(np, archive, [(128, 128)] * 8, [bank] * 8)
    except ValueError as error:
        rejected = 'raw target offset identity' in str(error)
    _require(rejected, 'tampered relation was not rejected for the intended reason')
    return {'status': 'PASS', 'complete': True,
            'checks': ['identity', '3-4-5 error', 'co-movement and endpoint drift',
                       'epsilon-aware asymmetric bound',
                       'full valid-negative archive',
                       'tampered raw relation rejection'],
            'analytic_axis_minimum_px': minimum,
            'analytic_axis_maximum_px': maximum}
