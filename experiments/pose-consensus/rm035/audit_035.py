"""Independent RM035 saved-selection audit; pure stdlib, no file/data loader.

No selector or production metric implementation is imported. The caller must
authenticate complete pool/selection seals before providing saved errors.
This module independently checks selection identities and every reported
numeric table, bootstrap replicate and frozen gate. It does not authenticate
the external files/NPY payloads referenced by a supplied source dictionary.
"""
import hashlib
import json
import math
import random


ARMS = ('FIRST', 'MEDOID', 'MAX')
DRAWS = 10
ROOMS = tuple(range(707, 727))
TOLERANCE = 1e-10


def _need(condition, label):
    if not condition:
        raise ValueError('Independent035: ' + label)


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _wire(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'),
                       ensure_ascii=True, allow_nan=False) + '\n').encode('ascii')


def _eligible(hypothesis):
    if hypothesis['success'] is not True:
        return False
    r, t = hypothesis['R'], hypothesis['t']
    if not all(_number(v) for v in r + t):
        return False
    # Preserve the old qualified scalar ROW-Gram predicate, not a new test.
    for i in range(3):
        for j in range(3):
            value = sum(r[3*i+k] * r[3*j+k] for k in range(3))
            if not math.isfinite(value) or abs(value - (1.0 if i == j else 0.0)) > 1e-5:
                return False
    det = (r[0]*(r[4]*r[8]-r[5]*r[7])
           - r[1]*(r[3]*r[8]-r[5]*r[6])
           + r[2]*(r[3]*r[7]-r[4]*r[6]))
    norm = math.sqrt(sum(v*v for v in t))
    return (math.isfinite(det) and abs(det-1.0) <= 1e-5
            and math.isfinite(norm) and norm > 2.220446049250313e-16)


def _distance(a, b):
    # trace(a.T @ b) is the Frobenius inner product of the flat matrices.
    cosine = (sum(x*y for x, y in zip(a, b)) - 1.0) / 2.0
    # Canonical radians also preserve the specified exact floating score ties.
    return math.acos(max(-1.0, min(1.0, cosine)))


def _choices(hypotheses):
    eligibility = [_eligible(h) for h in hypotheses]
    valid = [i for i, value in enumerate(eligibility) if value]
    chosen = dict(FIRST=0 if eligibility[0] else None, MEDOID=None, MAX=None)
    if valid:
        costs = {i: sum(_distance(hypotheses[i]['R'], hypotheses[j]['R'])
                        for j in valid if j != i) for i in valid}
        chosen['MEDOID'] = min(valid, key=lambda i: (costs[i], i))
        chosen['MAX'] = max(valid, key=lambda i: (hypotheses[i]['gc_inliers'], -i))
    return eligibility, chosen


def _auc(errors, threshold):
    """Independent segment integration of the qualified recall trapezoids."""
    _need(errors, 'AUC denominator cannot be empty')
    previous_x, previous_y = 0.0, 0.0
    segments = []
    count = len(errors)
    for rank, value in enumerate(sorted(errors), 1):
        if value >= threshold:
            break
        recall = rank / count
        segments.append((value-previous_x) * (previous_y+recall) / 2.0)
        previous_x, previous_y = value, recall
    segments.append((threshold-previous_x) * previous_y)
    return 100.0 * math.fsum(segments) / threshold


def _percentile(values, probability):
    ordered = sorted(values)
    position = probability * (len(ordered)-1)
    left = int(math.floor(position))
    right = int(math.ceil(position))
    return ordered[left] + (ordered[right]-ordered[left]) * (position-left)


def _statistics(values, failures, indices):
    per_draw = []
    for draw in range(DRAWS):
        errors = [values[i][draw] for i in indices]
        row = dict(repeat=draw, num_pairs=len(indices),
                   failed_poses=sum(failures[i][draw] for i in indices),
                   severe_pose_errors_gt90=sum(e > 90 for e in errors))
        for threshold in (5, 10, 20):
            row['auc' + str(threshold)] = _auc(errors, threshold)
        per_draw.append(row)
    result = dict(num_pairs=len(indices), num_repeats=DRAWS,
                  pose_attempts=len(indices)*DRAWS,
                  failed_poses=sum(failures[i][d] for i in indices for d in range(DRAWS)),
                  severe_pose_errors_gt90=sum(values[i][d] > 90 for i in indices for d in range(DRAWS)),
                  per_draw=per_draw)
    for threshold in (5, 10, 20):
        name = 'auc' + str(threshold)
        mean = math.fsum(row[name] for row in per_draw) / DRAWS
        result[name + '_mean'] = mean
        result[name + '_std'] = math.sqrt(math.fsum((row[name]-mean)**2 for row in per_draw) / DRAWS)
    return result


def _bootstrap(values, rooms):
    members = {room: [i for i, actual in enumerate(rooms) if actual == room] for room in ROOMS}
    rng = random.Random(20261008)
    digest = hashlib.sha256()
    replicates = {arm: [] for arm in ARMS}
    for _ in range(1000):
        sampled_rooms = [rng.choice(ROOMS) for _ in ROOMS]
        digest.update((json.dumps(sampled_rooms, separators=(',', ':')) + '\n').encode('ascii'))
        selected = [i for room in sampled_rooms for i in members[room]]
        for arm in ARMS:
            per_draw = [_auc([values[arm][i][d] for i in selected], 5) for d in range(DRAWS)]
            replicates[arm].append(math.fsum(per_draw)/DRAWS)
    differences = {arm: [x-y for x, y in zip(replicates[arm], replicates['FIRST'])]
                   for arm in ('MEDOID', 'MAX')}
    intervals = {arm: [_percentile(v, .025), _percentile(v, .975)]
                 for arm, v in differences.items()}
    return dict(seed=20261008, replicates=1000,
                generator='Python3.11 random.Random; one choice per sampled room',
                sorted_rooms=list(ROOMS), room_sample_stream_sha256=digest.hexdigest(),
                pairs=300, draws=DRAWS, shared_room_samples_all_arms=True,
                all_pair_draw_multiplicity_retained=True,
                contrast_baseline='FIRST', primary='MEDOID',
                delta_auc5_ci95=intervals, primary_ci95=intervals['MEDOID'],
                delta_auc5_replicates=differences, arm_auc5_replicates=replicates)


def _compare(actual, expected, path, measured):
    if type(expected) is dict:
        _need(type(actual) is dict and set(actual) == set(expected), path + ': exact dictionary keys')
        for key in expected:
            _compare(actual[key], expected[key], path + '.' + str(key), measured)
    elif type(expected) is list:
        _need(type(actual) is list and len(actual) == len(expected), path + ': list shape')
        for index, (a, b) in enumerate(zip(actual, expected)):
            _compare(a, b, path + '[' + str(index) + ']', measured)
    elif type(expected) is float:
        _need(_number(actual), path + ': finite numeric value')
        difference = abs(actual-expected)
        measured['numeric_fields_checked'] += 1
        measured['numeric_fields_exactly_equal'] += actual == expected
        measured['maximum_absolute_metric_difference'] = max(measured['maximum_absolute_metric_difference'], difference)
        _need(difference <= TOLERANCE, path + ': independent numeric mismatch')
    else:
        _need(type(actual) is type(expected) and actual == expected, path + ': exact typed value')
        measured['exact_fields_checked'] += 1


def audit(pool, selection, scored, original_errors):
    """Recompute all fixed cached-stage identities, statistics and both gates."""
    _need(type(pool) is dict and pool.get('status') == 'GT_FREE_B_POOL_SEALED'
          and type(pool.get('records')) is list and len(pool['records']) == 300,
          'complete sealed 300-pair GT-free pool')
    _need(type(selection) is dict and selection.get('status') == 'ALL_GLOBAL_SELECTIONS_SEALED'
          and selection.get('GT_errors_read') is False
          and selection.get('whole_hypotheses_only') is True
          and type(selection.get('rows')) is list and len(selection['rows']) == 3000,
          'complete global selection barrier before supplied saved errors')
    _need(all(type(selection.get(k)) is int and selection[k] == v for k, v in
              (('logical_draws', 10), ('triple_size', 3), ('logical_rows_per_arm', 3000),
               ('unique_cached_fits_per_pair', 10))) and selection.get('overlapping_triples') is True,
          'cached cyclic readout counts and dependence are explicit')
    _need(type(original_errors) is dict and set(original_errors) == set(range(300)),
          'all 300 directed source identities in saved error input')
    measured = dict(numeric_fields_checked=0, numeric_fields_exactly_equal=0,
                    exact_fields_checked=0, maximum_absolute_metric_difference=0.0)
    values = {arm: [[None]*DRAWS for _ in range(300)] for arm in ARMS}
    failures = {arm: [[None]*DRAWS for _ in range(300)] for arm in ARMS}
    rooms = []
    source_checks = 0
    for pair, record in enumerate(pool['records']):
        _need(type(record) is dict and type(record.get('pair_index')) is int
              and type(record.get('source_pair_index')) is int
              and record['pair_index'] == record['source_pair_index'] == pair
              and type(record.get('room')) is int and record['room'] in ROOMS,
              'ordered pair/source/physical-room identity')
        rooms.append(record['room'])
        hypotheses = record.get('hypotheses')
        _need(type(hypotheses) is list and len(hypotheses) == DRAWS
              and type(original_errors[pair]) is dict
              and set(original_errors[pair]) == set(range(DRAWS)), 'all ten physical hypotheses and errors')
        for draw, hypothesis in enumerate(hypotheses):
            _need(type(hypothesis) is dict and type(hypothesis.get('success')) is bool,
                  'typed original success flag')
            for key, length in (('R', 9), ('t', 3)):
                _need(type(hypothesis.get(key)) is list and len(hypothesis[key]) == length
                      and all(type(v) in (int, float) for v in hypothesis[key]), 'flat whole-pose field: ' + key)
            essential = hypothesis.get('E')
            _need(type(essential) is list and len(essential) % 9 == 0
                  and all(_number(v) for v in essential)
                  and (essential or hypothesis['success'] is False), 'finite original essential stack supply')
            _need(type(hypothesis.get('gc_inliers')) is int and hypothesis['gc_inliers'] >= 0
                  and type(hypothesis.get('source')) is dict and hypothesis['source'], 'actual GC count and whole source mapping')
            source = hypothesis['source']
            _need(all(type(source.get(k)) is int for k in ('pair_index', 'source_pair_index', 'physical_draw'))
                  and source.get('pair_index') == source.get('source_pair_index') == pair
                  and source.get('physical_draw') == draw
                  and source.get('method') == 'xfeat_star_xrefine'
                  and source.get('policy') == 'conditional_depth'
                  and source.get('gc_prefix') == 'm3_r' + str(draw)
                  and source.get('pose_prefix') == 'c3_r' + str(draw), 'original whole-pose source identity')
            _wire(source)
            error = original_errors[pair][draw]
            _need(type(error) is dict and type(error.get('success')) is bool
                  and error['success'] == hypothesis['success'], 'saved error/success join')
            for key in ('pose_error_deg', 'rotation_error_deg', 'translation_error_deg'):
                _need(_number(error.get(key)) and 0 <= error[key] <= 180, 'finite saved angular error')
            if error['success']:
                _need(abs(error['pose_error_deg'] - max(error['rotation_error_deg'], error['translation_error_deg'])) <= TOLERANCE,
                      'saved pose error is maximum rotation/translation error')
            else:
                _need(all(error[k] == 180 for k in ('pose_error_deg', 'rotation_error_deg', 'translation_error_deg')),
                      'all failed saved error fields remain 180')
        for draw in range(DRAWS):
            row = selection['rows'][pair*DRAWS+draw]
            triple = [draw, (draw+1) % DRAWS, (draw+2) % DRAWS]
            _need(type(row) is dict and type(row.get('pair_index')) is int
                  and type(row.get('room')) is int and type(row.get('draw')) is int
                  and (row['pair_index'], row['room'], row['draw']) == (pair, record['room'], draw)
                  and type(row.get('triple')) is list and all(type(v) is int for v in row['triple'])
                  and row['triple'] == triple, 'all ordered cyclic logical identities')
            eligibility, local = _choices([hypotheses[j] for j in triple])
            _need(type(row.get('eligible')) is list and all(type(v) is bool for v in row['eligible'])
                  and row['eligible'] == eligibility, 'independent qualified eligibility')
            expected_indices = {arm: None if index is None else triple[index] for arm, index in local.items()}
            _need(type(row.get('selected')) is dict and set(row['selected']) == set(ARMS)
                  and all(v is None or type(v) is int for v in row['selected'].values())
                  and row['selected'] == expected_indices, 'independent FIRST/medoid/MAX selection')
            _need(type(row.get('selected_sources')) is dict and set(row['selected_sources']) == set(ARMS),
                  'all selected whole-source mappings')
            for arm in ARMS:
                selected = expected_indices[arm]
                expected_source = None if selected is None else hypotheses[selected]['source']
                _need(_wire(row['selected_sources'][arm]) == _wire(expected_source), 'selected whole-pose source equality')
                source_checks += 1
                error = None if selected is None else original_errors[pair][selected]
                values[arm][pair][draw] = 180.0 if error is None else float(error['pose_error_deg'])
                failures[arm][pair][draw] = error is None or not error['success']
    _need(sorted(set(rooms)) == list(ROOMS)
          and all(rooms.count(room) == 15 for room in ROOMS), '20 complete physical rooms, 15 pairs each')
    all_pairs = list(range(300))
    tables = {}
    for arm in ARMS:
        table = _statistics(values[arm], failures[arm], all_pairs)
        table['per_scene'] = {str(room).zfill(4): _statistics(values[arm], failures[arm],
                              [i for i, value in enumerate(rooms) if value == room]) for room in ROOMS}
        tables[arm] = table
    bootstrap = _bootstrap(values, rooms)
    contrasts, harms = {}, {}
    for control in ('FIRST', 'MAX'):
        base, candidate = tables[control], tables['MEDOID']
        delta = candidate['auc5_mean'] - base['auc5_mean']
        per_draw = [y['auc5']-x['auc5'] for x, y in zip(base['per_draw'], candidate['per_draw'])]
        replicates = [x-y for x, y in zip(bootstrap['arm_auc5_replicates']['MEDOID'],
                                         bootstrap['arm_auc5_replicates'][control])]
        interval = [_percentile(replicates, .025), _percentile(replicates, .975)]
        gates = dict(margin=delta >= .5 if control == 'FIRST' else delta > 0,
                     positive_draws=sum(v > 0 for v in per_draw) >= 8,
                     ci_lower_positive=interval[0] > 0,
                     failures_no_increase=candidate['failed_poses'] <= base['failed_poses'],
                     severe_no_increase=candidate['severe_pose_errors_gt90'] <= base['severe_pose_errors_gt90'])
        contrasts[control] = dict(delta_auc5=delta,
                                 delta_auc10=candidate['auc10_mean']-base['auc10_mean'],
                                 delta_auc20=candidate['auc20_mean']-base['auc20_mean'],
                                 per_draw_delta=per_draw, positive_draws=sum(v > 0 for v in per_draw),
                                 ci95=interval, gates=gates, pass_all=all(gates.values()),
                                 delta_auc5_replicates=replicates)
        counts = dict.fromkeys(('angular_better', 'angular_worse', 'angular_equal',
                               'failure_rescued', 'failure_harmed', 'severe_rescued', 'severe_harmed'), 0)
        for pair in all_pairs:
            for draw in range(DRAWS):
                old, new = values[control][pair][draw], values['MEDOID'][pair][draw]
                counts['angular_better' if new < old else 'angular_worse' if new > old else 'angular_equal'] += 1
                counts['failure_rescued'] += failures[control][pair][draw] and not failures['MEDOID'][pair][draw]
                counts['failure_harmed'] += not failures[control][pair][draw] and failures['MEDOID'][pair][draw]
                counts['severe_rescued'] += old > 90 and new <= 90
                counts['severe_harmed'] += old <= 90 and new > 90
        harms[control] = counts
    _need(type(scored) is dict and scored.get('status') == 'CACHED_SCREEN_SCORED', 'actual cached score artifact')
    for key, expected in (('methods', tables), ('bootstrap', bootstrap), ('contrasts', contrasts), ('paired_harms', harms)):
        _need(key in scored, 'complete scored field: ' + key)
        _compare(scored[key], expected, key, measured)
    coverage = dict(pairs=300, rooms=20, draws=10, arms=3, logical_outcomes=9000, unique_cached_B_fits=3000)
    _compare(scored.get('coverage'), coverage, 'coverage', measured)
    _need(type(scored.get('new_native_or_model_calls')) is int and scored['new_native_or_model_calls'] == 0,
          'cached reductions introduce no declared model/native work')
    verdict = ('QUALIFIED_FOR_NEXT_STAGE' if all(row['pass_all'] for row in contrasts.values())
               else 'REJECT_FIXED_MEDOID')
    _need(scored.get('verdict') == verdict, 'both frozen contrasts determine the verdict')
    return dict(status='INDEPENDENT_AUDIT_PASS', verdict=verdict,
                independent_selection=True, independent_metrics=True,
                selector_or_production_metrics_imported=False,
                logical_selection_rows_checked=3000, selected_whole_source_checks=source_checks,
                coverage=coverage, comparison=measured,
                room_sample_stream_sha256=bootstrap['room_sample_stream_sha256'],
                recomputed_contrasts={name: {key: value for key, value in row.items()
                                      if key != 'delta_auc5_replicates'} for name, row in contrasts.items()},
                recomputed_paired_harms=harms,
                tolerance=TOLERANCE,
                scope='Independent reductions from supplied sealed projected hypotheses and validated saved errors; external-file/NPY authentication belongs to the caller and independent input review; exposed overlapping cached DEV screen only')
