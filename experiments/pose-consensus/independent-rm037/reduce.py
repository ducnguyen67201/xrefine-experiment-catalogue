"""Independent RM037 scalar replay; SOURCE ONLY, author executed no target code.

No producer module is imported and no file is opened.  The root auditor owns
all authentication, source association, pool ordering and score-row joins.

choose(hs) receives three ordered whole-hypothesis dictionaries and returns
local indices FIRST/MEDOID/MAX, or None.  It never changes a hypothesis.

calculate(values, failed, rooms) receives dictionaries keyed by ARMS.  Each
value is a 300-by-10 list/tuple of saved scalar pose errors; each failure array
has the same shape and True means an actual failed estimate.  rooms contains
300 integer physical-room IDs.  The result includes independent statistics,
both fixed decisions, paired severe tables and every bootstrap replicate.

compare(actual, expected) checks the same structure recursively: integer
indices/counts, Boolean decisions, identities and keys are EXACT; finite
floating quantities have absolute agreement tolerance 1e-10.  That tolerance
never changes selection, threshold decisions, signs or reported numbers.
"""
import hashlib
import json
import math
import random

ARMS = ('FIRST', 'MEDOID', 'MAX')
THRESHOLDS = (5, 10, 20)
ROOMS = tuple(range(707, 727))
PAIRS = 300
DRAWS = 10
REPLICATES = 1000
SEED = 20261008
EPSILON = 2.220446049250313e-16
AGREEMENT = 1e-10


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite_scalar(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def vector(value, length):
    return (type(value) in (tuple, list) and len(value) == length and
            all(finite_scalar(element) for element in value))


def proper_rotation(matrix):
    """The registered scalar row-Gram/determinant predicate, without repair."""
    if not vector(matrix, 9):
        return False
    for left in range(3):
        for right in range(3):
            product = sum(matrix[3*left+k]*matrix[3*right+k] for k in range(3))
            if abs(product - int(left == right)) > 1e-5:
                return False
    determinant = (
        matrix[0]*(matrix[4]*matrix[8] - matrix[5]*matrix[7])
        - matrix[1]*(matrix[3]*matrix[8] - matrix[5]*matrix[6])
        + matrix[2]*(matrix[3]*matrix[7] - matrix[4]*matrix[6]))
    return math.isfinite(determinant) and abs(determinant - 1) <= 1e-5


def eligible(hypothesis):
    if hypothesis['success'] is not True:
        return False
    if not proper_rotation(hypothesis['R']) or not vector(hypothesis['t'], 3):
        return False
    # Preserve the registered finite-norm/epsilon rule, including overflow
    # rejection.  A more permissive rescaled normalization would change it.
    length = math.sqrt(sum(coordinate*coordinate for coordinate in hypothesis['t']))
    return math.isfinite(length) and length > EPSILON


def angle_radians(left, right):
    require(proper_rotation(left) and proper_rotation(right), 'Proper rotations required')
    trace = sum(left[index]*right[index] for index in range(9))
    cosine = (trace - 1.0)/2.0
    return math.acos(max(-1.0, min(1.0, cosine)))


def choose(hypotheses):
    """Independently replay the registered selectors; return original indices."""
    require(type(hypotheses) in (list, tuple) and len(hypotheses) == 3,
            'Exactly three ordered hypotheses')
    for hypothesis in hypotheses:
        require(type(hypothesis) is dict and set(hypothesis) ==
                {'success', 'R', 't', 'E', 'gc_inliers', 'source'}, 'Exact hypothesis fields')
        require(type(hypothesis['success']) is bool, 'Actual success Boolean')
        count = hypothesis['gc_inliers']
        require(type(count) is int and count >= 0, 'Original GC count')
        essential = hypothesis['E']
        require(type(essential) in (list, tuple) and len(essential) <= 90 and
                len(essential) % 9 == 0 and all(finite_scalar(x) for x in essential),
                'Finite whole essential stack')
        require(not hypothesis['success'] or len(essential) >= 9,
                'Successful estimate has its original essential stack')
        require(type(hypothesis['source']) is dict and bool(hypothesis['source']),
                'Original whole-source association')
        # Structural integrity only: source values are never selection inputs.
        try:
            json.dumps(hypothesis['source'], sort_keys=True, allow_nan=False)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError('Finite JSON source association') from error
    available = [index for index in range(3) if eligible(hypotheses[index])]
    if not available:
        return {'FIRST': None, 'MEDOID': None, 'MAX': None}
    pair_distance = {}
    for position, left in enumerate(available):
        for right in available[position+1:]:
            pair_distance[left, right] = angle_radians(
                hypotheses[left]['R'], hypotheses[right]['R'])
    totals = {}
    for index in available:
        totals[index] = math.fsum(pair_distance[min(index, other), max(index, other)]
                                  for other in available if other != index)
    # Tuple minimization is stable by the original index, including exact ties.
    middle = min(available, key=lambda index: (totals[index], index))
    largest = min(available, key=lambda index: (-hypotheses[index]['gc_inliers'], index))
    return {'FIRST': 0 if 0 in available else None, 'MEDOID': middle, 'MAX': largest}


def auc(errors, threshold):
    """Scalar empirical-recall trapezoids; threshold observations are excluded."""
    require(bool(errors) and threshold > 0, 'Nonempty AUC denominator')
    ordered = sorted(errors)
    previous_x, previous_y = 0.0, 0.0
    pieces = []
    accepted = 0
    for error in ordered:
        if error >= threshold:
            break
        ordinate = (accepted + 1)/len(ordered)
        pieces.append((error - previous_x)*(previous_y + ordinate)/2)
        previous_x, previous_y = error, ordinate
        accepted += 1
    pieces.append((threshold - previous_x)*previous_y)
    return math.fsum(pieces)*100/threshold


def describe(errors, failures, indices):
    per_draw = []
    for draw in range(DRAWS):
        column = [errors[pair][draw] for pair in indices]
        per_draw.append(dict(repeat=draw, num_pairs=len(indices),
            failed_poses=sum(failures[pair][draw] for pair in indices),
            severe_pose_errors_gt90=sum(value > 90 for value in column),
            **{'auc'+str(threshold): auc(column, threshold) for threshold in THRESHOLDS}))
    result = dict(num_pairs=len(indices), num_repeats=DRAWS,
        pose_attempts=len(indices)*DRAWS, per_draw=per_draw,
        failed_poses=sum(row['failed_poses'] for row in per_draw),
        severe_pose_errors_gt90=sum(row['severe_pose_errors_gt90'] for row in per_draw))
    for threshold in THRESHOLDS:
        name = 'auc'+str(threshold)
        mean = math.fsum(row[name] for row in per_draw)/DRAWS
        result[name+'_mean'] = mean
        result[name+'_std'] = math.sqrt(math.fsum(
            (row[name]-mean)**2 for row in per_draw)/DRAWS)
    return result


def contrast(candidate, reference):
    rows = []
    for draw in range(DRAWS):
        rows.append(dict(repeat=draw, **{'delta_auc'+str(threshold):
            candidate['per_draw'][draw]['auc'+str(threshold)] -
            reference['per_draw'][draw]['auc'+str(threshold)] for threshold in THRESHOLDS}))
    return dict(**{'delta_auc'+str(threshold): candidate['auc'+str(threshold)+'_mean'] -
        reference['auc'+str(threshold)+'_mean'] for threshold in THRESHOLDS},
        per_draw=rows,
        positive_draw_differences_auc5=sum(row['delta_auc5'] > 0 for row in rows),
        negative_draw_differences_auc5=sum(row['delta_auc5'] < 0 for row in rows),
        zero_draw_differences_auc5=sum(row['delta_auc5'] == 0 for row in rows))


def paired_table(candidate, reference, candidate_failed, reference_failed, indices, draws):
    result = dict(n=0, both_nonsevere=0, harm=0, rescue=0, both_severe=0,
                  candidate_severe=0, reference_severe=0, candidate_failed=0, reference_failed=0)
    for pair in indices:
        for draw in draws:
            left, right = candidate[pair][draw] > 90, reference[pair][draw] > 90
            cell = ('both_severe' if left and right else 'harm' if left else
                    'rescue' if right else 'both_nonsevere')
            result['n'] += 1
            result[cell] += 1
            result['candidate_severe'] += left
            result['reference_severe'] += right
            result['candidate_failed'] += candidate_failed[pair][draw]
            result['reference_failed'] += reference_failed[pair][draw]
    result['net_severe'] = result['candidate_severe'] - result['reference_severe']
    require(result['harm'] - result['rescue'] == result['net_severe'], 'Paired severe reconciliation')
    return result


def percentile(values, percent):
    ordered = sorted(values)
    require(bool(ordered), 'Nonempty percentile')
    position = (len(ordered)-1)*percent/100
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper]-ordered[lower])*(position-lower)


def calculate(values, failed, rooms):
    """All fixed population, scalar, paired-room and decision calculations."""
    require(type(values) is dict and type(failed) is dict and
            set(values) == set(failed) == set(ARMS), 'Exactly FIRST/MEDOID/MAX')
    require(type(rooms) in (tuple, list) and len(rooms) == PAIRS and
            all(type(room) is int for room in rooms) and tuple(sorted(set(rooms))) == ROOMS,
            'Complete registered 300-pair,20-room population')
    for arm in ARMS:
        require(type(values[arm]) in (list, tuple) and type(failed[arm]) in (list, tuple) and
                len(values[arm]) == len(failed[arm]) == PAIRS, 'Complete arm population')
        for errors, failures in zip(values[arm], failed[arm]):
            require(vector(errors, DRAWS) and all(0 <= error <= 180 for error in errors),
                    'Ten finite saved pose errors')
            require(type(failures) in (list, tuple) and len(failures) == DRAWS and
                    all(type(flag) is bool for flag in failures), 'Ten actual failure flags')
            require(all(not flag or error == 180 for flag, error in zip(failures, errors)),
                    'Failure rows retain180; successful180 is not inferred to fail')
    all_pairs = list(range(PAIRS))
    members = {room: [pair for pair in all_pairs if rooms[pair] == room] for room in ROOMS}
    methods = {}
    for arm in ARMS:
        methods[arm] = describe(values[arm], failed[arm], all_pairs)
        methods[arm]['per_room'] = {room: describe(values[arm], failed[arm], members[room]) for room in ROOMS}
    rng = random.Random(SEED)
    stream = hashlib.sha256()
    replicates = []
    for repeat in range(REPLICATES):
        sampled = [rng.choice(ROOMS) for _ in ROOMS]
        stream.update((json.dumps(sampled, separators=(',', ':'), ensure_ascii=True)+'\n').encode('ascii'))
        indices = [pair for room in sampled for pair in members[room]]
        means = {}
        for arm in ARMS:
            means[arm] = math.fsum(auc([values[arm][pair][draw] for pair in indices], 5)
                                   for draw in range(DRAWS))/DRAWS
        replicates.append(dict(replicate=repeat, rooms=sampled, auc5_mean=means,
            MEDOID_minus_FIRST=means['MEDOID']-means['FIRST'],
            MEDOID_minus_MAX=means['MEDOID']-means['MAX']))
    comparisons = {}
    for reference in ('FIRST', 'MAX'):
        result = contrast(methods['MEDOID'], methods[reference])
        result['candidate'], result['reference'] = 'MEDOID', reference
        deltas = [record['MEDOID_minus_'+reference] for record in replicates]
        interval = [percentile(deltas, 2.5), percentile(deltas, 97.5)]
        result['room_cluster_ci95'] = interval
        result['paired'] = dict(global_table=paired_table(values['MEDOID'], values[reference],
            failed['MEDOID'], failed[reference], all_pairs, range(DRAWS)),
            by_room={room: paired_table(values['MEDOID'], values[reference], failed['MEDOID'],
                failed[reference], members[room], range(DRAWS)) for room in ROOMS},
            by_draw={draw: paired_table(values['MEDOID'], values[reference], failed['MEDOID'],
                failed[reference], all_pairs, (draw,)) for draw in range(DRAWS)})
        result['gates'] = dict(mean_auc5_delta=(result['delta_auc5'] >= .5 if reference == 'FIRST'
                else result['delta_auc5'] > 0),
            strictly_positive_draw_deltas=result['positive_draw_differences_auc5'] >= 8,
            paired_room95_lower=interval[0] > 0,
            global_failed_poses=methods['MEDOID']['failed_poses'] <= methods[reference]['failed_poses'],
            global_pose_errors_strict_gt90=methods['MEDOID']['severe_pose_errors_gt90'] <=
                methods[reference]['severe_pose_errors_gt90'])
        result['passed'] = all(result['gates'].values())
        comparisons[reference] = result
    passed = all(result['passed'] for result in comparisons.values())
    return dict(methods=methods, comparisons=comparisons, passed=passed,
        decision='QUALIFIED_FOR_NEXT_STAGE' if passed else 'REJECT_FIXED_MEDOID',
        bootstrap=dict(seed=SEED, replicates=REPLICATES, rooms=list(ROOMS), sample_count=len(ROOMS),
            stream_sha256=stream.hexdigest(),
            stream_serialization='1000 consecutive compactASCII JSON20-int room lists, each followed by LF',
            samples=replicates), pairs=PAIRS, draws=DRAWS,
        failure_flag_semantics='True means actual failed estimate; never inferred from180',
        scope='Independent fixed-protocol audit replay; dependent cached triples, no new gate or candidate')


def compare(actual, expected, path='result'):
    """Exact structure/count/index/gate comparison; floats within1e-10 only."""
    if type(expected) is dict:
        require(type(actual) is dict and set(actual) == set(expected), 'Exact keys at '+path)
        for key in expected:
            compare(actual[key], expected[key], path+'/'+str(key))
    elif type(expected) in (list, tuple):
        require(type(actual) in (list, tuple) and len(actual) == len(expected), 'Exact length at '+path)
        for index, (left, right) in enumerate(zip(actual, expected)):
            compare(left, right, path+'/'+str(index))
    elif type(expected) is float:
        require(finite_scalar(actual) and math.isfinite(expected) and abs(actual-expected) <= AGREEMENT,
                'Scalar disagreement at '+path)
    else:
        require(type(actual) is type(expected) and actual == expected, 'Exact disagreement at '+path)
    return True
