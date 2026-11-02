"""RM035 fixed-three whole-hypothesis selectors; pure stdlib, GT-free.

FIRST preserves ordered replica0, including its failure. MEDOID minimizes the
sum of SO(3) geodesic distances over eligible estimated rotations. MAX chooses
the largest actual original-GC self-inlier count in that SAME eligible pool.
Exact score/count ties retain the first original index. Returned indices refer
to the caller's entire original R/t/E/GC-mask/source association; no component
is averaged, copied into another hypothesis, or independently selected.

Input is exactly three mappings with success(bool), R(flat9), t(flat3),
E(the whole finite flat essential stack, at most10branches), gc_inliers(int),
and source(nonempty JSON provenance dict including the caller's payload/mask
pins and pool/pair/draw/policy identity). The caller authenticates those pins
and computes gc_inliers from the ORIGINAL GC mask, not recovery masks or truth.
The selector has no file/label/scorer API and does not inspect source values.
Eligibility is exactly the old qualified scalar row-Gram/determinant1e-5 rule,
finite properR, finite nonzero t norm greater than FP64 epsilon, and success.
All-invalid returns None for every arm; the caller must retain failed180 rows.
Stable rotations need not be correct; neither consistent wrong clusters nor
translation/scene degeneracy are resolved by this known ensemble adaptation.
"""
import json
import math

ROTATION_TOLERANCE = 1e-5
TRANSLATION_EPSILON = 2.220446049250313e-16
FIELDS = {'success','R','t','E','gc_inliers','source'}


def _require(value, message):
    if not value:
        raise ValueError(message)


def _finite(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def _vector(value, size):
    return type(value) in (list, tuple) and len(value) == size and all(_finite(x) for x in value)


def _proper_rotation(R):
    if not _vector(R, 9):
        return False
    if not all(abs(sum(R[3*i+k]*R[3*j+k] for k in range(3))-float(i == j)) <= ROTATION_TOLERANCE
               for i in range(3) for j in range(3)):
        return False
    a,b,c,d,e,f,g,h,i = R
    determinant = a*(e*i-f*h)-b*(d*i-f*g)+c*(d*h-e*g)
    return math.isfinite(determinant) and abs(determinant-1.0) <= ROTATION_TOLERANCE


def eligible(hypothesis):
    """Return a Boolean using estimated success/R/t only; never a correctness claim."""
    if type(hypothesis) is not dict or hypothesis.get('success') is not True:
        return False
    R,t = hypothesis.get('R'),hypothesis.get('t')
    if not _proper_rotation(R) or not _vector(t, 3):
        return False
    norm = math.sqrt(sum(value*value for value in t))
    return math.isfinite(norm) and norm > TRANSLATION_EPSILON


def rotation_distance(left, right):
    """SO(3) geodesic in radians, with the standard acos-domain roundoff clip."""
    _require(_proper_rotation(left) and _proper_rotation(right), 'eligible proper rotations required')
    cosine = (sum(a*b for a,b in zip(left,right))-1.0)/2.0
    return math.acos(max(-1.0, min(1.0, cosine)))


def _validate(hypotheses):
    _require(type(hypotheses) in (list, tuple) and len(hypotheses) == 3,
             'exactly three ordered whole hypotheses required')
    for hypothesis in hypotheses:
        _require(type(hypothesis) is dict and set(hypothesis) == FIELDS,
                 'exact GT-free hypothesis fields required')
        _require(type(hypothesis['success']) is bool, 'plain success Boolean required')
        _require(type(hypothesis['gc_inliers']) is int and hypothesis['gc_inliers'] >= 0,
                 'actual nonnegative original-GC self-inlier count required')
        E = hypothesis['E']
        _require(type(E) in (list, tuple) and len(E) <= 90 and len(E) % 9 == 0
                 and all(_finite(x) for x in E), 'whole finite flat essential stack required')
        _require(not hypothesis['success'] or len(E) >= 9, 'successful hypothesis retains a nonempty essential stack')
        _require(type(hypothesis['source']) is dict and hypothesis['source'],
                 'whole-hypothesis provenance required')
        try:
            json.dumps(hypothesis['source'], sort_keys=True, allow_nan=False)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError('finite JSON provenance required') from error


def select(hypotheses):
    """Return FIRST/MEDOID/MAX original indices, or None for each failed arm.

    No inputs are mutated. FIRST does not fall forward when replica0 is invalid.
    MEDOID and MAX may select a later eligible replica from the same fixed pool.
    """
    _validate(hypotheses)
    valid = [index for index,hypothesis in enumerate(hypotheses) if eligible(hypothesis)]
    if not valid:
        return dict(FIRST=None, MEDOID=None, MAX=None)
    first = 0 if 0 in valid else None
    distances = {(i,j): rotation_distance(hypotheses[i]['R'], hypotheses[j]['R'])
                 for position,i in enumerate(valid) for j in valid[position+1:]}
    def score(index):
        return math.fsum(distances[(min(index,j), max(index,j))] for j in valid if j != index)
    medoid = valid[0]
    best_score = score(medoid)
    maximum = valid[0]
    for index in valid[1:]:
        candidate_score = score(index)
        if candidate_score < best_score:
            medoid, best_score = index, candidate_score
        if hypotheses[index]['gc_inliers'] > hypotheses[maximum]['gc_inliers']:
            maximum = index
    return dict(FIRST=first, MEDOID=medoid, MAX=maximum)
