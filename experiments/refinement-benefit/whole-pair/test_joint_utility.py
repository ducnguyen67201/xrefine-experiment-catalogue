"""Fabricated fixtures only: no image/cache/GT/native/model-forward access."""
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from joint_utility_core import (FEATURE_NAMES, SCENES, UtilityAdapter, canonical_names, compute_features,
    fit_cross_scene, fit_ridge, matched_shuffle, mean_confidence_feature, predict_ridge, shuffle_key)


def affine_pair():
    a = np.asarray([[0, 0], [0, 2], [2, 0], [2, 2]] * 2, dtype=np.float32)
    b = a * 2 + np.asarray([3, -7], dtype=np.float32)
    descriptors = np.zeros((8, 64), dtype=np.float32)
    descriptors[:, 0] = 1
    return (a, b), (descriptors, descriptors.copy()), (np.eye(3), np.eye(3))


def cross_scene_fixture():
    records = [dict(pair_index=p, scene=SCENES[p // 8],
        image_names=[SCENES[p // 8] + f"/images/{p}-a.jpg", SCENES[p // 8] + f"/images/{p}-b.jpg"]) for p in range(16)]
    x = np.asarray([[np.sin((p+1)*(j+1)/7) + p/10 for j in range(12)] for p in range(16)], dtype=np.float64)
    projections = {scene:dict(eval_scene=scene,train_scene=SCENES[1-i],
        records=[dict(pair_index=p,target=(p-8)/20) for p in range(16) if records[p]["scene"] != scene]) for i,scene in enumerate(SCENES)}
    return x, records, projections


def constant_checkpoint(intercept):
    def model(indices):
        n = len(indices)
        return dict(feature_indices=indices, training_mean=[0.0]*n, training_scale=[1.0]*n,
                    clipped_training_mean=[0.0]*n, coefficients=[0.0]*n, intercept=intercept)
    return dict(schema="joint-utility-checkpoint-v1", feature_names=list(FEATURE_NAMES), models={
        scene:{"utility_gate":model(list(range(12))), "confidence_gate":model([4])} for scene in SCENES})


def test_exact_affine_and_geometric_summary():
    raw, descriptors, cameras = affine_pair()
    features = compute_features(raw, descriptors, cameras)
    assert features.shape == (12,) and features.dtype == np.float64
    assert features[0] == pytest.approx(np.log(9))
    assert features[1:5] == pytest.approx([1, 1, 1, 1])
    assert features[5] == 0
    assert features[6:8] == pytest.approx([.25, .25])
    assert features[8:10] == pytest.approx([1, 1])
    assert features[10:] == pytest.approx([0, 0], abs=1e-12)


def test_cosine_linear_percentiles_and_population_sd():
    a = np.asarray([[0, 0], [1, 0], [2, 0]], dtype=np.float64)
    d0 = np.asarray([[1, 0]] * 3, dtype=np.float64)
    d1 = np.asarray([[-1, 0], [0, 1], [1, 0]], dtype=np.float64)
    x = compute_features((a,a), (d0,d1), (np.eye(3),np.eye(3)))
    assert x[1:6] == pytest.approx([-.8, 0, .8, 0, np.sqrt(2/3)])
    assert x[10:].tolist() == [0,0]


@pytest.mark.parametrize("count", [0, 1, 7, 8])
def test_empty_constant_and_rank_deficient_inputs_stay_finite(count):
    points = np.zeros((count,2), dtype=np.float64)
    descriptors = np.zeros((count,64), dtype=np.float64)
    features = compute_features((points,points), (descriptors,descriptors), (np.eye(3),np.eye(3)))
    assert np.isfinite(features).all()
    assert features[1:6].tolist() == [0]*5
    assert features[8:].tolist() == [0]*4
    assert features[6] == (0 if count == 0 else 1/16)


def test_non_affine_correspondences_produce_nonzero_residual_features():
    raw, descriptors, cameras = affine_pair()
    changed = raw[1].copy()
    changed[-1] += [7,-3]
    features = compute_features((raw[0],changed), descriptors, cameras)
    assert features[10] > 0 and features[11] >= features[10]


def test_feature_computation_preserves_raw_arrays_and_confidence_parity():
    raw, descriptors, cameras = affine_pair()
    before = tuple(a.tobytes() for a in raw+descriptors+cameras)
    x = compute_features(raw,descriptors,cameras)
    confidence = mean_confidence_feature(raw,descriptors)
    assert confidence[4] == x[4]
    assert tuple(a.tobytes() for a in raw+descriptors+cameras) == before


def test_invalid_geometry_and_nonfinite_descriptors_are_refused():
    raw, descriptors, cameras = affine_pair()
    bad = descriptors[0].copy()
    bad[0,0] = np.nan
    with pytest.raises(ValueError):
        compute_features(raw,(bad,descriptors[1]),cameras)
    with pytest.raises(ValueError,match="Singular"):
        compute_features(raw,descriptors,(np.zeros((3,3)),cameras[1]))


def test_clipped_training_recenter_preserves_unpenalized_intercept():
    x = np.zeros((200,12),dtype=np.float64)
    x[-1,0] = 1e6
    y = np.asarray([-.25]*199 + [.75],dtype=np.float64)
    model = fit_ridge(x,y,list(range(12)))
    # Clipping this outlier makes the first standardized-column mean nonzero.
    assert abs(model["clipped_training_mean"][0]) > .01
    predictions = predict_ridge(x,model)
    assert np.mean(predictions) == pytest.approx(np.mean(y),abs=1e-12)
    baseline = np.sum((y-np.mean(y))**2)
    objective = np.sum((predictions-y)**2) + 10*np.sum(np.asarray(model["coefficients"])**2)
    assert objective < baseline


def test_constant_features_have_zero_coefficients_and_mean_prediction():
    x = np.ones((8,12),dtype=np.float64)
    y = np.asarray([-.2,.1,.3,.4,-.5,.8,.2,-.1])
    model = fit_ridge(x,y,[4])
    assert model["coefficients"] == [0.0]
    assert predict_ridge(np.zeros(12),model) == pytest.approx(np.mean(y))


def test_cross_scene_heldout_labels_and_features_cannot_change_its_model():
    x,records,projections = cross_scene_fixture()
    models,details = fit_cross_scene(x,records,projections)
    changed = json.loads(json.dumps(projections))
    # These0015 labels only train the0022-evaluation model.
    for row in changed["0022"]["records"]:
        row["target"] = .9
    changed_x = x.copy()
    changed_x[:8] += 1000
    rerun,_ = fit_cross_scene(changed_x,records,changed)
    assert rerun["0015"] == models["0015"]
    assert rerun["0022"] != models["0022"]
    assert set(details["0015"]["train_indices"]) == set(range(8,16))
    assert not set(details["0015"]["train_images"]).intersection(details["0015"]["eval_images"])
    for info in details.values():
        for proof in info["fit_proofs"].values():
            assert proof["normal_equation_maxabs"] < 1e-10
            assert proof["intercept_gradient_abs"] < 1e-10


@pytest.mark.parametrize("bad", ["heldout", "duplicate", "missing"])
def test_fold_label_projection_refuses_leakage_or_incomplete_inventory(bad):
    x,records,projections = cross_scene_fixture()
    if bad == "heldout":
        projections["0015"]["records"][0]["pair_index"] = 0
    elif bad == "duplicate":
        projections["0015"]["records"].append(projections["0015"]["records"][0].copy())
    else:
        projections["0015"]["records"].pop()
    with pytest.raises(ValueError):
        fit_cross_scene(x,records,projections)


def test_shuffle_has_exact_locked_wire_and_per_scene_allocation_counts():
    _,records,_ = cross_scene_fixture()
    expected = hashlib.sha256(b'[42,"0015/images/a.jpg","0015/images/b.jpg"]\n').hexdigest()
    assert shuffle_key(["0015/images/a.jpg","0015/images/b.jpg"]) == expected
    primary = np.asarray([False,True,False,True,True,True,True,True] + [True]*7 + [False],dtype=bool)
    shuffled = matched_shuffle(records,primary)
    assert sum(row["raw"] for row in shuffled[:8]) == 2
    assert sum(row["raw"] for row in shuffled[8:]) == 1
    assert shuffled == matched_shuffle(records,primary)


@pytest.mark.parametrize("kind", ["utility_gate", "confidence_gate"])
@pytest.mark.parametrize("intercept", [-.5, 0, .5])
def test_adapter_action_ignores_stock_movement_and_keeps_whole_endpoint_block(tmp_path,kind,intercept):
    checkpoint = tmp_path/"checkpoint.json"
    checkpoint.write_text(json.dumps(constant_checkpoint(intercept)))
    adapter = UtilityAdapter(checkpoint,kind)
    raw,descriptors,cameras = affine_pair()
    stock = tuple(p+3 for p in raw)
    pair = SimpleNamespace(raw=raw,stock=stock,descriptors=descriptors,intrinsics=cameras,
                           image_names=("0015/images/a.jpg","0015/images/b.jpg"),pair_index=123)
    choice = adapter.choose_stock(pair)
    output = adapter.refine(pair)
    expected = stock if intercept >= 0 else raw
    assert np.array_equal(output.points0,expected[0]) and np.array_equal(output.points1,expected[1])
    pair.stock = tuple(p+10000 for p in raw)
    pair.pair_index = 987
    assert adapter.choose_stock(pair) == choice


def test_shuffled_adapter_reads_only_frozen_identity_allocation(tmp_path):
    checkpoint,shuffle = tmp_path/"checkpoint.json",tmp_path/"shuffled.json"
    checkpoint.write_text(json.dumps(constant_checkpoint(.5)))
    names = ("0015/images/a.jpg","0015/images/b.jpg")
    shuffle.write_text(json.dumps(dict(schema="joint-utility-shuffled-v1",seed=42,entries=[dict(image_names=names,raw=True)])))
    adapter = UtilityAdapter(checkpoint,"shuffled_gate",shuffle)
    pair = SimpleNamespace(image_names=names)
    assert adapter.choose_stock(pair) is False
    pair.image_names = ("0015/images/other.jpg","0015/images/b.jpg")
    with pytest.raises(ValueError,match="outside frozen panel"):
        adapter.choose_stock(pair)


def test_unknown_scene_is_refused_instead_of_claiming_universal_deployment():
    with pytest.raises(ValueError):
        canonical_names(["0099/images/a.jpg","0099/images/b.jpg"])
