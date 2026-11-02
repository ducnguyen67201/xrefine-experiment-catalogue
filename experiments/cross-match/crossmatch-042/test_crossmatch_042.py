"""RM042 synthetic affected checks; real checkpoint parity is separate."""

import copy
import math

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

import crossmatch_042 as m

UID = "synthetic-rm042-pair"
ARMS = ("context", "independent", "shuffled")


class TinyArgmax(nn.Module):
    """3x3 pixel expectation with the declared max subtraction and epsilon."""

    def forward(self, scores):
        shape = scores.shape[:-2]
        flat = scores.reshape(*shape, 9)
        e = (flat - flat.amax(-1, keepdim=True)).exp()
        p = e / (e.sum(-1, keepdim=True) + 1e-6)
        x = scores.new_tensor([0, 1, 2, 0, 1, 2, 0, 1, 2])
        y = scores.new_tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])
        return torch.stack(((p * x).sum(-1), (p * y).sum(-1)), -1)


class TinyHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_channels = 64
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(64, 1, 3, padding=1)
        with torch.no_grad():
            w = torch.arange(64 * 9, dtype=torch.float32)
            self.conv.weight.copy_(
                ((w.remainder(11) - 5) / 2000).reshape(1, 64, 3, 3)
            )
            self.conv.bias.fill_(0.02)

    def forward(self, x):
        return self.conv(self.relu(x)).tanh()


class TinyNet(nn.Module):
    """Frozen decoder fixture, never a claimed Small-checkpoint replica."""

    def __init__(self):
        super().__init__()
        self.desc_dim = 64
        self.adjust_only_second_keypoint = False
        self.normalized_coordinates = False
        self.score_map_head = TinyHead()
        self.spatial_argmax = TinyArgmax()
        self.eval()
        self.requires_grad_(False)


def _normalizer():
    return {
        "mean": np.zeros(128, dtype=np.float64),
        "std": np.ones(128, dtype=np.float64),
    }


def _reference(net, cache):
    """Independent out-of-place head decode with original 256/tail shapes."""
    results = []
    for endpoint in (1, 0):
        pieces = []
        for start in range(0, len(cache["row_ids"]), 256):
            x = torch.from_numpy(cache[f"x{endpoint}"][start:start + 256]).clone()
            conv = net.score_map_head.conv
            scores = F.conv2d(F.relu(x), conv.weight, conv.bias, padding=1).tanh()
            flat = scores.reshape(1, len(x), 9)
            e = (flat - flat.amax(-1, keepdim=True)).exp()
            p = e / (e.sum(-1, keepdim=True) + 1e-6)
            xs = p.new_tensor([0, 1, 2, 0, 1, 2, 0, 1, 2])
            ys = p.new_tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])
            position = torch.stack(
                ((p * xs).sum(-1), (p * ys).sum(-1)), -1
            )[0]
            origin = torch.from_numpy(
                cache[f"p{endpoint}"][start:start + 256]
            ).floor() + 0.5
            pieces.append(origin + 5.0 * (position - 1.0))
        results.append(torch.cat(pieces) if pieces else torch.empty(0, 2))
    return results[1], results[0]


def _cache(net, n):
    a = np.arange(n * 64 * 9, dtype=np.float32).reshape(n, 64, 3, 3)
    x0 = ((a % 23) / 11.0 - 1.0).astype(np.float32)
    x1 = (((a + 7) % 29) / 13.0 - 1.0).astype(np.float32)
    row = np.arange(n, dtype=np.float32)
    p0 = np.stack((row % 17 + 0.25, row % 13 + 0.75), -1)
    p1 = np.stack((row % 19 - 0.25, row % 11 + 0.125), -1)
    result = {
        "pair_uid": UID,
        "x0": x0,
        "x1": x1,
        "p0": p0.astype(np.float32),
        "p1": p1.astype(np.float32),
        "geometry": np.concatenate(
            (p0 / 32.0, p1 / 32.0), -1
        ).astype(np.float64),
        "row_ids": np.arange(n, dtype=np.int64),
    }
    q0, q1 = _reference(net, result)
    result["q0"] = q0.detach().numpy().copy()
    result["q1"] = q1.detach().numpy().copy()
    return _canonical_chunks(result)


def _adapter(active=False):
    torch.manual_seed(20261027)
    result = m.Adapter()
    if active:
        with torch.no_grad():
            first, last = result.layers[0], result.layers[2]
            first.weight.zero_()
            first.bias.fill_(0.7)
            first.weight[0, 0] = 0.15
            first.weight[1, 128] = 0.2
            last.weight.zero_()
            last.bias.zero_()
            last.weight[0, 0] = 0.1
            last.weight[1, 1] = 0.2
            last.weight[64, 1] = -0.3
    return result


def _pooled64(cache):
    return np.concatenate(
        (
            cache["x0"].astype(np.float64).mean((2, 3)),
            cache["x1"].astype(np.float64).mean((2, 3)),
        ),
        axis=1,
    )


def _pooled(cache):
    return torch.from_numpy(_pooled64(cache).astype(np.float32))


def _permute(cache, order):
    return {
        key: value[order].copy() if isinstance(value, np.ndarray) else value
        for key, value in cache.items()
    }


def _assert_cache_equal(actual, saved):
    assert actual.keys() == saved.keys()
    for key in actual:
        if isinstance(actual[key], np.ndarray):
            assert actual[key].dtype == saved[key].dtype
            assert actual[key].shape == saved[key].shape
            assert actual[key].tobytes() == saved[key].tobytes()
        else:
            assert actual[key] == saved[key]


def _assert_state_equal(net, saved):
    for key, value in net.state_dict().items():
        assert torch.equal(value, saved[key]), key
    assert all(p.grad is None for p in net.parameters())
    assert all(not p.requires_grad for p in net.parameters())


def test_argmax_fixture_includes_epsilon():
    scores = torch.zeros(1, 1, 3, 3, dtype=torch.float64)
    expected = 9.0 / (9.0 + 1e-6)
    actual = TinyArgmax()(scores)
    torch.testing.assert_close(
        actual, torch.full_like(actual, expected), rtol=0, atol=1e-12
    )


@pytest.mark.parametrize("n", [0, 1, 2, 9, 255, 256, 257])
def test_zero_residual_baseline_parity_same_original_call_shapes(n):
    net = TinyNet()
    cache = _cache(net, n)
    out = m.decode_pair(net, cache, _adapter(), _normalizer(), "context")
    expected0, expected1 = _reference(net, cache)
    assert torch.equal(out["q0"], expected0)
    assert torch.equal(out["q1"], expected1)
    assert out["gamma"].shape == (n, 128)
    assert torch.equal(out["gamma"], torch.zeros_like(out["gamma"]))
    assert out["q0"].shape == out["q1"].shape == (n, 2)


def test_adapter_gradient_through_frozen_inplace_decoder():
    net = TinyNet()
    cache = _cache(net, 12)
    saved_state = copy.deepcopy(net.state_dict())
    saved_cache = copy.deepcopy(cache)
    adapter = _adapter()
    opt = torch.optim.SGD(adapter.parameters(), lr=0.05)

    def objective():
        out = m.decode_pair(net, cache, adapter, _normalizer(), "context")
        return out["q0"][:, 0].sum() + 0.37 * out["q1"][:, 1].sum()

    objective().backward()
    final = adapter.layers[2]
    assert final.weight.grad is not None
    assert torch.isfinite(final.weight.grad).all()
    assert final.weight.grad.abs().sum().item() > 0
    for parameter in adapter.layers[0].parameters():
        assert parameter.grad is not None
        assert torch.equal(parameter.grad, torch.zeros_like(parameter.grad))
    opt.step()
    opt.zero_grad(set_to_none=True)
    objective().backward()
    hidden = adapter.layers[0].weight.grad
    assert hidden is not None and torch.isfinite(hidden).all()
    assert hidden.abs().sum().item() > 0
    _assert_cache_equal(cache, saved_cache)
    _assert_state_equal(net, saved_state)


def test_all_arms_preserve_cache_decoder_and_baseline_after_backward():
    net = TinyNet()
    cache = _cache(net, 17)
    saved_cache = copy.deepcopy(cache)
    saved_state = copy.deepcopy(net.state_dict())
    baseline = tuple(q.clone() for q in _reference(net, cache))
    for arm in ("shuffled", "context", "independent", "context"):
        adapter = _adapter(active=True)
        out = m.decode_pair(net, cache, adapter, _normalizer(), arm)
        assert torch.isfinite(out["gamma"]).all()
        assert out["gamma"].abs().max().item() <= 0.1
        (out["q0"].square().mean() + out["q1"].square().mean()).backward()
        _assert_cache_equal(cache, saved_cache)
        _assert_state_equal(net, saved_state)
        assert all(
            torch.equal(a, b)
            for a, b in zip(baseline, _reference(net, cache))
        )


def test_whole_pair_graph_crosses_original_256_boundary():
    net = TinyNet()
    cache = _cache(net, 258)
    z = np.full((258, 4), 0.7, dtype=np.float64)
    z[0] = 0
    z[256] = 0.001
    z[257] = 0.002
    cache["geometry"] = z
    expected = np.array([256, 257, 1, 2, 3, 4, 5, 6])
    graphs = [
        m.build_neighbors(z, cache["row_ids"], query_chunk=size)
        for size in (1, 7, 128)
    ]
    assert all(np.array_equal(graphs[0], graph) for graph in graphs)
    assert np.array_equal(graphs[0][0], expected)
    adapter = _adapter(active=True)
    out = m.decode_pair(net, cache, adapter, _normalizer(), "context")
    g = _pooled(cache)
    c0 = g[expected].to(torch.float64).mean(0).to(torch.float32)
    with torch.no_grad():
        expected_gamma = 0.1 * torch.tanh(
            adapter.layers(torch.cat((g[0], c0))[None])
        )[0]
    torch.testing.assert_close(
        out["gamma"][0], expected_gamma, rtol=2e-6, atol=2e-7
    )


def test_ties_and_storage_permutation_preserve_row_identity():
    net = TinyNet()
    cache = _cache(net, 12)
    cache["geometry"][:] = 0.25
    cache["row_ids"] = np.array(
        [100, 5, 45, 2, 19, 300, 7, 21, 1, 11, 17, 9], dtype=np.int64
    )
    graph = m.build_neighbors(cache["geometry"], cache["row_ids"], query_chunk=3)
    for i, row_id in enumerate(cache["row_ids"]):
        expected_ids = sorted(
            int(j) for j in cache["row_ids"] if j != row_id
        )[:8]
        assert cache["row_ids"][graph[i]].tolist() == expected_ids
    order = np.array([8, 3, 11, 0, 5, 2, 10, 1, 7, 4, 9, 6])
    inverse = np.argsort(order)
    other = _permute(cache, order)
    other_graph = m.build_neighbors(
        other["geometry"], other["row_ids"], query_chunk=7
    )
    assert np.array_equal(
        cache["row_ids"][graph], other["row_ids"][other_graph][inverse]
    )
    for arm in ARMS:
        a = m.decode_pair(net, cache, _adapter(active=True), _normalizer(), arm)
        b = m.decode_pair(net, other, _adapter(active=True), _normalizer(), arm)
        for key in ("q0", "q1", "gamma"):
            torch.testing.assert_close(
                a[key], b[key][inverse], rtol=2e-6, atol=2e-6
            )


def test_context_matches_scalar_fp64_mean_independent_uses_own_row():
    ids = np.array([90, 7, 32, 4, 18, 2, 61, 8, 51], dtype=np.int64)
    graph = m.build_neighbors(np.zeros((len(ids), 4), dtype=np.float64), ids)
    g = torch.arange(len(ids) * 128, dtype=torch.float32).reshape(len(ids), 128) / 17
    actual = m.context(g, graph, "context", ids, UID)
    expected = np.array(
        [
            [
                math.fsum(float(g[j, c]) for j in graph[i]) / len(graph[i])
                for c in range(128)
            ]
            for i in range(len(ids))
        ],
        dtype=np.float32,
    )
    torch.testing.assert_close(
        actual, torch.from_numpy(expected), rtol=0, atol=1e-6
    )
    assert torch.equal(m.context(g, graph, "independent", ids, UID), g)


def test_independent_ignores_other_features_candidate_responds():
    net = TinyNet()
    cache = _cache(net, 12)
    changed = copy.deepcopy(cache)
    changed["x0"][1:, 0] += 2.0
    before_i = m.decode_pair(
        net, cache, _adapter(active=True), _normalizer(), "independent"
    )
    after_i = m.decode_pair(
        net, changed, _adapter(active=True), _normalizer(), "independent"
    )
    assert torch.equal(before_i["q0"][0], after_i["q0"][0])
    assert torch.equal(before_i["q1"][0], after_i["q1"][0])
    before_c = m.decode_pair(
        net, cache, _adapter(active=True), _normalizer(), "context"
    )
    after_c = m.decode_pair(
        net, changed, _adapter(active=True), _normalizer(), "context"
    )
    assert not torch.equal(before_c["gamma"][0], after_c["gamma"][0])
    assert (
        not torch.equal(before_c["q0"][0], after_c["q0"][0])
        or not torch.equal(before_c["q1"][0], after_c["q1"][0])
    )


@pytest.mark.parametrize("n", [0, 1, 2, 3, 8, 9, 17])
def test_shuffle_other_row_provenance_degree_and_reproducibility(n):
    ids = np.arange(n, dtype=np.int64) * 17 + 3
    z = np.zeros((n, 4), dtype=np.float64)
    graph = m.build_neighbors(z, ids)
    shuffled = m.shuffled_neighbors(graph, ids, UID, seed=20261027)
    again = m.shuffled_neighbors(graph, ids, UID, seed=20261027)
    assert graph.shape == shuffled.shape == (n, min(8, max(0, n - 1)))
    assert np.array_equal(shuffled, again)
    for i in range(n):
        assert i not in graph[i]
        assert i not in shuffled[i]
        assert len(set(shuffled[i].tolist())) == len(shuffled[i])
        if n > 2:
            assert np.all(shuffled[i] != graph[i])
        else:
            assert np.array_equal(shuffled[i], graph[i])
    if n:
        order = np.arange(n - 1, -1, -1)
        other_ids = ids[order]
        other_graph = m.build_neighbors(z[order], other_ids)
        other_shuffled = m.shuffled_neighbors(
            other_graph, other_ids, UID, seed=20261027
        )
        assert np.array_equal(
            ids[shuffled], other_ids[other_shuffled][order]
        )


def test_shuffle_is_nonnull_when_eight_neighbors_are_a_proper_subset():
    n = 17
    ids = np.arange(n, dtype=np.int64)
    graph = m.build_neighbors(np.zeros((n, 4), dtype=np.float64), ids)
    values = torch.tensor([2.0 ** i for i in range(n)], dtype=torch.float32)
    g = values[:, None].expand(n, 128).clone()
    c = m.context(g, graph, "context", ids, UID)
    s = m.context(g, graph, "shuffled", ids, UID)
    assert torch.all(c[:, 0] != s[:, 0])


@pytest.mark.parametrize("n", [0, 1, 2, 3, 8, 9])
def test_shuffle_full_other_set_has_same_mean_when_pair_is_small(n):
    ids = np.arange(n, dtype=np.int64)
    graph = m.build_neighbors(np.zeros((n, 4), dtype=np.float64), ids)
    values = torch.tensor([2.0 ** i for i in range(n)], dtype=torch.float32)
    g = values[:, None].expand(n, 128).clone()
    c = m.context(g, graph, "context", ids, UID)
    s = m.context(g, graph, "shuffled", ids, UID)
    assert torch.equal(c, s)


def test_fit_normalizer_uses_population_moments_and_fp64_standardization():
    net = TinyNet()
    caches = [_cache(net, 2), _cache(net, 2)]
    caches[0]["pair_uid"] = "synthetic-fit-a"
    caches[1]["pair_uid"] = "synthetic-fit-b"
    expected_keys = {
        "pair_uid", "x0", "x1", "p0", "p1",
        "q0", "q1", "geometry", "row_ids", "chunks",
    }
    for cache, values in zip(caches, ((1.0, 3.0), (5.0, 7.0))):
        assert type(cache) is dict and set(cache) == expected_keys
        for endpoint in (0, 1):
            cache[f"x{endpoint}"][:] = 2.0
            cache[f"x{endpoint}"][:, 0] = np.asarray(
                values, dtype=np.float32
            )[:, None, None]
        q0, q1 = _reference(net, cache)
        cache["q0"] = q0.detach().numpy().copy()
        cache["q1"] = q1.detach().numpy().copy()
    saved = copy.deepcopy(caches)
    normalizer = m.fit_normalizer(caches)
    expected_mean = np.full(128, 2.0, dtype=np.float64)
    expected_std = np.full(128, 1e-6, dtype=np.float64)
    expected_mean[[0, 64]] = 4.0
    expected_std[[0, 64]] = math.sqrt(5.0)
    assert normalizer["mean"].dtype == np.float64
    assert normalizer["std"].dtype == np.float64
    assert not normalizer["mean"].flags.writeable
    assert not normalizer["std"].flags.writeable
    np.testing.assert_array_equal(normalizer["mean"], expected_mean)
    np.testing.assert_array_equal(normalizer["std"], expected_std)
    for cache in caches:
        adapter = _adapter(active=True)
        captured = []

        def capture_input(module, args):
            captured.append(args[0].detach().clone())

        hook = adapter.layers[0].register_forward_pre_hook(capture_input)
        try:
            m.decode_pair(net, cache, adapter, normalizer, "independent")
        finally:
            hook.remove()
        expected_z = torch.from_numpy(
            ((_pooled64(cache) - expected_mean) / expected_std).astype(np.float32)
        )
        actual_input = torch.cat(captured, dim=0)
        assert actual_input.shape == (2, 256)
        assert torch.equal(actual_input[:, :128], expected_z)
        assert torch.equal(actual_input[:, 128:], expected_z)
    for actual, original in zip(caches, saved):
        _assert_cache_equal(actual, original)


@pytest.mark.parametrize("n", [0, 1, 2, 8, 9, 257])
def test_all_rows_and_both_endpoints_remain_finite(n):
    net = TinyNet()
    cache = _cache(net, n)
    for arm in ARMS:
        out = m.decode_pair(
            net, cache, _adapter(active=True), _normalizer(), arm
        )
        assert out["q0"].shape == out["q1"].shape == (n, 2)
        assert out["gamma"].shape == (n, 128)
        assert torch.isfinite(out["q0"]).all()
        assert torch.isfinite(out["q1"]).all()
        assert torch.isfinite(out["gamma"]).all()
    if n == 1:
        graph = m.build_neighbors(cache["geometry"], cache["row_ids"])
        g = _pooled(cache)
        assert torch.equal(
            m.context(g, graph, "context", cache["row_ids"], UID),
            torch.zeros_like(g),
        )


def test_invalid_graph_inputs_are_rejected():
    z = np.zeros((3, 4), dtype=np.float64)
    with pytest.raises((ValueError, RuntimeError)):
        m.build_neighbors(z, np.array([1, 1, 2], dtype=np.int64))
    bad = z.copy()
    bad[1, 2] = np.nan
    with pytest.raises((ValueError, RuntimeError)):
        m.build_neighbors(bad, np.arange(3, dtype=np.int64))


def _loss_inputs():
    k0 = np.array(
        [[200, 0, 40], [0, 300, 50], [0, 0, 1]], dtype=np.float64
    )
    k1 = np.array(
        [[600, 0, 70], [0, 150, 20], [0, 0, 1]], dtype=np.float64
    )
    q0 = torch.tensor(
        [[42.0, 80.0], [46.0, 110.0]], requires_grad=True
    )
    q1 = torch.tensor(
        [[76.0, 35.0], [88.0, 50.0]], requires_grad=True
    )
    r = np.eye(3, dtype=np.float64)
    t = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    gamma = torch.zeros(2, 128, requires_grad=True)
    return q0, q1, k0, k1, r, t, gamma


def test_calibrated_epipolar_zero_uses_each_endpoint_intrinsics():
    q0, q1, k0, k1, r, t, gamma = _loss_inputs()
    loss = m.calibrated_pixel_sampson(
        q0, q1, k0, k1, r, t, q0.detach(), q1.detach(), gamma
    )
    assert loss is not None and loss.ndim == 0 and torch.isfinite(loss)
    assert abs(loss.item()) < 1e-9
    scale = np.diag([2.0, 2.0, 1.0])
    loss_scaled = m.calibrated_pixel_sampson(
        2 * q0, 2 * q1, scale @ k0, scale @ k1, r, t,
        2 * q0.detach(), 2 * q1.detach(), gamma,
    )
    assert loss_scaled is not None and torch.isfinite(loss_scaled)
    assert abs(loss_scaled.item()) < 1e-9


def test_coordinate_loss_is_differentiable_and_finite():
    q0, q1, k0, k1, r, t, gamma = _loss_inputs()
    bad_q1 = (
        q1.detach() + q1.new_tensor([0.0, 3.0])
    ).requires_grad_(True)
    loss = m.calibrated_pixel_sampson(
        q0, bad_q1, k0, k1, r, t,
        q0.detach(), bad_q1.detach(), gamma,
    )
    assert loss is not None and loss.ndim == 0 and torch.isfinite(loss)
    assert loss.item() > 0
    loss.backward()
    assert q0.grad is not None and torch.isfinite(q0.grad).all()
    assert bad_q1.grad is not None and torch.isfinite(bad_q1.grad).all()
    assert bad_q1.grad[:, 1].abs().sum().item() > 0


def test_zero_translation_and_empty_pairs_are_invalid():
    q0, q1, k0, k1, r, t, gamma = _loss_inputs()
    assert m.calibrated_pixel_sampson(
        q0, q1, k0, k1, r, np.zeros(3),
        q0.detach(), q1.detach(), gamma,
    ) is None
    assert m.calibrated_pixel_sampson(
        q0[:0], q1[:0], k0, k1, r, t,
        q0[:0].detach(), q1[:0].detach(), gamma[:0],
    ) is None


def test_actual_weak_objective_backpropagates_through_decode_to_adapter():
    net = TinyNet()
    cache = _cache(net, 12)
    saved_cache = copy.deepcopy(cache)
    saved_state = copy.deepcopy(net.state_dict())
    adapter = _adapter()
    out = m.decode_pair(net, cache, adapter, _normalizer(), "context")
    qbase0 = torch.from_numpy(cache["q0"]).clone()
    qbase1 = torch.from_numpy(cache["q1"]).clone()
    assert torch.equal(out["q0"], qbase0)
    assert torch.equal(out["q1"], qbase1)
    assert torch.equal(out["gamma"], torch.zeros_like(out["gamma"]))
    k0 = np.array(
        [[100, 0, 0], [0, 100, 0], [0, 0, 1]], dtype=np.float64
    )
    k1 = np.array(
        [[130, 0, 0], [0, 150, 0], [0, 0, 1]], dtype=np.float64
    )
    loss = m.calibrated_pixel_sampson(
        out["q0"], out["q1"], k0, k1,
        np.eye(3, dtype=np.float64),
        np.array([1.0, 0.0, 0.0], dtype=np.float64),
        qbase0, qbase1, out["gamma"],
    )
    assert loss is not None and loss.ndim == 0
    assert torch.isfinite(loss) and loss.item() > 0
    loss.backward()
    grad = adapter.layers[2].weight.grad
    assert grad is not None and torch.isfinite(grad).all()
    assert grad.abs().sum().item() > 0
    _assert_cache_equal(cache, saved_cache)
    _assert_state_equal(net, saved_state)


def _canonical_chunks(cache):
    cache["chunks"] = []
    for start in range(0, len(cache["row_ids"]), 256):
        stop = min(start + 256, len(cache["row_ids"]))
        layout = {
            "shape": [stop - start, 64, 3, 3],
            "stride": [576, 9, 3, 1],
            "storage_offset": 0,
            "contiguous": True,
            "channels_last": False,
        }
        cache["chunks"].append({
            "start": start, "stop": stop,
            "endpoint0": copy.deepcopy(layout),
            "endpoint1": copy.deepcopy(layout),
        })
    return cache


def _tensor_layout(x):
    return {
        "shape": list(x.shape), "stride": list(x.stride()),
        "storage_offset": x.storage_offset(),
        "contiguous": x.is_contiguous(),
        "channels_last": x.is_contiguous(memory_format=torch.channels_last),
    }


def _expected_head_layouts(cache):
    return [
        chunk[key] for chunk in cache["chunks"]
        for key in ("endpoint1", "endpoint0")
    ]


class _CaptureStopped(RuntimeError):
    pass


def _collector_fixture(n, *, fail_after_first_head=False):
    """Strict-name frozen interface fixture, explicitly not a checkpoint."""
    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.variant = "small"
            self.patch_size_reduction = 8

    class SpatialSoftArgmax2d(TinyArgmax):
        def __init__(self):
            super().__init__()
            self.temperature = 1.0
            self.eps = 1e-6

    class ScoreMapHead(TinyHead):
        def __init__(self):
            super().__init__()
            self.seen_layouts = []

        def forward(self, x):
            self.seen_layouts.append(_tensor_layout(x))
            return super().forward(x)

    class AttentionBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.positional_encoding_type = "learnable"
            self.skip_connection = True
            self.apply_layer_norm = False
            self.learned_positional_encoding_query = nn.Parameter(
                torch.zeros(1, 9, 64))
            self.learned_positional_encoding_kv = nn.Parameter(
                torch.zeros(1, 9, 64))
            self.multihead_attn = nn.MultiheadAttention(
                embed_dim=64, num_heads=4, batch_first=True)

    class SimpleJointAttnTuner(nn.Module):
        def __init__(self):
            super().__init__()
            self.desc_dim = 64
            self.color_normalization_strategy = "orig"
            self.spatial_argmax_type = "soft"
            self.attn_with_patch = True
            self.num_attention_blocks = 1
            self.adjust_only_second_keypoint = False
            self.image_values_are_normalized = True
            self.encoder = Encoder()
            self.cross_attn_patch_blocks = nn.ModuleList([AttentionBlock()])
            self.score_map_head = ScoreMapHead()
            self.spatial_argmax = SpatialSoftArgmax2d()

    class XRefine(nn.Module):
        def __init__(self):
            super().__init__()
            self.patch_radius = 5
            self.adjust_only_second_keypoint = False
            self.net = SimpleJointAttnTuner()
            source = _cache(TinyNet(), n)
            self.source0, self.source1 = source["x0"].copy(), source["x1"].copy()
            self.p0, self.p1 = source["p0"].copy(), source["p1"].copy()
            self.cursor, self.calls, self.views, self.results = 0, [], [], []
            self.fail_after_first_head = fail_after_first_head
            self.eval()
            self.requires_grad_(False)

        @staticmethod
        def fresh_view(source):
            values = torch.from_numpy(source.copy())
            count = len(values)
            flat = torch.empty(7 + values.numel(), dtype=torch.float32)
            view = torch.as_strided(
                flat, (count, 64, 3, 3),
                (64, 1, 3 * count * 64, count * 64), storage_offset=7)
            view.copy_(values)
            return view

        def forward(self, p0, p1, image0, image1):
            start, stop = self.cursor, self.cursor + len(p0)
            self.cursor = stop
            self.calls.append({
                "start": start, "stop": stop,
                "grad_enabled": torch.is_grad_enabled(),
                "p0": p0.detach().clone(), "p1": p1.detach().clone(),
                "p0_copied": not np.shares_memory(p0.numpy(), self.p0),
                "p1_copied": not np.shares_memory(p1.numpy(), self.p1),
            })
            assert not torch.is_grad_enabled()
            assert image0.shape == (3, 32, 40)
            assert image1.shape == (3, 36, 44)
            assert p0.dtype == p1.dtype == torch.float32
            x1 = self.fresh_view(self.source1[start:stop])
            self.views.append(x1)
            scores1 = self.net.score_map_head(x1)
            if self.fail_after_first_head:
                raise _CaptureStopped("controlled failure after first head")
            x0 = self.fresh_view(self.source0[start:stop])
            self.views.append(x0)
            scores0 = self.net.score_map_head(x0)

            def finish(scores, points):
                position = self.net.spatial_argmax(
                    scores.reshape(1, len(points), 3, 3))[0]
                return points.floor() + 0.5 + 5.0 * (position - 1.0)

            q0, q1 = finish(scores0, p0), finish(scores1, p1)
            self.results.append((q0.detach().clone(), q1.detach().clone()))
            return q0, q1

    return (
        XRefine(),
        torch.full((3, 32, 40), 0.2, dtype=torch.float32),
        torch.full((3, 36, 44), 0.3, dtype=torch.float32),
    )


@pytest.mark.parametrize("n", [1, 258])
def test_collect_pair_pre_relu_capture_order_layout_and_cleanup(n):
    model, image0, image1 = _collector_fixture(n)
    saved_state = copy.deepcopy(model.state_dict())
    saved_p0, saved_p1 = model.p0.copy(), model.p1.copy()
    saved_image0, saved_image1 = image0.clone(), image1.clone()
    ids = np.arange(n, dtype=np.int64)
    head = model.net.score_map_head
    assert not head._forward_pre_hooks and not head._forward_hooks
    cache = m.collect_pair(
        model, model.p0, model.p1, image0, image1, ids, UID)
    assert type(cache) is dict and set(cache) == {
        "pair_uid", "x0", "x1", "p0", "p1", "q0", "q1",
        "geometry", "row_ids", "chunks",
    }
    assert [(c["start"], c["stop"]) for c in model.calls] == [
        (start, min(start + 256, n)) for start in range(0, n, 256)]
    assert all(not c["grad_enabled"] for c in model.calls)
    assert all(c["p0_copied"] and c["p1_copied"] for c in model.calls)
    for call in model.calls:
        a, b = call["start"], call["stop"]
        assert torch.equal(call["p0"], torch.from_numpy(saved_p0[a:b]))
        assert torch.equal(call["p1"], torch.from_numpy(saved_p1[a:b]))
    assert all(torch.all(view >= 0) for view in model.views)
    assert np.any(cache["x0"] < 0) and np.any(cache["x1"] < 0)
    np.testing.assert_array_equal(cache["x0"], model.source0)
    np.testing.assert_array_equal(cache["x1"], model.source1)
    np.testing.assert_array_equal(cache["p0"], saved_p0)
    np.testing.assert_array_equal(cache["p1"], saved_p1)
    np.testing.assert_array_equal(cache["row_ids"], ids)
    assert cache["pair_uid"] == UID
    assert head.seen_layouts == _expected_head_layouts(cache)
    assert all(layout["storage_offset"] == 7 for layout in head.seen_layouts)
    assert all(not layout["contiguous"] for layout in head.seen_layouts)
    assert all(layout["channels_last"] == (n == 1)
               for layout in head.seen_layouts)
    expected_geometry = np.column_stack((
        saved_p0[:, 0].astype(np.float64) / 40,
        saved_p0[:, 1].astype(np.float64) / 32,
        saved_p1[:, 0].astype(np.float64) / 44,
        saved_p1[:, 1].astype(np.float64) / 36,
    ))
    np.testing.assert_array_equal(cache["geometry"], expected_geometry)
    assert cache["geometry"].dtype == np.float64
    assert cache["geometry"].flags.c_contiguous
    assert np.any(cache["geometry"] < 0)
    q0 = torch.cat([result[0] for result in model.results]).numpy()
    q1 = torch.cat([result[1] for result in model.results]).numpy()
    np.testing.assert_array_equal(cache["q0"], q0)
    np.testing.assert_array_equal(cache["q1"], q1)
    assert not head._forward_pre_hooks and not head._forward_hooks
    assert torch.equal(image0, saved_image0)
    assert torch.equal(image1, saved_image1)
    _assert_state_equal(model, saved_state)

    saved_cache = copy.deepcopy(cache)
    head.seen_layouts.clear()
    out = m.decode_pair(
        model.net, cache, _adapter(), _normalizer(), "context")
    assert head.seen_layouts == _expected_head_layouts(cache)
    assert torch.equal(out["q0"], torch.from_numpy(cache["q0"]))
    assert torch.equal(out["q1"], torch.from_numpy(cache["q1"]))
    _assert_cache_equal(cache, saved_cache)
    _assert_state_equal(model, saved_state)


def test_collect_pair_removes_hook_after_controlled_exception():
    model, image0, image1 = _collector_fixture(9, fail_after_first_head=True)
    head = model.net.score_map_head
    saved_state = copy.deepcopy(model.state_dict())
    assert not head._forward_pre_hooks and not head._forward_hooks
    with pytest.raises(_CaptureStopped, match="after first head"):
        m.collect_pair(
            model, model.p0, model.p1, image0, image1,
            np.arange(9, dtype=np.int64), UID)
    assert len(head.seen_layouts) == len(model.views) == len(model.calls) == 1
    assert not head._forward_pre_hooks and not head._forward_hooks
    _assert_state_equal(model, saved_state)


def test_empty_collector_calls_no_model_and_cleans_hook():
    model, image0, image1 = _collector_fixture(0)
    head = model.net.score_map_head
    cache = m.collect_pair(
        model, model.p0, model.p1, image0, image1,
        np.empty(0, dtype=np.int64), UID)
    assert model.calls == model.views == head.seen_layouts == []
    assert cache["chunks"] == []
    assert cache["x0"].shape == cache["x1"].shape == (0, 64, 3, 3)
    assert cache["q0"].shape == cache["q1"].shape == (0, 2)
    assert not head._forward_pre_hooks and not head._forward_hooks


def test_noncontiguous_restore_keeps_actual_weak_loss_gradient():
    model, image0, image1 = _collector_fixture(9)
    cache = m.collect_pair(
        model, model.p0, model.p1, image0, image1,
        np.arange(9, dtype=np.int64), UID)
    saved_cache = copy.deepcopy(cache)
    saved_state = copy.deepcopy(model.state_dict())
    head = model.net.score_map_head
    head.seen_layouts.clear()
    adapter = _adapter()
    out = m.decode_pair(
        model.net, cache, adapter, _normalizer(), "context")
    assert head.seen_layouts == _expected_head_layouts(cache)
    assert torch.equal(out["q0"], torch.from_numpy(cache["q0"]))
    assert torch.equal(out["q1"], torch.from_numpy(cache["q1"]))
    loss = m.calibrated_pixel_sampson(
        out["q0"], out["q1"],
        np.diag([100.0, 100.0, 1.0]), np.diag([130.0, 150.0, 1.0]),
        np.eye(3, dtype=np.float64),
        np.array([1.0, 0.0, 0.0], dtype=np.float64),
        torch.from_numpy(cache["q0"]), torch.from_numpy(cache["q1"]),
        out["gamma"])
    assert loss is not None and torch.isfinite(loss) and loss.item() > 0
    loss.backward()
    grad = adapter.layers[2].weight.grad
    assert grad is not None and torch.isfinite(grad).all()
    assert grad.abs().sum().item() > 0
    _assert_cache_equal(cache, saved_cache)
    _assert_state_equal(model, saved_state)


def _training_cases(net, *, one_valid):
    caches, targets = [], {}
    for i in range(150):
        cache = _cache(net, 3 if one_valid and i == 0 else 2)
        cache["pair_uid"] = f"synthetic-fit-{i:03d}"
        if i:
            cache["x0"][:, 0] = 10.0
            cache["x1"][:, 0] = -6.0
            q0, q1 = _reference(net, cache)
            cache["q0"] = q0.detach().numpy().copy()
            cache["q1"] = q1.detach().numpy().copy()
        caches.append(cache)
        targets[cache["pair_uid"]] = {
            "K0": np.diag([100.0, 100.0, 1.0]),
            "K1": np.diag([130.0, 150.0, 1.0]),
            "R": np.eye(3, dtype=np.float64),
            "t": np.array(
                [1.0 if one_valid and i == 0 else 0.0, 0.0, 0.0],
                dtype=np.float64),
        }
    return caches, targets


def _seeded_initial_state():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20261027)
        return copy.deepcopy(m.Adapter().state_dict())


def _assert_training_common(result, caches, skipped):
    metadata = result["metadata"]
    assert metadata["status"] == "LAST_EPOCH_STATES_RETURNED"
    assert metadata["epochs"] == 20
    assert metadata["checkpoint_selection"] == "LAST_ONLY"
    assert metadata["identical_initialization"] is True
    assert metadata["separate_optimizers"] is True
    assert metadata["final_heads_frozen"] is True
    assert metadata["fit_pairs"] == 150
    assert metadata["fit_rows"] == sum(len(c["row_ids"]) for c in caches)
    assert metadata["graph_builds"] == 150
    assert metadata["empty_pair_uids"] == []
    assert metadata["skipped_unique_pairs"] == skipped
    assert len(metadata["epochs_actual"]) == 20
    uids = [cache["pair_uid"] for cache in caches]
    for epoch, row in enumerate(metadata["epochs_actual"]):
        expected_order = [
            uids[int(i)]
            for i in np.random.default_rng(20261027 + epoch).permutation(150)]
        assert row["epoch"] == epoch
        assert row["pair_order_uids"] == expected_order
        assert row["skipped_pairs"] == skipped
        for arm in ARMS:
            assert row["arms"][arm]["updates"] == 150 - skipped
            assert math.isfinite(row["arms"][arm]["loss_sum"])

    pooled = np.concatenate([_pooled64(cache) for cache in caches], axis=0)
    expected_mean = pooled.mean(axis=0, dtype=np.float64)
    expected_std = np.maximum(
        np.sqrt(((pooled - expected_mean) ** 2).mean(axis=0)), 1e-6)
    np.testing.assert_allclose(
        result["normalizer"]["mean"], expected_mean, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        result["normalizer"]["std"], expected_std, rtol=1e-12, atol=1e-12)
    assert set(result["heads"]) == set(result["optimizer_states"]) == set(ARMS)
    parameter_ids = []
    for arm in ARMS:
        head = result["heads"][arm]
        assert not head.training
        assert all(not p.requires_grad and p.grad is None for p in head.parameters())
        parameter_ids.append({id(p) for p in head.parameters()})
        groups = result["optimizer_states"][arm]["param_groups"]
        assert len(groups) == 1
        assert groups[0]["lr"] == 0.001
        assert groups[0]["betas"] == (0.9, 0.999)
        assert groups[0]["eps"] == 1e-8
        assert groups[0]["weight_decay"] == 1e-4
    for i in range(len(parameter_ids)):
        for j in range(i):
            assert parameter_ids[i].isdisjoint(parameter_ids[j])


def test_real_trainer_twenty_valid_steps_not_three_thousand():
    net = TinyNet()
    caches, targets = _training_cases(net, one_valid=True)
    saved_caches = copy.deepcopy(caches)
    saved_net = copy.deepcopy(net.state_dict())
    initial = _seeded_initial_state()
    result = m.train_heads(net, caches, targets)
    _assert_training_common(result, caches, skipped=149)
    assert set(result["metadata"]["zero_translation_pair_uids"]) == {
        cache["pair_uid"] for cache in caches[1:]}
    for arm in ARMS:
        assert result["metadata"]["updates"][arm] == 20
        state = result["optimizer_states"][arm]["state"]
        assert len(state) == len(list(result["heads"][arm].parameters()))
        for parameter_state in state.values():
            step = parameter_state["step"]
            assert float(step.item() if torch.is_tensor(step) else step) == 20.0
        final = result["heads"][arm].state_dict()
        assert any(not torch.equal(final[key], initial[key]) for key in initial)
    for cache, saved in zip(caches, saved_caches):
        _assert_cache_equal(cache, saved)
    _assert_state_equal(net, saved_net)


def test_real_trainer_all_invalid_no_adam_clock_or_weight_decay():
    net = TinyNet()
    caches, targets = _training_cases(net, one_valid=False)
    saved_caches = copy.deepcopy(caches)
    saved_net = copy.deepcopy(net.state_dict())
    initial = _seeded_initial_state()
    result = m.train_heads(net, caches, targets)
    _assert_training_common(result, caches, skipped=150)
    assert set(result["metadata"]["zero_translation_pair_uids"]) == {
        cache["pair_uid"] for cache in caches}
    for arm in ARMS:
        assert result["metadata"]["updates"][arm] == 0
        assert result["optimizer_states"][arm]["state"] == {}
        final = result["heads"][arm].state_dict()
        assert final.keys() == initial.keys()
        for key in initial:
            assert torch.equal(final[key], initial[key]), (arm, key)
    for cache, saved in zip(caches, saved_caches):
        _assert_cache_equal(cache, saved)
    _assert_state_equal(net, saved_net)


def test_sampson_and_regularizers_match_independent_scalar_reference():
    q0 = torch.tensor([[2.0, 4.0], [3.0, 7.0]], dtype=torch.float32)
    q1 = torch.tensor([[5.0, 1.0], [9.0, 2.0]], dtype=torch.float32)
    d0 = torch.tensor([[0.5, -0.25], [1.0, -0.5]], dtype=torch.float32)
    d1 = torch.tensor([[-0.75, 0.25], [0.125, -1.25]], dtype=torch.float32)
    base0, base1 = q0 - d0, q1 - d1
    gamma = torch.empty(2, 128, dtype=torch.float32)
    gamma[0].fill_(0.03125)
    gamma[1].fill_(-0.0625)
    k0 = np.array(
        [[120, 0, 1], [0, 100, 2], [0, 0, 1]], dtype=np.float64)
    k1 = np.array(
        [[180, 0, -2], [0, 150, -3], [0, 0, 1]], dtype=np.float64)
    denominator = 1.0 / 100.0 ** 2 + 1.0 / 150.0 ** 2
    terms = []
    for i in range(2):
        residual = (
            (float(q0[i, 1]) - 2.0) / 100.0
            - (float(q1[i, 1]) + 3.0) / 150.0)
        terms.append(math.log1p(residual ** 2 / denominator))
    epipolar = math.fsum(terms) / 2.0
    anchor = 0.01 * math.fsum(
        float(value) ** 2
        for value in torch.cat((d0, d1), dim=1).reshape(-1)) / 8.0
    gamma_penalty = 0.001 * math.fsum(
        float(value) ** 2 for value in gamma.reshape(-1)) / 256.0
    r, translation = np.eye(3, dtype=np.float64), np.array([7.0, 0.0, 0.0])
    actual = m.calibrated_pixel_sampson(
        q0, q1, k0, k1, r, translation, base0, base1, gamma)
    epi_only = m.calibrated_pixel_sampson(
        q0, q1, k0, k1, r, translation, q0, q1, torch.zeros_like(gamma))
    assert actual is not None and epi_only is not None
    assert actual.item() == pytest.approx(
        epipolar + anchor + gamma_penalty, rel=1e-10, abs=1e-10)
    assert epi_only.item() == pytest.approx(epipolar, rel=1e-10, abs=1e-10)
    assert (actual - epi_only).item() == pytest.approx(
        anchor + gamma_penalty, rel=1e-10, abs=1e-10)


def test_overlapping_layout_is_rejected_before_decoder_head():
    net = TinyNet()
    cache = _cache(net, 3)
    cache["chunks"][0]["endpoint0"]["stride"] = [1, 1, 1, 1]
    with pytest.raises(ValueError, match="dense nonoverlapping"):
        m.decode_pair(net, cache, _adapter(), _normalizer(), "context")
