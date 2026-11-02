"""Fabricated array fixtures; no real caches, images, GT or native pose work."""
import ast
import hashlib
import io
import json
import math
from pathlib import Path
import struct
import unittest

import numpy as np

import multiview_consensus_core as core


def descriptors(count, token=1):
    result = np.zeros((count, 64), dtype=np.float32)
    result[:, 0] = token
    return result


def observation(partner, index, raw, stock, desc, side=0):
    return {"partner": partner, "pair_index": index, "side": side,
            "raw": raw, "stock": stock, "descriptors": desc}


def basic_pair(count=1):
    raw = (np.asarray([[10.25 + row, 12.75] for row in range(count)], dtype=np.float32).reshape(-1, 2),
           np.asarray([[20.25 + row, 22.75] for row in range(count)], dtype=np.float32).reshape(-1, 2))
    stock = tuple(value.copy() for value in raw)
    desc = (descriptors(count), descriptors(count, 2))
    return raw, stock, desc


def contexts(raw, stock, desc, extra0=None, extra1=None):
    extra0 = stock[0] + np.float32(0.25) if extra0 is None else extra0
    extra1 = stock[1] - np.float32(0.25) if extra1 is None else extra1
    first = [observation("B", 0, raw[0], stock[0], desc[0]),
             observation("C", 1, raw[0], extra0, desc[0])]
    second = [observation("A", 0, raw[1], stock[1], desc[1], 1),
              observation("D", 2, raw[1], extra1, desc[1])]
    return (core.build_image_context("A", first), core.build_image_context("B", second))


def independent_digest(values):
    return hashlib.sha256(json.dumps(values, ensure_ascii=True,
                           separators=(",", ":")).encode("ascii")).hexdigest()


def scalar_huber(anchor, votes):
    """Scalar Python/math reference, independent of vector padding/reductions."""
    z = [float(value) for value in anchor]
    for _ in range(32):
        gradient = [z[0] - anchor[0], z[1] - anchor[1]]
        for vote in votes:
            dx, dy = z[0] - vote[0], z[1] - vote[1]
            length = math.hypot(dx, dy)
            factor = 1.0 if length <= 0.5 else 0.5 / length
            gradient[0] += factor * dx / len(votes)
            gradient[1] += factor * dy / len(votes)
        z[0] -= 0.5 * gradient[0]
        z[1] -= 0.5 * gradient[1]
    return z


class MultiViewConsensusTests(unittest.TestCase):
    def result(self, raw, stock, desc, ctx, index=0):
        return core.transform_pair(index, ("A", "B"), raw, stock, desc, ctx)

    def test_all_plugin_metadata_is_entirely_literal_for_original_plan_parser(self):
        directory = Path(__file__).resolve().parent
        common = ["multiview_consensus_core.py", "plugbench_multiview_transport.py",
                  "../../research/multiview-consensus-20261020/prepared/manifest.json",
                  "../../research/multiview-consensus-20261020/prepared/packed/bank-0000-0500.zip",
                  "../../research/multiview-consensus-20261020/prepared/packed/bank-0500-1000.zip",
                  "../../research/multiview-consensus-20261020/prepared/packed/bank-1000-1500.zip"]
        for kind in core.METHODS:
            path = directory / ("plugbench_multiview_" + kind + ".py")
            tree = ast.parse(path.read_bytes(), filename=str(path))
            assigned = [node.value for node in tree.body if isinstance(node, ast.Assign)
                        and any(isinstance(target, ast.Name) and target.id == "METADATA"
                                for target in node.targets)]
            with self.subTest(kind=kind):
                self.assertEqual(len(assigned), 1)
                metadata = ast.literal_eval(assigned[0])
                expected = common if kind == "huber" else ["plugbench_multiview_huber.py", *common]
                self.assertEqual(metadata, {"name": "multiview_" + kind,
                    "requires": ["stock", "descriptors"], "provenance_files": expected})

    def test_canonical_hash_has_no_line_feed(self):
        values = [42, "A", "0000000000000000", "C"]
        self.assertEqual(core.canonical_digest(values), independent_digest(values))
        wire = json.dumps(values, separators=(",", ":")).encode("ascii")
        self.assertNotEqual(core.canonical_digest(values), hashlib.sha256(wire + b"\n").hexdigest())

    def test_empty_context_and_pair_are_valid_noops(self):
        raw, stock, desc = basic_pair(0)
        ctx = (core.build_image_context("A", []), core.build_image_context("B", []))
        result = self.result(raw, stock, desc, ctx)
        self.assertEqual(result["eligible"].shape, (0,))
        self.assertEqual(result["support_counts"].shape, (0, 2))
        self.assertEqual(result["context_dispersion"].shape, (0, 2))
        for points in result["predictions"].values():
            self.assertEqual(points[0].shape, (0, 2))
            self.assertEqual(points[1].dtype, np.float32)
        self.assertEqual(result["stats"]["eligible_matches"], 0)

    def test_context_roundtrip_is_nonpickle_and_readonly(self):
        raw, stock, desc = basic_pair(2)
        first, _ = contexts(raw, stock, desc)
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **first.to_arrays())
        with np.load(io.BytesIO(buffer.getvalue()), allow_pickle=False) as archive:
            second = core.ImageContext.from_arrays({key: archive[key] for key in archive.files})
        self.assertEqual(second.image_name, first.image_name)
        self.assertEqual(second.statistics, first.statistics)
        self.assertEqual(second.node_keys.dtype, np.dtype("V264"))
        self.assertEqual(second.ambiguity_keys.dtype, np.dtype("V8"))
        for name in ("node_keys", "raw", "offsets", "partner_names", "partner_ids", "means"):
            np.testing.assert_array_equal(getattr(second, name), getattr(first, name))
            self.assertFalse(getattr(second, name).flags.writeable)

    def test_per_partner_mean_has_one_vote_and_all_target_repeats_are_excluded(self):
        raw, stock, desc = basic_pair()
        first = [observation("B", 0, raw[0], stock[0], desc[0]),
                 observation("B", 8, raw[0], np.full((1, 2), 1000, np.float32), desc[0], 1)]
        for index, value in enumerate((1, 3, 5), 1):
            first.append(observation("C", index, raw[0], np.full((1, 2), value, np.float32), desc[0]))
        first.append(observation("D", 4, raw[0], np.full((1, 2), 5, np.float32), desc[0]))
        ctx = (core.build_image_context("A", first), contexts(raw, stock, desc)[1])
        result = self.result(raw, stock, desc, ctx)
        np.testing.assert_array_equal(result["support_counts"], [[2, 1]])
        expected = ((stock[0].astype(np.float64) + 4) / 2).astype(np.float32)
        np.testing.assert_array_equal(result["predictions"]["mean"][0], expected)
        self.assertAlmostEqual(float(result["context_dispersion"][0, 0]), math.sqrt(2), places=14)
        self.assertEqual(ctx[0].statistics["observations"], 6)
        self.assertEqual(ctx[0].statistics["distinct_partner_votes"], 3)

    def test_partner_limit_uses_exact_raw_wire_and_hash_order(self):
        raw, stock, desc = basic_pair()
        partners = ["C" + str(index) for index in range(12)]
        first = [observation("B", 0, raw[0], stock[0], desc[0])]
        for index, partner in enumerate(partners, 1):
            first.append(observation(partner, index, raw[0], np.full((1, 2), index, np.float32), desc[0]))
        ctx = (core.build_image_context("A", first), contexts(raw, stock, desc)[1])
        raw_hex = struct.pack("<ff", float(raw[0][0, 0]), float(raw[0][0, 1])).hex()
        chosen = sorted((independent_digest([42, "A", raw_hex, partner]), partner, index)
                        for index, partner in enumerate(partners, 1))[:8]
        expected_vote = sum(index for _, _, index in chosen) / 8
        result = self.result(raw, stock, desc, ctx)
        np.testing.assert_array_equal(result["support_counts"], [[8, 1]])
        expected = ((stock[0].astype(np.float64) + expected_vote) / 2).astype(np.float32)
        np.testing.assert_array_equal(result["predictions"]["mean"][0], expected)

    def test_observation_input_order_does_not_change_context_bytes(self):
        raw, stock, desc = basic_pair(2)
        items = [observation("B", 0, raw[0], stock[0], desc[0]),
                 observation("D", 3, raw[0], stock[0] + 3, desc[0]),
                 observation("C", 2, raw[0], stock[0] + 2, desc[0]),
                 observation("C", 1, raw[0], stock[0] + 1, desc[0])]
        left = core.build_image_context("A", items).to_arrays()
        right = core.build_image_context("A", reversed(items)).to_arrays()
        for key in left:
            self.assertEqual(left[key].dtype, right[key].dtype)
            self.assertEqual(left[key].tobytes(), right[key].tobytes())

    def test_nearby_raw_coordinate_is_not_joined(self):
        raw, stock, desc = basic_pair()
        nearby = raw[0].copy()
        nearby[0, 0] = np.nextafter(nearby[0, 0], np.float32(np.inf))
        first = core.build_image_context("A", [
            observation("B", 0, raw[0], stock[0], desc[0]),
            observation("C", 1, nearby, stock[0] + 1, desc[0])])
        result = self.result(raw, stock, desc, (first, contexts(raw, stock, desc)[1]))
        self.assertFalse(result["eligible"][0])
        for points in result["predictions"].values():
            self.assertEqual(points[0].tobytes(), stock[0].tobytes())
            self.assertEqual(points[1].tobytes(), stock[1].tobytes())

    def test_descriptor_conflict_erases_prior_votes_and_disables_both_endpoints(self):
        raw, stock, desc = basic_pair()
        changed = desc[0].copy()
        changed.view(np.uint32)[0, 0] ^= np.uint32(1)
        first = core.build_image_context("A", [
            observation("B", 0, raw[0], stock[0], desc[0]),
            observation("C", 1, raw[0], stock[0] + 1, desc[0]),
            observation("D", 2, raw[0], stock[0] + 2, changed)])
        self.assertEqual(first.statistics["ambiguous_raw_coordinates"], 1)
        self.assertEqual(first.statistics["ambiguous_observations"], 3)
        self.assertEqual(first.statistics["unambiguous_nodes"], 0)
        self.assertEqual(len(first.means), 0)
        result = self.result(raw, stock, desc, (first, contexts(raw, stock, desc)[1]))
        self.assertEqual(result["stats"]["ambiguous_endpoints"], 1)
        np.testing.assert_array_equal(result["support_counts"], [[0, 1]])
        for points in result["predictions"].values():
            self.assertEqual(points[0].tobytes(), stock[0].tobytes())
            self.assertEqual(points[1].tobytes(), stock[1].tobytes())

    def test_query_descriptor_must_match_exactly(self):
        raw, stock, desc = basic_pair()
        ctx = contexts(raw, stock, desc)
        other = (descriptors(1, 3), desc[1])
        result = self.result(raw, stock, other, ctx)
        np.testing.assert_array_equal(result["support_counts"], [[0, 1]])
        self.assertEqual(result["stats"]["eligible_matches"], 0)

    def test_both_supported_mask_and_unsupported_byte_preservation(self):
        raw, stock, desc = basic_pair(2)
        first = contexts(raw, stock, desc)[0]
        second = core.build_image_context("B", [
            observation("A", 0, raw[1], stock[1], desc[1], 1),
            observation("D", 2, raw[1][:1], stock[1][:1] - 1, desc[1][:1])])
        result = self.result(raw, stock, desc, (first, second))
        np.testing.assert_array_equal(result["eligible"], [True, False])
        np.testing.assert_array_equal(result["support_counts"], [[1, 1], [1, 0]])
        for points in result["predictions"].values():
            self.assertEqual(points[0].shape, (2, 2))
            self.assertEqual(points[0][1:].tobytes(), stock[0][1:].tobytes())
            self.assertEqual(points[1][1:].tobytes(), stock[1][1:].tobytes())

    def test_mean_and_pixel_center_shrink_equations(self):
        raw, stock, desc = basic_pair(2)
        stock = (stock[0] + np.float32(0.125), stock[1] - np.float32(0.375))
        q = (stock[0] + np.float32(1.5), stock[1] - np.float32(2.5))
        result = self.result(raw, stock, desc, contexts(raw, stock, desc, *q))
        for side in (0, 1):
            expected_mean = ((stock[side].astype(np.float64) + q[side].astype(np.float64)) / 2).astype(np.float32)
            expected_shrink = ((stock[side].astype(np.float64) +
                                np.floor(raw[side].astype(np.float64)) + 0.5) / 2).astype(np.float32)
            np.testing.assert_array_equal(result["predictions"]["mean"][side], expected_mean)
            np.testing.assert_array_equal(result["predictions"]["shrink"][side], expected_shrink)

    def test_huber_matches_independent_scalar_reference_and_ignores_padding(self):
        anchors = np.asarray([[2., 3.], [-1., 0.]], dtype=np.float64)
        votes = np.full((2, 8, 2), 1e20, dtype=np.float64)
        lists = [[(2.2, 3.1), (2.4, 2.8), (100., -100.)], [(-1., 0.)]]
        for row, values in enumerate(lists):
            votes[row, :len(values)] = values
        counts = np.asarray([3, 1], dtype=np.int64)
        expected = [scalar_huber(anchor, values) for anchor, values in zip(anchors.tolist(), lists)]
        np.testing.assert_allclose(core.radial_huber_32(anchors, votes, counts), expected, atol=1e-14, rtol=0)

    def test_huber_quadratic_regime_has_known_midpoint_solution(self):
        anchor = np.asarray([[1., 2.]], dtype=np.float64)
        votes = np.zeros((1, 8, 2), dtype=np.float64)
        votes[0, 0] = [1.2, 2.1]
        result = core.radial_huber_32(anchor, votes, np.asarray([1], dtype=np.int64))
        np.testing.assert_allclose(result, [[1.1, 2.05]], atol=1e-15, rtol=0)

    def test_huber_radial_outlier_has_fixed_iteration_cap(self):
        anchor = np.zeros((1, 2), dtype=np.float64)
        votes = np.zeros((1, 8, 2), dtype=np.float64)
        votes[0, 0] = [30., 40.]
        result = core.radial_huber_32(anchor, votes, np.asarray([1], dtype=np.int64))
        expected_radius = 0.5 * (1 - 0.5 ** 32)
        np.testing.assert_allclose(result, [[0.6 * expected_radius, 0.8 * expected_radius]], atol=1e-14, rtol=0)
        self.assertLess(float(np.linalg.norm(result[0])), 0.5)

    def test_huber_rotation_and_translation_equivariance(self):
        anchors = np.asarray([[1., 2.], [3., -1.]], dtype=np.float64)
        votes = np.zeros((2, 8, 2), dtype=np.float64)
        votes[:, :2] = [[[5., 2.], [1.1, 2.2]], [[2., 3.], [3.2, -1.3]]]
        counts = np.asarray([2, 2], dtype=np.int64)
        original = core.radial_huber_32(anchors, votes, counts)
        rotation = np.asarray([[0., -1.], [1., 0.]], dtype=np.float64)
        translated = np.asarray([10., -20.], dtype=np.float64)
        result = core.radial_huber_32(anchors @ rotation.T + translated,
                                     votes @ rotation.T + translated, counts)
        np.testing.assert_allclose(result, original @ rotation.T + translated, atol=1e-13, rtol=0)

    def test_shuffle_rotates_joint_final_float32_delta_blocks_exactly(self):
        raw, _, desc = basic_pair(5)
        stock = (np.zeros((5, 2), np.float32), np.zeros((5, 2), np.float32))
        q = (np.asarray([[0.1 * (row + 1), 0.05] for row in range(5)], np.float32),
             np.asarray([[-0.05, -0.1 * (row + 1)] for row in range(5)], np.float32))
        result = self.result(raw, stock, desc, contexts(raw, stock, desc, *q), index=19)
        order = [row for _, row in sorted((independent_digest([42, 19, row]), row) for row in range(5))]
        source = order[-1:] + order[:-1]
        for side in (0, 1):
            primary = result["predictions"]["huber"][side]
            shuffled = result["predictions"]["shuffle"][side]
            np.testing.assert_array_equal(shuffled[order], primary[source])
        before = sorted(tuple(float(x) for x in np.r_[result["predictions"]["huber"][0][row],
                        result["predictions"]["huber"][1][row]]) for row in range(5))
        after = sorted(tuple(float(x) for x in np.r_[result["predictions"]["shuffle"][0][row],
                       result["predictions"]["shuffle"][1][row]]) for row in range(5))
        self.assertEqual(before, after)
        self.assertEqual(result["stats"]["shuffled_nonself_rows"], 5)
        self.assertEqual(result["stats"]["shuffled_self_rows"], 0)

    def test_shuffle_adds_delta_to_target_stock_after_fp32_primary_rounding(self):
        raw, stock, desc = basic_pair(3)
        extra0 = stock[0] + np.asarray([[0.1, 0.2], [0.4, -0.1], [-0.3, 0.2]], np.float32)
        extra1 = stock[1] + np.asarray([[0.3, -0.1], [-0.2, 0.3], [0.1, 0.4]], np.float32)
        result = self.result(raw, stock, desc, contexts(raw, stock, desc, extra0, extra1), index=4)
        order = [row for _, row in sorted((independent_digest([42, 4, row]), row) for row in range(3))]
        source = order[-1:] + order[:-1]
        for side in (0, 1):
            base = stock[side].astype(np.float64)
            delta = result["predictions"]["huber"][side].astype(np.float64) - base
            expected = stock[side].copy()
            expected[order] = (base[order] + delta[source]).astype(np.float32)
            np.testing.assert_array_equal(result["predictions"]["shuffle"][side], expected)

    def test_singleton_shuffle_is_explicit_self_shuffle(self):
        raw, stock, desc = basic_pair()
        result = self.result(raw, stock, desc, contexts(raw, stock, desc))
        self.assertEqual(result["stats"]["shuffled_self_rows"], 1)
        self.assertEqual(result["stats"]["shuffled_nonself_rows"], 0)
        for side in (0, 1):
            np.testing.assert_array_equal(result["predictions"]["shuffle"][side],
                                          result["predictions"]["huber"][side])

    def test_joint_virtual_motion_counterexample_can_increase_residual(self):
        raw, _, desc = basic_pair()
        stock = (np.asarray([[1., 1.]], np.float32), np.asarray([[1., 1.]], np.float32))
        q = (np.asarray([[1., 2.]], np.float32), np.asarray([[1., 0.]], np.float32))
        result = self.result(raw, stock, desc, contexts(raw, stock, desc, *q))
        primary = result["predictions"]["huber"]
        self.assertEqual(float(stock[0][0, 1] - stock[1][0, 1]), 0)
        self.assertGreater(abs(float(primary[0][0, 1] - primary[1][0, 1])), 0.5)

    def test_inputs_are_not_mutated_and_outputs_are_finite_float32(self):
        raw, stock, desc = basic_pair(3)
        originals = [value.tobytes() for pair in (raw, stock, desc) for value in pair]
        result = self.result(raw, stock, desc, contexts(raw, stock, desc))
        self.assertEqual(originals, [value.tobytes() for pair in (raw, stock, desc) for value in pair])
        for points in result["predictions"].values():
            for value in points:
                self.assertEqual(value.dtype, np.float32)
                self.assertEqual(value.shape, (3, 2))
                self.assertTrue(np.isfinite(value).all())
        json.dumps(result["stats"], allow_nan=False)

    def test_invalid_input_shapes_dtypes_finite_values_and_names_fail(self):
        raw, stock, desc = basic_pair()
        ctx = contexts(raw, stock, desc)
        bad = raw[0].copy()
        bad[0, 0] = np.nan
        cases = [((raw[0].astype(np.float64), raw[1]), stock, desc),
                 ((bad, raw[1]), stock, desc),
                 (raw, (stock[0][:0], stock[1]), desc),
                 (raw, stock, (desc[0][:, :63], desc[1]))]
        for values in cases:
            with self.subTest(values=[pair[0].shape for pair in values]), self.assertRaises(ValueError):
                self.result(*values, ctx)
        with self.assertRaises(ValueError):
            core.transform_pair(0, ("A", "A"), raw, stock, desc, ctx)
        with self.assertRaises(ValueError):
            core.transform_pair(True, ("A", "B"), raw, stock, desc, ctx)
        with self.assertRaises(ValueError):
            self.result(raw, stock, desc, (ctx[1], ctx[0]))

    def test_duplicate_observation_and_self_image_are_rejected(self):
        raw, stock, desc = basic_pair()
        item = observation("B", 0, raw[0], stock[0], desc[0])
        with self.assertRaises(ValueError):
            core.build_image_context("A", [item, dict(item)])
        with self.assertRaises(ValueError):
            core.build_image_context("B", [item])
        extra = dict(item, score=1.0)
        with self.assertRaises(ValueError):
            core.build_image_context("A", [extra])

    def test_context_tampering_is_rejected(self):
        raw, stock, desc = basic_pair()
        ctx = contexts(raw, stock, desc)[0]
        arrays = {key: value.copy() for key, value in ctx.to_arrays().items()}
        changes = []
        extra = dict(arrays, unexpected=np.zeros(1))
        changes.append(extra)
        wrong_raw = {key: value.copy() for key, value in arrays.items()}
        wrong_raw["raw"][0, 0] += 1
        changes.append(wrong_raw)
        wrong_ids = {key: value.copy() for key, value in arrays.items()}
        wrong_ids["partner_ids"][:] = 0
        changes.append(wrong_ids)
        wrong_stats = dict(arrays, statistics_json=np.asarray("{}"))
        changes.append(wrong_stats)
        for item in changes:
            with self.subTest(keys=sorted(item)), self.assertRaises(ValueError):
                core.ImageContext.from_arrays(item)

    def test_huber_rejects_zero_overflow_counts_and_nonfinite_values(self):
        anchor = np.zeros((1, 2), dtype=np.float64)
        votes = np.zeros((1, 8, 2), dtype=np.float64)
        for value in (0, 9):
            with self.assertRaises(ValueError):
                core.radial_huber_32(anchor, votes, np.asarray([value], dtype=np.int64))
        votes[0, 0, 0] = np.inf
        with self.assertRaises(ValueError):
            core.radial_huber_32(anchor, votes, np.asarray([1], dtype=np.int64))


if __name__ == "__main__":
    unittest.main()
