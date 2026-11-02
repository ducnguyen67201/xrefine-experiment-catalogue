import unittest

def suite(v, audit):
    import numpy as np
    import torch
    class Tests(unittest.TestCase):
        def test_reject_coarse_aliases(self):
            a = dict(l_fine_inputs=np.zeros((2, 128), np.float32), l_fine_logits=np.zeros((2, 64), np.float32),
                     l_pre_i0=np.array([0, 1], np.int64), l_pre_i1=np.array([0, 1], np.int64),
                     l_dual_0_descriptors=np.zeros((3, 64), np.float32), l_dual_1_descriptors=np.zeros((3, 64), np.float32))
            audit.validate_native_arrays(np, a)
            for bad in (np.array([-1, 1], np.int64), np.array([0, 0], np.int64), np.array([0, 3], np.int64), np.array([0, 1], np.int32), np.array([[0, 1]], np.int64)):
                with self.assertRaises(ValueError): audit.validate_native_arrays(np, dict(a, l_pre_i1=bad))
        def test_reject_extra_fine_columns(self):
            a = dict(l_fine_inputs=np.zeros((1, 129), np.float32), l_fine_logits=np.zeros((1, 64), np.float32))
            with self.assertRaises(ValueError): audit.validate_native_arrays(np, a)
        def test_counts(self):
            self.assertEqual([v.retained_count(n) for n in (0, 3, 5, 8, 10, 11, 331)], [0, 3, 5, 5, 5, 6, 166])
        def test_cosine_and_zero(self):
            a = torch.zeros(3, 64); a[:, :3] = torch.eye(3)
            margin, halves = v.normalized_margin(torch, torch.cat((a * 5, a * 2), 1))
            self.assertTrue(torch.allclose(margin, torch.ones(3)))
            z, _ = v.normalized_margin(torch, torch.zeros(4, 128))
            self.assertTrue(torch.equal(z, torch.zeros(4)))
        def test_negative_valid(self):
            a = torch.zeros(2, 64); b = a.clone(); a[0, 0] = a[1, 1] = 1
            b[0, 1] = b[1, 0] = 1
            score, _ = v.normalized_margin(torch, torch.cat((a, b), 1))
            self.assertTrue(torch.equal(score, -torch.ones(2)))
        def test_nonfinite(self):
            a = torch.zeros(1, 128); a[0, 0] = float('nan')
            with self.assertRaises(ValueError): v.normalized_margin(torch, a)
        def test_empty_single(self):
            for n in (0, 1):
                a, _ = v.normalized_margin(torch, torch.zeros(n, 128))
                self.assertEqual(a.shape, (n,)); self.assertTrue(torch.isfinite(a).all())
        def test_single_cell_ties(self):
            p = np.zeros((11, 2)); c = v.coverage(np, p, p, [20, 20], [20, 20], 6)
            self.assertEqual(c['quotas'].tolist(), [6])
            np.testing.assert_array_equal(v.covered_selected(np, np.ones(11), c), np.arange(6))
        def test_largest_remainder_tie(self):
            p = np.array([[1, 1]] * 3 + [[16, 1]] * 3 + [[16, 16]] * 3, dtype=np.float32)
            c = v.coverage(np, p, p, [20, 20], [20, 20], 5)
            self.assertEqual(c['quotas'].tolist(), [2, 2, 1])
        def test_coarsening(self):
            x = np.array([(i % 4, i // 4) for i in range(16)], dtype=np.float32)
            p0 = np.repeat(x, 16, axis=0) + .1; p1 = np.tile(x, (16, 1)) + .1
            c = v.coverage(np, p0, p1, [4, 4], [4, 4], 128)
            self.assertEqual(int(c['grid']), 2)
            self.assertEqual(int(c['quotas'].sum()), 128)
        def test_borders_overshoot(self):
            p = np.array([[-1, 1], [20, 5], [21, 22], [10, 10], [0, 0]], dtype=np.float32)
            before = p.tobytes(); c = v.coverage(np, p, p, [20, 20], [20, 20], 5)
            self.assertEqual(int(c['outside'].sum()), 3); self.assertEqual(p.tobytes(), before)
            np.testing.assert_array_equal(v.covered_selected(np, np.ones(5), c), np.arange(5))
        def test_empty_full_and_capacity(self):
            rng = np.random.default_rng(17)
            for n in range(36):
                for _ in range(8):
                    p0, p1 = rng.uniform(-1, 22, (n, 2)), rng.uniform(-1, 22, (n, 2))
                    k = v.retained_count(n); c = v.coverage(np, p0, p1, [20, 20], [20, 20], k)
                    ix = v.covered_selected(np, rng.normal(size=n), c)
                    self.assertEqual(len(ix), k); self.assertEqual(len(set(ix)), k)
                    for cell, q in zip(c['occupied'], c['quotas']):
                        self.assertEqual(int(np.sum(c['cells'][ix] == cell)), int(q))
        def test_all10_counts_and_bytes(self):
            n = 11; p = np.arange(2 * n, dtype=np.float32).reshape(n, 2)
            bank = {k: p.copy() for k in ('star0', 'star1', 'primary0', 'primary1', 'stock0', 'stock1')}
            bank['acquired_row_ids'] = np.arange(n, dtype=np.int64)
            before = {k: a.tobytes() for k, a in bank.items()}
            record = dict(pair_index=36, hw0=[20, 20], hw1=[20, 20], K0=np.eye(3), K1=np.eye(3))
            cap = dict(v_scores=np.linspace(-1, 1, n), v_confidence=np.arange(n))
            a, b = v.predictions(np, bank, cap, record), v.predictions(np, bank, cap, record)
            for m in range(10):
                self.assertEqual(len(a[f'm{m}_indices']), n if m in (0, 8, 9) else 6)
                for s in (0, 1):
                    np.testing.assert_array_equal(a[f'm{m}_p{s}'], bank[f'primary{s}'][a[f'm{m}_indices']])
            for k in a: np.testing.assert_array_equal(a[k], b[k])
            self.assertEqual(before, {k: a.tobytes() for k, a in bank.items()})
    return unittest.defaultTestLoader.loadTestsFromTestCase(Tests)
