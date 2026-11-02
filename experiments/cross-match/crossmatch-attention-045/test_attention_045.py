"""Meaningful affected checks; called inside the authenticated local worker."""
import unittest
import math


def make_suite(core):
    np, torch = core.np, core.torch

    class AttentionChecks(unittest.TestCase):
        def setUp(self):
            torch.manual_seed(2718)
            self.g = torch.randn(12, 128)
            self.ids = np.arange(12, dtype=np.int64)
            geom = np.arange(48, dtype=np.float64).reshape(12, 4)/50
            self.neighbors = core.build_neighbors(geom, self.ids)
            self.head = core.Adapter()

        def indices(self, arm):
            return core.context(self.g, self.neighbors, arm, self.ids, 'RM045/fixture')

        def test_capacity_and_identical_initialization(self):
            heads = []
            for _ in range(3):
                torch.manual_seed(20261027)
                heads.append(core.Adapter())
            self.assertTrue(all(sum(p.numel() for p in h.parameters()) == 79520 for h in heads))
            self.assertEqual(len({core._module_digest(h) for h in heads}), 1)

        def test_zero_film_is_exact_neutral(self):
            for arm in core.ARMS:
                c = self.head.attend(self.g, self.indices(arm))
                self.assertTrue(bool((self.head(torch.cat((self.g, c), 1)) == 0).all()))

        def test_independent_cannot_read_other_features(self):
            indices = self.indices('independent')
            changed = self.g.clone()
            changed[1:] += torch.randn_like(changed[1:])*100
            a = self.head.attend(self.g, indices)[0]
            b = self.head.attend(changed, indices)[0]
            self.assertTrue(torch.equal(a, b))

        def test_candidate_reads_other_features(self):
            indices = self.indices('context')
            changed = self.g.clone()
            changed[indices[0, 0]] += torch.randn(128)*10
            self.assertFalse(torch.equal(self.head.attend(self.g, indices)[0],
                                         self.head.attend(changed, indices)[0]))

        def test_all_attention_parameters_receive_finite_gradients(self):
            for arm in core.ARMS:
                self.head.zero_grad(set_to_none=True)
                self.head.attend(self.g, self.indices(arm)).square().sum().backward()
                for parameter in (self.head.rank_embedding, *self.head.attention.parameters()):
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(bool(torch.isfinite(parameter.grad).all()))
                qk = self.head.attention.in_proj_weight.grad[:256]
                self.assertTrue(bool((qk != 0).any()), arm)

        def test_attention_matches_independent_scalar_formula(self):
            g = self.g[:3]
            idx = torch.tensor([[1, 2], [0, 2], [0, 1]])
            actual = self.head.attend(g, idx).detach().double()
            attn = self.head.attention
            weights = attn.in_proj_weight.detach().double().tolist()
            biases = attn.in_proj_bias.detach().double().tolist()
            gv = g.double().tolist()
            emb = self.head.rank_embedding.detach().double().tolist()
            def linear(x, offset):
                return [math.fsum(a*b for a,b in zip(x, weights[offset+d]))+biases[offset+d]
                        for d in range(128)]
            output_w = attn.out_proj.weight.detach().double().tolist()
            output_b = attn.out_proj.bias.detach().double().tolist()
            expected = []
            for i in range(3):
                q = linear(gv[i], 0)
                tokens = [[a+b for a,b in zip(gv[int(j)],emb[r])] for r,j in enumerate(idx[i])]
                ks, vs = [linear(t,128) for t in tokens], [linear(t,256) for t in tokens]
                joined = []
                for h in range(4):
                    sl = slice(h*32,(h+1)*32)
                    logits = [math.fsum(a*b for a,b in zip(q[sl],k[sl]))/math.sqrt(32) for k in ks]
                    values = [math.exp(v-max(logits)) for v in logits]
                    probs = [v/math.fsum(values) for v in values]
                    joined += [math.fsum(p*v[d] for p,v in zip(probs,vs)) for d in range(h*32,(h+1)*32)]
                expected.append([math.fsum(a*b for a,b in zip(joined,row))+bias
                                 for row,bias in zip(output_w, output_b)])
            self.assertTrue(torch.allclose(actual,torch.tensor(expected,dtype=torch.float64),atol=2e-6,rtol=2e-6))

        def test_shuffle_is_fixed_degree_and_other_ids(self):
            a,b = self.indices('context'),self.indices('shuffled')
            self.assertEqual(a.shape,b.shape)
            self.assertTrue(bool((a != b).all()))
            self.assertTrue(bool((b != torch.arange(12)[:,None]).all()))
            self.assertTrue(torch.equal(b,self.indices('shuffled')))

        def test_whole_pair_graph_crosses_256_boundary(self):
            ids = np.arange(257,dtype=np.int64)
            geometry = np.zeros((257,4),np.float64)
            geometry[:,0] = np.arange(257)
            graph = core.build_neighbors(geometry,ids)
            self.assertIn(256,graph[255])

        def test_empty_singleton_and_small_pairs(self):
            for n in (0,1,2,8,9):
                ids = np.arange(n,dtype=np.int64)
                g = torch.zeros(n,128)
                neighbors = core.build_neighbors(np.zeros((n,4),np.float64),ids)
                for arm in core.ARMS:
                    indices = core.context(g,neighbors,arm,ids,'RM045/small')
                    result = self.head.attend(g,indices)
                    self.assertEqual(tuple(result.shape),(n,128))
                    self.assertTrue(bool(torch.isfinite(result).all()))

        def test_bad_indices_rejected(self):
            with self.assertRaises(ValueError):
                self.head.attend(self.g,torch.full((12,8),12,dtype=torch.int64))

        def test_strict_checkpoint_roundtrip(self):
            other = core.Adapter()
            other.load_state_dict(self.head.state_dict(),strict=True)
            self.assertTrue(torch.equal(other.attend(self.g,self.indices('context')),
                                        self.head.attend(self.g,self.indices('context'))))

        def test_query_chunking_preserves_full_candidate_search(self):
            g = torch.cat([self.g]*22)[:257]
            indices = torch.arange(8)[None].repeat(257,1)
            chunked = self.head.attend(g,indices)
            tokens = g[indices]+self.head.rank_embedding[None]
            direct,_ = self.head.attention(g[:,None],tokens,tokens,need_weights=False)
            self.assertTrue(torch.allclose(chunked,direct[:,0],atol=2e-6,rtol=2e-6))

    return unittest.defaultTestLoader.loadTestsFromTestCase(AttentionChecks)
