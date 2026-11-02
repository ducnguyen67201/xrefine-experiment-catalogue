import unittest

def suite(v):
    import numpy as np
    import torch
    class Tests(unittest.TestCase):
        def test_counts(self):
            self.assertEqual([v.retained_count(n) for n in (0,3,5,8,10,11,331)],[0,3,5,5,5,6,166])
        def test_margin_distinct(self):
            a=torch.eye(3);self.assertTrue(torch.equal(v.margin(torch,a,a),torch.ones(3)))
        def test_margin_repeated(self):
            a=torch.ones((3,2));a=a/torch.linalg.norm(a,dim=1)[:,None]
            self.assertTrue(torch.all(v.margin(torch,a,a)==0))
        def test_two_directions(self):
            a=torch.tensor([[1.,0],[0.,1.]]);b=torch.tensor([[1.,0],[.8,.6]])
            self.assertTrue(torch.allclose(v.margin(torch,a,b),torch.tensor([.2,-.2]),atol=1e-7))
        def test_ties(self):
            np.testing.assert_array_equal(v.selected(np,np.ones(9),4),np.arange(4))
        def test_bad_scores(self):
            with self.assertRaises(ValueError):v.selected(np,np.array([float('nan')]),1)
        def test_spatial(self):
            p=np.array([[0,0],[.5,.5],[1,1],[0,1],[1,0]],dtype=np.float32)
            got=v.spatial(np,p,p,[1,1],[1,1],3)
            np.testing.assert_array_equal(got,[0,1,2])
        def test_subsets_and_no_motion(self):
            n=11;p=np.arange(n*2,dtype=np.float32).reshape(n,2)
            bank=dict(star0=p,star1=p+1,primary0=p+.125,primary1=p+1.25,stock0=p[:7],stock1=p[:7]+1,acquired_row_ids=np.arange(n))
            before={k:a.tobytes() for k,a in bank.items()}
            r=dict(K0=np.eye(3),K1=np.eye(3),hw0=[40,40],hw1=[40,40],pair_index=0)
            out=v.predictions(np,bank,np.tile(np.arange(n),(3,1)),np.arange(n),r)
            self.assertEqual([len(out[f'm{m}_indices']) for m in range(8)],[11,6,6,6,6,6,11,7])
            for m in range(6):
                for s in (0,1):np.testing.assert_array_equal(out[f'm{m}_p{s}'],bank[f'primary{s}'][out[f'm{m}_indices']])
            self.assertEqual(before,{k:a.tobytes() for k,a in bank.items()})
        def test_deterministic(self):
            a=np.arange(40,dtype=np.float32).reshape(20,2);bank={k:a.copy() for k in ('star0','star1','primary0','primary1','stock0','stock1')};bank['acquired_row_ids']=np.arange(20)
            r=dict(K0=np.eye(3),K1=np.eye(3),hw0=[40,40],hw1=[40,40],pair_index=37)
            args=(np,bank,np.tile(np.arange(20),(3,1)),np.arange(20),r)
            x=v.predictions(*args);y=v.predictions(*args)
            for k in x:np.testing.assert_array_equal(x[k],y[k])
        def test_zero_rows(self):
            self.assertEqual(tuple(v.margin(torch,torch.empty(0,64),torch.empty(0,64)).shape),(0,))
    return unittest.defaultTestLoader.loadTestsFromTestCase(Tests)
