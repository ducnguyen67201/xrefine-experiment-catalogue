"""Affected production-policy fixtures; no dataset/pose/GT access."""
import unittest
import numpy as np
def suite(v,unused=None):
    class Checks(unittest.TestCase):
        def go(self,s,c=None,p=None,cuts=None):
            s=np.asarray(s,dtype=np.float32);n=len(s)
            c=np.ones(n,dtype=np.float32) if c is None else np.asarray(c,dtype=np.float32)
            p=np.zeros((n,2),dtype=np.float32) if p is None else np.asarray(p,dtype=np.float32)
            return v.selections(np,p,p,(100,100),(100,100),s,c,cuts or dict(margin=.5,confidence=.5),7)
        def test_no_weak_cell(self):
            p=[[0,0]]*9+[[99,99]]
            ix,a=self.go([1]*9+[0],p=p)
            self.assertNotIn(9,ix[1]);self.assertEqual(len(ix[1]),5)
        def test_none(self):
            ix,a=self.go([0]*10)
            self.assertEqual([len(x) for x in ix],[10]+[0]*7)
            self.assertEqual(a['v_labels'].shape,(0,4))
        def test_fewer_than_five_no_fill(self):
            ix,a=self.go([1,0,0,0,0,0])
            self.assertEqual(ix[1].tolist(),[0]);self.assertEqual(len(ix[2]),1)
        def test_empty(self):
            ix,a=self.go([])
            self.assertTrue(all(len(x)==0 for x in ix))
        def test_inclusive_both(self):
            ix,a=self.go([.5,.5,1,1],[.5,.49,.5,.1])
            self.assertEqual(a['v_quality_ids'].tolist(),[0,2])
        def test_original_ids(self):
            ix,a=self.go([0,1,0,1,0,1,0,1,0,1])
            self.assertEqual(ix[1].tolist(),[1,3,5,7,9])
        def test_no_double_half(self):
            ix,a=self.go([1]*7+[0]*3)
            self.assertEqual(len(ix[1]),5)
        def test_stable_ties(self):
            ix,a=self.go([1]*10)
            self.assertEqual(ix[1].tolist(),list(range(5)))
        def test_joint_shuffle_count(self):
            ix,a=self.go([1,1,0,0,1,1,0,0],[1,0,1,0,1,0,1,0])
            self.assertEqual(int(a['v_eligible'].sum()),int(a['v_joint_eligible'].sum()))
        def test_within_gate_fixed_pool(self):
            ix,a=self.go([1]*6+[0]*6)
            self.assertTrue(set(ix[4])<=set(range(6)))
        def test_all_eligible_fit_same_selection(self):
            ix,a=self.go([1]*3+[0]*7)
            self.assertEqual(ix[1].tolist(),ix[7].tolist())
        def test_repeat_determinism(self):
            x,a=self.go(np.arange(20)/20);y,b=self.go(np.arange(20)/20)
            self.assertEqual([i.tolist() for i in x],[i.tolist() for i in y])
        def test_boundary_labels_not_coordinates(self):
            p=np.array([[-1,101]]*10,dtype=np.float32);copy=p.copy()
            ix,a=self.go([1]*10,p=p)
            self.assertTrue(np.array_equal(p,copy));self.assertEqual(int(a['v_outside'].sum()),10)
        def test_nonfinite_rejected(self):
            with self.assertRaises(ValueError):self.go([1,float('nan')])
        def test_eligible_coverage(self):
            p=[[1,1]]*6+[[80,80]]*4;ix,a=self.go([1]*10,p=p)
            self.assertTrue(set(range(6,10))&set(ix[1]))
            self.assertTrue(set(range(6))&set(ix[1]))
    return unittest.defaultTestLoader.loadTestsFromTestCase(Checks)
