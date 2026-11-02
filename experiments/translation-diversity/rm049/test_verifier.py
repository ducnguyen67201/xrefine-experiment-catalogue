"""Analytic and actual-boundary checks for the frozen RM049 mechanism."""
import unittest

def suite(v,reference,audit):
    import numpy as np
    class Study(unittest.TestCase):
        def fixture(self,n=16):
            q=np.array([[50+17*(j%4),40+19*(j//4)] for j in range(n)],dtype=np.float32);q1=q.copy();q1[:,0]+=np.linspace(2,7,n,dtype=np.float32)
            ids=np.arange(n,dtype=np.int64);K=np.array([[300.,0.,160.],[0.,300.,120.],[0.,0.,1.]])
            bank=dict(primary0=q,primary1=q1,star0=q.copy(),star1=q1.copy(),stock0=q.copy(),stock1=q1.copy(),acquired_row_ids=ids)
            capture=dict(v_scores=np.ones(n,dtype=np.float32),v_confidence=np.linspace(.3,.9,n,dtype=np.float32));record=dict(K0=K,K1=K,hw0=[240,320],hw1=[240,320],pair_index=1)
            R=np.eye(3);t=np.array([1.,0.,0.]);E=v.essential(np,R,t)
            models=[dict(success=True,R=R.copy(),t=t.copy(),E=E.copy(),fit_ids=(ids if p is None else ids[ids%2==p])) for p in (None,0,1)]
            bounds=dict(max_rotation_spread_deg=180.,max_unsigned_translation_spread_deg=90.,min_heldout_support=0.,min_absolute_gap_per_row=1e-20,max_coplanarity=1.)
            return bank,capture,record,models,dict(normal_cap=.1,bounds=bounds)
        def test_actual_opposite_parity_and_raw_magnitude(self):
            bank,capture,r,models,freeze=self.fixture();g=v.geometry(np,bank,capture,r,models,.1)
            independent,_=audit.reference_geometry(np,bank,capture,r,models,.1,reference)
            for key in ('normals','normal_norms','weights','contributions','opposite_parity_residual'):np.testing.assert_allclose(g[key],independent[key],rtol=0,atol=1e-12)
            self.assertTrue(np.max(g['normal_norms'])<1.);self.assertGreater(np.min(g['normal_norms']),0.)
            models[1]['fit_ids']=models[2]['fit_ids']
            with self.assertRaises(ValueError):v.geometry(np,bank,capture,r,models,.1)
        def test_opposite_model_weights_are_not_E0(self):
            b,c,r,m,f=self.fixture();a=v.geometry(np,b,c,r,m,.1);m[0]['E']=np.eye(3);z=v.geometry(np,b,c,r,m,.1)
            np.testing.assert_array_equal(a['weights'],z['weights'])
            m[1]['E']=v.essential(np,np.eye(3),np.array([0.,1.,0.]));z=v.geometry(np,b,c,r,m,.1)
            np.testing.assert_array_equal(a['weights'][::2],z['weights'][::2]);self.assertTrue(np.any(a['weights'][1::2]!=z['weights'][1::2]))
        def test_declared_failure_zero_placeholders_and_common_fallback(self):
            b,c,r,m,f=self.fixture();m[1]=dict(success=False,R=np.zeros((3,3)),t=np.zeros(3),E=np.zeros((3,3)),fit_ids=m[1]['fit_ids'])
            met=v.model_metrics(np,m);self.assertIsNone(met['max_rotation_spread_deg']);self.assertIsNone(met['min_heldout_support'])
            q=reference.prepass_metrics(np=np,R_all=m[0]['R'],t_all=m[0]['t'],R_even=m[1]['R'],t_even=m[1]['t'],R_odd=m[2]['R'],t_odd=m[2]['t'],successes=[True,False,True],heldout_even_to_odd=None,heldout_odd_to_even=None,eps=1e-12,rotation_tolerance=1e-5)
            self.assertFalse(q['available']);a,w=v.predictions(np,b,c,r,m,f);self.assertTrue(w['fallback'])
            for i in range(8):np.testing.assert_array_equal(a[f'm{i}_indices'],np.arange(16))
        def test_exact_greedy_rank_one_ties_and_negative_completion(self):
            n=9;ids=np.arange(n,dtype=np.int64);p=v.policy(np,np.zeros((n,2)),np.zeros((n,2)),[10,10],[10,10],7)
            A=np.zeros((n,3,3),dtype=np.float64);A[:,0,0]=np.arange(1,n+1);c=np.full(n,.5);a=v.greedy(np,A,ids,p,c,7)
            z=reference.greedy_select(np=np,original_ids=ids,eligible_ids=ids,contributions=A,confidence=c,eligible_cells=p['cells'],quotas={int(k):int(t) for k,t in zip(p['occupied'],p['quotas'])},k=7,eps=1e-12,psd_tolerance=1e-12,candidate_batch_size=3)
            np.testing.assert_array_equal(a['order'],np.arange(7));np.testing.assert_array_equal(a['order'],z['chosen_order']);self.assertEqual(len(a['ids']),7)
            A=np.asarray([np.diag([0.,1.,0.]),np.diag([0.,0.,1.]),np.diag([1.,0.,0.])]);p=v.policy(np,np.zeros((3,2)),np.zeros((3,2)),[10,10],[10,10],3);z=v.greedy(np,A,np.arange(3,dtype=np.int64),p,np.asarray([.9,.8,.7]),3)
            self.assertEqual(len(z['ids']),3);self.assertLess(float(z['history'][-1,0]),float(z['history'][-2,0]))
        def test_raw_quota_population_and_exact_capacity(self):
            ids=np.arange(12,dtype=np.int64);raw=np.array([[1+25*(i%4),1+32*(i//4)] for i in range(12)],dtype=np.float32);small=np.zeros_like(raw)
            p=v.policy(np,raw,raw,[100,100],[100,100],7);q=reference.eligible_joint_quotas(np=np,q0=raw,q1=raw,hw0=[100,100],hw1=[100,100],original_ids=ids,eligible_ids=ids,k=7)
            np.testing.assert_array_equal(p['cells'],q['cells']);self.assertEqual(int(p['quotas'].sum()),7);self.assertFalse(np.array_equal(p['cells'],v.policy(np,small,small,[100,100],[100,100],7)['cells']))
        def test_no_fill_and_original_half(self):
            b,c,r,m,f=self.fixture();c['v_scores'][:]=0.;c['v_scores'][:3]=1.;a,w=v.predictions(np,b,c,r,m,f)
            self.assertEqual(int(a['v_retained_count']),3);self.assertEqual(int(a['v_target_count']),8)
            if not w['fallback']:
                for i in range(1,8):self.assertEqual(len(a[f'm{i}_indices']),3)
        def test_whole_E_information_gate_is_candidate_independent(self):
            b,c,r,m,f=self.fixture();a,w=v.predictions(np,b,c,r,m,f);self.assertFalse(w['fallback'])
            f['bounds']['min_absolute_gap_per_row']=100.;a,w=v.predictions(np,b,c,r,m,f);self.assertTrue(w['fallback'])
            self.assertEqual(w['whole_eligible_information']['count'],16)
            for i in range(1,8):self.assertEqual(len(a[f'm{i}_indices']),16)
        def test_fixed_order_statistic_and_complete_three_records(self):
            q=reference.floor_quantile(values=[0.,1.,2.,3.,4.,5.,6.,7.,8.,9.],probability=.9);self.assertEqual(q['index'],8);self.assertEqual(q['value'],8.)
            with self.assertRaises(ValueError):reference.calibrate_normal_cap(np=np,records=[],expected_record_ids=[0,1,2],expected_room_id=707,probability=.9,eps=1e-12)
        def test_PSD_tolerance_and_information_rank(self):
            a=v.information(np,np.diag([0.,0.,9.]),3);self.assertEqual(a['absolute_gap'],0.);self.assertEqual(a['Q'],0.)
            self.assertEqual(v.information(np,np.diag([-1e-14,1.,2.]),3)['clipped_negative_eigenvalues'],1)
            with self.assertRaises(ValueError):v.information(np,np.diag([-1e-6,1.,2.]),3)
        def test_seed_original_order_cell_shuffle_and_exact_subsets(self):
            b,c,r,m,f=self.fixture();a,w=v.predictions(np,b,c,r,m,f);z,zz=v.predictions(np,b,c,r,m,f)
            self.assertFalse(w['fallback'])
            for key in a:np.testing.assert_array_equal(a[key],z[key])
            self.assertEqual(int(a['C_order'][0]),15)
            for i in range(1,6):
                self.assertEqual(len(a[f'm{i}_indices']),8)
                for cell,quota in zip(a['v_occupied'],a['v_quotas']):self.assertEqual(int(np.sum(a['v_cells'][np.searchsorted(a['v_quality_ids'],a[f'm{i}_indices'])]==cell)),int(quota))
    return unittest.defaultTestLoader.loadTestsFromTestCase(Study)
