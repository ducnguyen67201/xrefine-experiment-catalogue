"""RM035 analytic fixtures only; explanatory truth never enters the selector."""
import copy
import json
import math
import unittest

from selector_035 import eligible, rotation_distance, select, TRANSLATION_EPSILON


def rz(degrees):
    angle = math.radians(degrees)
    c,s = math.cos(angle), math.sin(angle)
    return [c,-s,0.0,s,c,0.0,0.0,0.0,1.0]


def hypothesis(index, angle=0, inliers=1, success=True, translation=None):
    return dict(success=success, R=rz(angle), t=([1.0,0.0,0.0] if translation is None else translation),
                E=[float(index+1)]*9 if success else [], gc_inliers=inliers,
                source=dict(pool='analytic-only',source_pair=0,draw=index,replica=index,
                            policy='conditional_depth',poseprefix='c3_r'+str(index),
                            payloadpin='fixture-pose-'+str(index),maskpin='fixture-GC-mask-'+str(index)))


class PoseSelectorTests(unittest.TestCase):
    def test_unique_medoid_and_equal_pool_max_control(self):
        pool = [hypothesis(0,0,9),hypothesis(1,30,3),hypothesis(2,90,12)]
        self.assertEqual(select(pool), {'FIRST':0,'MEDOID':1,'MAX':2})
        self.assertAlmostEqual(rotation_distance(rz(0),rz(90)), math.pi/2)

    def test_geodesic_respects_rotation_wrap(self):
        pool = [hypothesis(0,170),hypothesis(1,-179),hypothesis(2,-175)]
        self.assertEqual(select(pool)['MEDOID'], 1)
        self.assertAlmostEqual(rotation_distance(rz(179),rz(-179)), math.radians(2))

    def test_duplicate_rotations_and_exact_count_ties_keep_first(self):
        pool = [hypothesis(0,30,7),hypothesis(1,30,7),hypothesis(2,90,3)]
        self.assertEqual(select(pool), {'FIRST':0,'MEDOID':0,'MAX':0})
        identical = [hypothesis(i,0,5) for i in range(3)]
        self.assertEqual(select(identical), {'FIRST':0,'MEDOID':0,'MAX':0})

    def test_two_valid_tie_and_first_replica_failure_preserved(self):
        pool = [hypothesis(0,0,999,False,[0.0,0.0,0.0]),
                hypothesis(1,20,5),hypothesis(2,80,5)]
        self.assertEqual(select(pool), {'FIRST':None,'MEDOID':1,'MAX':1})
        pool[2]['gc_inliers'] = 6
        self.assertEqual(select(pool), {'FIRST':None,'MEDOID':1,'MAX':2})

    def test_one_valid_and_all_invalid_keep_output_denominator(self):
        pool = [hypothesis(0,success=False),hypothesis(1,translation=[0.0,0.0,0.0]),hypothesis(2)]
        self.assertEqual(select(pool), {'FIRST':None,'MEDOID':2,'MAX':2})
        pool[2]['success'] = False
        self.assertEqual(select(pool), {'FIRST':None,'MEDOID':None,'MAX':None})

    def test_qualified_finite_rotation_translation_eligibility(self):
        valid = hypothesis(0)
        self.assertTrue(eligible(valid))
        cases = []
        for vector in ([0.0,0.0,0.0],[TRANSLATION_EPSILON,0.0,0.0],
                       [float('nan'),0.0,0.0],[float('inf'),0.0,0.0],[1e308,1e308,0.0]):
            changed = copy.deepcopy(valid); changed['t'] = vector; cases.append(changed)
        for R in ([1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,-1.0],
                  [1.0,0.01,0.0,0.0,1.0,0.0,0.0,0.0,1.0],
                  [float('nan')]*9,[float('inf')]*9,[1.0]*8):
            changed = copy.deepcopy(valid); changed['R'] = R; cases.append(changed)
        changed = copy.deepcopy(valid); changed['success'] = False; cases.append(changed)
        for changed in cases:
            self.assertFalse(eligible(changed))
        self.assertTrue(eligible(hypothesis(0,translation=[1e-15,0.0,0.0])))

    def test_returns_original_whole_identity_without_mutation_or_mixing(self):
        pool = [hypothesis(0,0,2,[True][0]),hypothesis(1,30,3),hypothesis(2,90,4)]
        pool[0]['t'] = [1.0,0.0,0.0];pool[1]['t'] = [0.0,1.0,0.0];pool[2]['t'] = [0.0,0.0,1.0]
        pool[1]['E'] = [2.0]*18  # Retain all essential branches, never truncate to9.
        before = json.dumps(pool,sort_keys=True)
        selection = select(pool); chosen = pool[selection['MEDOID']]
        self.assertIs(chosen, pool[1])
        self.assertEqual(chosen['t'], [0.0,1.0,0.0])
        self.assertEqual(chosen['E'], [2.0]*18)
        self.assertEqual(chosen['source']['maskpin'], 'fixture-GC-mask-1')
        self.assertEqual(chosen['source']['poseprefix'], 'c3_r1')
        self.assertEqual(json.dumps(pool,sort_keys=True), before)
        self.assertEqual(set(selection), {'FIRST','MEDOID','MAX'})

    def test_consistent_wrong_cluster_is_not_correctness(self):
        # Identity is explanatory fixture truth, never an input field or argument.
        pool = [hypothesis(0,0,1),hypothesis(1,90,10),hypothesis(2,90,10)]
        selection = select(pool)
        self.assertEqual(selection['MEDOID'], 1)
        self.assertEqual(selection['FIRST'], 0)
        self.assertEqual(rotation_distance(pool[0]['R'],rz(0)), 0.0)
        self.assertGreater(rotation_distance(pool[selection['MEDOID']]['R'],rz(0)), 1.0)

    def test_rotation_degeneracy_cannot_resolve_translation(self):
        pool = [hypothesis(0,0,2,translation=[1.0,0.0,0.0]),
                hypothesis(1,0,9,translation=[0.0,1.0,0.0]),
                hypothesis(2,0,1,translation=[0.0,0.0,1.0])]
        self.assertEqual(select(pool), {'FIRST':0,'MEDOID':0,'MAX':1})
        # Equal estimated rotations cannot identify which translation is correct.
        self.assertNotEqual(pool[select(pool)['MEDOID']]['t'], pool[1]['t'])

    def test_malformed_pool_or_provenance_is_not_a_scored_failure(self):
        pool = [hypothesis(i) for i in range(3)]
        for wrong in (pool[:2],pool+[hypothesis(3)]):
            with self.assertRaises(ValueError):select(wrong)
        for field,value in (('source',{}),('gc_inliers',-1),('gc_inliers',True),('E',[1.0]*10)):
            changed = copy.deepcopy(pool);changed[1][field] = value
            with self.assertRaises(ValueError):select(changed)
        changed = copy.deepcopy(pool);changed[1]['error_deg'] = 0.0
        with self.assertRaises(ValueError):select(changed)


if __name__ == '__main__':
    unittest.main(verbosity=2)
