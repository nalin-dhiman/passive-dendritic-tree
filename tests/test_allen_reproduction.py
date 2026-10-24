"""Small source and physical-invariant contracts; full reproduction is a CLI run."""
import hashlib
import json
import unittest
import tempfile
import shutil
from pathlib import Path

import numpy as np
from reproduction.allen.run import HERE, REFERENCE, analytic_fixtures, geometry, secondary_metrics, verify_outputs
from reproduction.allen.engine import build_tree, equivalence_metrics


class AllenReproductionContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config=json.loads((HERE/'config.json').read_text())
        cls.swc,cls.parent,cls.ports,cls.scales,cls.primary,cls.info=geometry(HERE/'data/588402092.swc',cls.config)

    def test_exact_source_and_fixed_physical_ports(self):
        self.assertEqual(hashlib.sha256((HERE/'data/588402092.swc').read_bytes()).hexdigest(),
            '8d3c5fc12fbee445d8cdbe3a2281029e4f87b219689856477e82e60f46d29bb9')
        self.assertEqual(len(self.swc),2534)
        self.assertEqual(self.swc[self.ports,0].astype(int).tolist(),[1,1621,1783,2165,1216,182,708])
        self.assertEqual(self.info['selected_basal_primary_source_id'],724)

    def test_material_change_has_two_explicit_denominators(self):
        baseline=build_tree(self.swc,self.parent,1,self.scales['baseline'])
        for candidate in ['uniform','selective']:
            tree=build_tree(self.swc,self.parent,1,self.scales[candidate])
            self.assertAlmostEqual(1-tree.dendritic_volume/baseline.dendritic_volume,.0591,places=12)
            whole=1-(tree.dendritic_volume+tree.soma_volume)/(baseline.dendritic_volume+baseline.soma_volume)
            self.assertAlmostEqual(whole,.02694504448359,places=12)
            self.assertEqual(tree.soma_area,baseline.soma_area)
            self.assertEqual(tree.soma_volume,baseline.soma_volume)

    def test_refinement_keeps_original_coordinates(self):
        tree=build_tree(self.swc,self.parent,2,self.scales['selective'])
        np.testing.assert_array_equal(tree.xyz[tree.original_to_mesh[self.ports]],self.swc[self.ports,2:5])
        self.assertEqual(len(tree.parent),5067)

    def test_independent_geometry_and_continuum_fixtures(self):
        fixtures=analytic_fixtures()
        self.assertEqual(len(fixtures),13)
        self.assertTrue(all(row['pass'] for row in fixtures))

    def test_normalizations_remain_distinct(self):
        data=np.load(REFERENCE/'allen_impedance.npz')
        primary=equivalence_metrics(data['baseline_k8'][-1],data['selective_k8'][-1])
        secondary,_=secondary_metrics(data['baseline_k8'][-1],data['selective_k8'][-1])
        self.assertLess(primary[0],.01)
        self.assertGreater(secondary[0],.049)
        self.assertLess(secondary[0],.05)
        self.assertGreater(primary[1],.28)
        self.assertAlmostEqual(primary[2],secondary[2],places=12)

    def test_verifier_rejects_missing_transient_inventory(self):
        expected=json.loads((REFERENCE/'allen_expected.json').read_text())
        with tempfile.TemporaryDirectory() as temporary:
            output=Path(temporary)
            shutil.copyfile(REFERENCE/'allen_impedance.npz',output/'impedance_matrices.npz')
            (output/'geometry_and_ports.json').write_text(json.dumps(expected['geometry']))
            maxima={}
            for family,key in [('primary','primary_maxima'),('secondary','secondary_maxima')]:
                maxima[family]={}
                for candidate in ['uniform','selective']:
                    old=expected[key][candidate]
                    maxima[family][candidate]={
                        'root_gain':old['root_gain' if family=='primary' else 'root_row_relative_gain'],
                        'all_port_gain':old['all_port_gain' if family=='primary' else 'max_port_row_relative_gain'],
                        'response_relative':old['response_relative']}
            for values in maxima.values():
                for candidate in values.values():
                    for row in candidate.values():
                        row['error_plus_allowance']=float(row['error_plus_allowance'])
            (output/'metric_maxima.json').write_text(json.dumps(maxima))
            with self.assertRaisesRegex(ValueError,'Transient inventory incomplete'):
                verify_outputs(output,include_transients=True)

    def test_failed_transient_scope_is_retained(self):
        expected=json.loads((REFERENCE/'allen_expected.json').read_text())
        self.assertFalse(expected['transient_gate']['pass'])
        self.assertEqual(expected['transient_gate']['runs_completed'],18)
        self.assertGreater(expected['transient_gate']['worst_finest_pair_mV'],.05)
        self.assertEqual(self.config['transient']['max_final_voltage_difference_mV'],.0025)


if __name__=='__main__':
    unittest.main()
