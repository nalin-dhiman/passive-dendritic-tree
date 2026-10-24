"""Known-answer and interface checks for the portable Y-cable experiment."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

REPO = Path(__file__).resolve().parents[1]
HERE = REPO / 'reproduction/y_cable'
sys.path.insert(0, str(HERE))
import engine as Y
import independent_reference as Independent
import metrics as Metrics


class YReproductionChecks(unittest.TestCase):
    def test_preserved_numerical_sources(self):
        provenance=json.loads((HERE/'source_provenance.json').read_text())
        for name, record in provenance['source_modules'].items():
            self.assertEqual(hashlib.sha256((HERE/name).read_bytes()).hexdigest(),record['source_sha256'])

    def test_known_transfer_matrices_and_independent_currents(self):
        frequencies=np.array([0.,10.,100.])
        values,current,residual=Y.continuum_state((1.,.8,.8),frequencies,inputs=Y.NAMES,outputs=Y.NAMES)
        second=Independent.solve(frequencies)
        with np.load(REPO/'reproduction/reference/y_impedance_reference.npz') as reference:
            np.testing.assert_allclose(values,reference['Z_baseline_nominal'][[0,10,100]],atol=1e-10,rtol=1e-12)
        np.testing.assert_allclose(values,second['voltage'],atol=1e-10,rtol=1e-12)
        np.testing.assert_allclose(current[:,:,0],second['J_start'],atol=1e-11,rtol=1e-10)
        np.testing.assert_allclose(current[:,:,1],-second['J_end'],atol=1e-11,rtol=1e-10)
        self.assertLess(float(np.max(residual)),1e-10)

    def test_matched_material_and_constructed_current(self):
        config=json.loads((HERE/'config.json').read_text())
        geometries={g['id']:g['radii'] for g in config['geometries']}
        baseline_volume=Y.material(geometries['baseline'])['volume_um3']
        for name in ['uniform','selective']:
            self.assertAlmostEqual(1-Y.material(geometries[name])['volume_um3']/baseline_volume,.0591,places=12)
        f=np.array([0.,10.,100.]);base=Y.continuum(geometries['baseline'],f,inputs=Y.NAMES,outputs=Y.NAMES)
        changed=Y.continuum(geometries['selective'],f,inputs=Y.NAMES,outputs=Y.NAMES)
        for kind in ['root_null','response_relative']:
            arrays,record=Metrics.witness_data(base,changed,f,config,kind)
            self.assertEqual(record['status'],'PASS')
            self.assertAlmostEqual(np.linalg.norm(arrays['input_phasor_nA']),.01,places=14)
        _,null=Metrics.witness_data(base,changed,f,config,'root_null')
        self.assertLess(null['root_delta_absolute_mV'],1e-10)
        self.assertGreater(null['max_delta_absolute_mV'],.1)

    def test_nonempty_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            marker=Path(folder)/'keep.txt';marker.write_text('preserve this output')
            result=subprocess.run([sys.executable,str(HERE/'run.py'),'--output',folder],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertEqual(marker.read_text(),'preserve this output')
            self.assertEqual([p.name for p in Path(folder).iterdir()],['keep.txt'])


if __name__=='__main__':unittest.main()
