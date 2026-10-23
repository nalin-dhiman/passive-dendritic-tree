"""Synthetic geometric and analytic cable fixtures; no biological datasets."""
import sys
import unittest
from pathlib import Path
import numpy as np
from scipy.integrate import quad
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
from passive_tree import frustum_geometry, axial_resistance_Mohm, build_tree, transient


def cylinder(points=21):
    # Soma sphere is retained explicitly; cable radius is independent of it.
    a=np.zeros((points,7));a[:,0]=np.arange(1,points+1);a[:,1]=3;a[0,1]=1
    a[:,2]=np.linspace(0,300,points);a[:,5]=2;a[0,5]=5
    parent=np.arange(-1,points-1);a[:,6]=np.r_[-1,np.arange(1,points)]
    return a,parent


class PassiveFixtures(unittest.TestCase):
    def test_frustum_independent_quadrature(self):
        L,r0,r1=17.,.7,2.3
        area,volume=frustum_geometry(L,r0,r1)
        slope=(r1-r0)/L
        self.assertAlmostEqual(volume,quad(lambda x:np.pi*(r0+slope*x)**2,0,L)[0],places=10)
        self.assertAlmostEqual(area,quad(lambda x:2*np.pi*(r0+slope*x)*np.sqrt(1+slope*slope),0,L)[0],places=10)
        self.assertAlmostEqual(axial_resistance_Mohm(L,r0,r1,150),quad(lambda x:.01*150/(np.pi*(r0+slope*x)**2),0,L)[0],places=10)

    def test_cylinder_dc_against_continuum_with_soma(self):
        a,parent=cylinder(161);tree=build_tree(a,parent,Ra=150)
        r=2e-4;L=.03;K=np.pi*r*r/150;q=np.sqrt(2*np.pi*r*1e-4/K)
        cableY=K*q*np.tanh(q*L)*1e6 # µS
        somaY=1e-4*(4*np.pi*5**2)*.01
        expected_root=1/(cableY+somaY);expected_tip=expected_root/np.cosh(q*L)
        ports=[0,len(tree.parent)-1];z,x,b=tree.impedance(ports)
        self.assertLess(abs(z[0,0].real-expected_root),2e-4)
        self.assertLess(abs(z[1,0].real-expected_tip),2e-4)
        self.assertLess(np.max(np.abs(x-tree.hines_solve(b)))/np.max(np.abs(x)),1e-10)
        self.assertLess(np.max(np.abs(z-z.T)),1e-10)
        self.assertGreater(np.min(np.linalg.eigvalsh(z.real)),0)

    def test_subdivision_geometry_and_source_coordinates(self):
        a,parent=cylinder(5);a[2:,5]=np.linspace(1,2,3)
        base=build_tree(a,parent,subdivisions=1);fine=build_tree(a,parent,subdivisions=3)
        self.assertAlmostEqual(base.dendritic_area,fine.dendritic_area,places=9)
        self.assertAlmostEqual(base.dendritic_volume,fine.dendritic_volume,places=9)
        np.testing.assert_allclose(fine.xyz[fine.original_to_mesh],a[:,2:5],atol=1e-14)

    def test_zero_current_rest_and_delivered_charge(self):
        a,parent=cylinder(7);tree=build_tree(a,parent)
        ports=[0,len(tree.parent)-1]
        times,v,_,charge=transient(tree,ports,ports[1],dt=.05,horizon=20,amplitude=0,on=5,off=10)
        self.assertEqual(times[-1],20);self.assertTrue(np.all(v==0));self.assertEqual(charge,0)
        _,v,_,charge=transient(tree,ports,ports[1],dt=.05,horizon=20,amplitude=.02,on=5,off=10)
        self.assertAlmostEqual(charge,.1,places=12);self.assertGreater(v.max(),0)

    def test_parameter_and_radius_validation(self):
        a,parent=cylinder(4)
        for option in ('Ra','gm','cm','radius_scale'):
            for invalid in (0,-1,np.nan,np.inf):
                with self.subTest(option=option,invalid=invalid):
                    with self.assertRaises(ValueError):build_tree(a,parent,**{option:invalid})
        with self.assertRaises(ValueError):build_tree(a,parent,radius_scale=[1,1])
        with self.assertRaises(ValueError):build_tree(a,np.array([-1,2,1,2]))
        with self.assertRaises(ValueError):build_tree(a,parent,subdivisions=1.2)


if __name__=='__main__':unittest.main(verbosity=2)
