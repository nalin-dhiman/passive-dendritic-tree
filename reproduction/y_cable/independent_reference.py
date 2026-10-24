"""Independent SI edge-state propagation reference for the ideal sealed Y tree.

This implementation does not form the production nodal-admittance matrix.
Unknowns are voltage and positive-edge-direction axial current at the start
of each of six cylindrical subedges. Five continuity constraints and seven
KCL equations determine those twelve unknowns. Source current is external
point current entering the membrane node. No junction/endcap membrane exists.

Canonical node order: parent_start, junction, daughter1_tip, daughter2_tip,
parent_mid, daughter1_mid, daughter2_mid. Edge directions: root -> junction; junction -> daughter tips.
Input/output units: um, ohm cm, S/cm2, uF/cm2, Hz, nA and mV. Transfer is
mV/nA == MOhm. `gaussian_quadrature` directly integrates the continuous
Fourier integral on Gauss-Legendre nodes; it uses neither an FFT nor modal FVM.
"""
import numpy as np
from functools import lru_cache
from scipy.special import roots_legendre

NODE_NAMES = ('parent_start','junction','daughter1_tip','daughter2_tip','parent_mid','daughter1_mid','daughter2_mid')
EDGES = ((0,4,0),(4,1,0),(1,5,1),(5,2,1),(1,6,2),(6,3,2))
LENGTHS_UM = (400.,400.,400.)


def _propagators(frequency_Hz, radii_um, ra, gm, cm):
    """Exact SI cylinder propagators expressed in mV/nA state coordinates."""
    angular = 2*np.pi*float(frequency_Hz)
    result=[]
    for left,right,region in EDGES:
        radius=float(radii_um[region])*1e-6
        distance=LENGTHS_UM[region]*.5e-6
        axial_coefficient=np.pi*radius**2/(float(ra)*.01)  # S m
        membrane=2*np.pi*radius*(float(gm)*1e4+1j*angular*float(cm)*.01) # S/m
        propagation=np.sqrt(membrane/axial_coefficient)
        x=propagation*distance
        c=np.cosh(x);s=np.sinh(x)
        admittance=axial_coefficient*propagation
        result.append(np.array([[c,-s/admittance*1e-6],[-admittance*s*1e6,c]],complex))
    return result


def solve(frequencies_Hz, radii_um=(1.,.8,.8), ra=100., gm=1e-4, cm=1.):
    """All seven point-current-source transfers and exact physical residuals.

    Returns voltage[F, node, source], J_start/J_end[F, edge, source], and
    source-node KCL/continuity/sealed-boundary residuals. Currents are nA for
    a unit nA source. Reconstructing the endpoint states after the solve gives
    an independently assembled KCL check, not only the linear-system residual.
    """
    if len(radii_um)!=3 or any(not np.isfinite(float(r)) or float(r)<=0 for r in radii_um):
        raise ValueError('Exactly three positive radii required')
    if any(not np.isfinite(float(p)) or float(p)<=0 for p in (ra,gm,cm)):
        raise ValueError('Positive passive constants required')
    frequencies=np.atleast_1d(frequencies_Hz).astype(float)
    if frequencies.ndim!=1 or not len(frequencies) or not np.all(np.isfinite(frequencies)) or np.any(frequencies<0):
        raise ValueError('Finite nonnegative frequency vector required')
    voltage=[];start_current=[];end_current=[];kcl=[];continuity=[];linear=[]
    for frequency in frequencies:
        propagators=_propagators(frequency,radii_um,ra,gm,cm)
        # Each edge contributes [V_start,J_start]. Endpoint rows are propagated.
        endpoints=[]
        for edge,T in enumerate(propagators):
            start=np.zeros((2,12),complex);start[:,2*edge:2*edge+2]=np.eye(2)
            endpoints.append((start,T@start))
        rows=[];rhs=[]
        # Root continuity is trivial; all other multi-edge node voltages agree.
        incident=[]
        for node in range(7):
            members=[]
            for edge,(left,right,_) in enumerate(EDGES):
                if node==left:members.append((endpoints[edge][0],+1))
                if node==right:members.append((endpoints[edge][1],-1))
            incident.append(members)
            for endpoint,_ in members[1:]:
                rows.append(endpoint[0]-members[0][0][0]);rhs.append(np.zeros(7))
        if len(rows)!=5:raise AssertionError('Independent graph continuity count')
        for node,members in enumerate(incident):
            rows.append(sum(direction*endpoint[1] for endpoint,direction in members))
            rhs.append(np.eye(7)[node])
        matrix=np.array(rows);forcing=np.array(rhs)
        states=np.linalg.solve(matrix,forcing)
        starts=np.array([a@states for a,b in endpoints])
        ends=np.array([b@states for a,b in endpoints])
        V=np.empty((7,7),complex);K=np.zeros((7,7),complex);C=[]
        for node,members in enumerate(incident):
            physical=[]
            for edge,(left,right,_) in enumerate(EDGES):
                if node==left:physical.append(starts[edge,0]);K[node]+=starts[edge,1]
                if node==right:physical.append(ends[edge,0]);K[node]-=ends[edge,1]
            V[node]=physical[0]
            C.extend([x-physical[0] for x in physical[1:]])
        voltage.append(V);start_current.append(starts[:,1]);end_current.append(ends[:,1])
        kcl.append(K-np.eye(7));continuity.append(np.array(C))
        linear.append(matrix@states-forcing)
    return dict(frequency_Hz=frequencies,node_names=np.array(NODE_NAMES),
                voltage=np.array(voltage),J_start=np.array(start_current),
                J_end=np.array(end_current),KCL_residual=np.array(kcl),
                continuity_residual_mV=np.array(continuity),linear_residual=np.array(linear))


@lru_cache(maxsize=64)
def _quadrature_kernel(radii_um, ra, gm, cm, order, upper_frequency_Hz):
    nodes,weights=roots_legendre(int(order))
    frequencies=(nodes+1)*float(upper_frequency_Hz)/2
    weights=weights*float(upper_frequency_Hz)/2
    transfer=solve(frequencies,radii_um,ra,gm,cm)['voltage']
    return frequencies,weights,transfer


def gaussian_quadrature(time_ms, source, amp_nA=.01, mu_ms=60., sigma_ms=4.,
                        radii_um=(1.,.8,.8), ra=100., gm=1e-4, cm=1.,
                        order=1024, upper_frequency_Hz=1000.):
    """Independent continuous Fourier Gaussian response at arbitrary times.

    Causal current is the whole-line Gaussian. Its negligible pre-zero tail
    must be bounded by the protocol when comparing a finite-time solver.
    Integral truncation and quadrature errors require prospective doubled-order
    and expanded-frequency checks; this routine does not declare them zero.
    """
    if not 0<=int(source)<7:raise ValueError('Unknown physical source node')
    frequencies,weights,all_transfers=_quadrature_kernel(tuple(map(float,radii_um)),float(ra),float(gm),float(cm),int(order),float(upper_frequency_Hz))
    transfer=all_transfers[:,:,int(source)]
    sigma_s=float(sigma_ms)/1000
    spectrum=float(amp_nA)*sigma_s*np.sqrt(2*np.pi)*np.exp(-.5*(2*np.pi*frequencies*sigma_s)**2)
    coefficient=(2*weights*spectrum)[:,None]*transfer
    times=np.atleast_1d(time_ms).astype(float)
    waveform=np.empty((len(times),7))
    for first in range(0,len(times),128):
        t=times[first:first+128]
        phase=np.exp(2j*np.pi*(t[:,None]-float(mu_ms))/1000*frequencies[None,:])
        waveform[first:first+128]=(phase@coefficient).real
    return waveform


def two_event_waveform(time_ms, sources=(5,3), first_source=0, amp_nA=.01,
                       mu_ms=60., sigma_ms=4., gap_ms=14., **parameters):
    """A-before-B/B-before-A; firstpulse atmu_ms, secondatmu_ms+gap_ms.

    Bothclasses preserve thetotal externallyinjected current pointwise.
    """
    if first_source not in (0,1):raise ValueError('first_source must be 0 or 1')
    early=mu_ms;late=mu_ms+gap_ms
    result=gaussian_quadrature(time_ms,sources[first_source],amp_nA,early,sigma_ms,**parameters)
    result+=gaussian_quadrature(time_ms,sources[1-first_source],amp_nA,late,sigma_ms,**parameters)
    return result
