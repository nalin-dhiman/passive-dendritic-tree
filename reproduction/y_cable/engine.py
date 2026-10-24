"""Synthetic ideal Y-tree passive cable: exact continuum and vertex finite volume.

Lengths parent/daughter1/daughter2 = 400/400/400 um. Radii are prescribed
positive cylinder radii in um. Parent runs root -> junction, daughters run
junction -> tip. SI continuum uses exp(+i omega t); FVM uses ms/mV/nA/nF/uS.
Only lateral membrane is represented; no endcaps, annular ledges or taper.
Ports are a named NAMES node or (region_index, distance_from_region_start_um).
No biological radius, ATP, transport, tissue or active-mechanism claims.
"""
from functools import lru_cache
import numpy as np
from scipy.linalg import eigh
from scipy.special import erfc, erfcx

LENGTHS_UM = np.array([400.,400.,400.])
LENGTHS_UM.flags.writeable=False
NAMES = ('parent_start','junction','daughter1_tip','daughter2_tip',
         'parent_mid','daughter1_mid','daughter2_mid')
DEFAULT_INPUTS=('daughter1_mid','daughter2_tip')
NAMED_PORTS={'parent_start':(0,0.),'junction':(0,400.),
             'daughter1_tip':(1,400.),'daughter2_tip':(2,400.),
             'parent_mid':(0,200.),'daughter1_mid':(1,200.),'daughter2_mid':(2,200.)}


def _positive(value,name):
    v=float(value)
    if not np.isfinite(v) or v<=0:raise ValueError(name+' must be finite and positive')
    return v


def _radii(value):
    v=np.asarray(value,float)
    if v.shape!=(3,) or not np.all(np.isfinite(v)) or np.any(v<=0):
        raise ValueError('Three positive finite radii in um required')
    return v


def _frequencies(value):
    v=np.atleast_1d(np.asarray(value,float))
    if v.ndim!=1 or len(v)==0 or not np.all(np.isfinite(v)) or np.any(v<0):
        raise ValueError('Finite nonnegative frequency vector in Hz required')
    return v


def _ports(ports):
    out=[]
    for port in ports:
        if isinstance(port,str):
            if port not in NAMED_PORTS:raise ValueError('Unknown physical node '+port)
            region,x=NAMED_PORTS[port]
        else:
            if len(port)!=2:raise ValueError('Port must be name or (region, arclength_um)')
            region,x=port
            if isinstance(region,bool) or int(region)!=region or not 0<=int(region)<3:
                raise ValueError('Region index must be 0,1,2')
            region=int(region);x=float(x)
            if not np.isfinite(x) or not 0<=x<=400:raise ValueError('Arc length outside region')
        # All three incident endpoint references denote one physical junction.
        if region in (1,2) and x==0:region,x=0,400.
        out.append((region,float(x)))
    if not out:raise ValueError('At least one physical port required')
    return out


def _graph(inputs,outputs):
    ports=_ports(inputs)+_ports(outputs)
    positions=[{0.,200.,400.} for _ in range(3)]
    for region,x in ports:positions[region].add(x)
    # Preserve existing seven global node identifiers exactly.
    lookup={NAMED_PORTS[name]:i for i,name in enumerate(NAMES)}
    lookup[(1,0.)]=lookup[(2,0.)]=1
    nextnode=7
    edges=[]
    for region,pos in enumerate(positions):
        pos=sorted(pos)
        for x in pos:
            if (region,x) not in lookup:lookup[(region,x)]=nextnode;nextnode+=1
        for left,right in zip(pos[:-1],pos[1:]):
            edges.append((lookup[(region,left)],lookup[(region,right)],region,right-left,left,right))
    mapped=np.array([lookup[p] for p in ports],int)
    return nextnode,edges,mapped[:len(inputs)],mapped[len(inputs):]


def continuum_state(radii,frequencies_Hz,inputs=DEFAULT_INPUTS,outputs=NAMES,ra=100.,gm=1e-4,cm=1.):
    """Return V[F,O,I] MOhm, outward I[F,E,2,I] nA/nA, residual[F].

    Edge endpoint currents point from node into edge (both endpoints outward).
    A source node's incident outward currents sum to +1 for its source column;
    nonsource sealed/junction node sums are zero. Edge order is parent then
    daughter1 then daughter2, each from low to high physical arclength. Every
    region is split at named midpoint and any additional physical port.
    Voltage responses are deviations from rest; one transfer column uses a
    unit external current (1 A in SI, converted to mV/nA using factor 1e-6).
    """
    r=_radii(radii);freq=_frequencies(frequencies_Hz)
    ra,gm,cm=_positive(ra,'ra'),_positive(gm,'gm'),_positive(cm,'cm')
    inputs=list(inputs);outputs=list(outputs)
    size,edges,src,out=_graph(inputs,outputs)
    result=np.empty((len(freq),len(out),len(src)),complex)
    currents=np.empty((len(freq),len(edges),2,len(src)),complex)
    residual=np.empty(len(freq))
    rhs=np.zeros((size,len(src)),complex);rhs[src,np.arange(len(src))]=1.
    for start in range(0,len(freq),1024):
        f=freq[start:start+1024]
        matrix=np.zeros((len(f),size,size),complex)
        diagonal=[];off=[]
        for left,right,region,length,_,_ in edges:
            radius=r[region]*1e-6;distance=length*1e-6
            axial=np.pi*radius**2/(ra*.01)
            q=np.sqrt(2*np.pi*radius*(gm*1e4+2j*np.pi*f*cm*.01)/axial)
            z=q*distance;decay=np.exp(-z);den=-np.expm1(-2*z)
            d=axial*q*(1+decay**2)/den;o=-axial*q*2*decay/den
            matrix[:,left,left]+=d;matrix[:,right,right]+=d
            matrix[:,left,right]+=o;matrix[:,right,left]+=o
            diagonal.append(d);off.append(o)
        voltage=np.linalg.solve(matrix,np.broadcast_to(rhs,(len(f),)+rhs.shape))
        result[start:start+len(f)]=voltage[:,out,:]*1e-6
        residual[start:start+len(f)]=np.max(abs(matrix@voltage-rhs),axis=(1,2))
        for k,(left,right,*_) in enumerate(edges):
            d=diagonal[k][:,None];o=off[k][:,None]
            currents[start:start+len(f),k,0,:]=d*voltage[:,left,:]+o*voltage[:,right,:]
            currents[start:start+len(f),k,1,:]=o*voltage[:,left,:]+d*voltage[:,right,:]
    return result,currents,residual


def continuum(radii,frequencies_Hz,inputs=DEFAULT_INPUTS,outputs=NAMES,ra=100.,gm=1e-4,cm=1.):
    """Complex transfer voltage[F,O,I], MOhm=mV/nA, at exact physical ports."""
    return continuum_state(radii,frequencies_Hz,inputs,outputs,ra,gm,cm)[0]

exact=continuum


def material(radii,cm=1.,gm=1e-4):
    """Regional/whole prescribed lateral-cylinder totals, no tissue/ATP claim."""
    r=_radii(radii);cm,gm=_positive(cm,'cm'),_positive(gm,'gm')
    area=2*np.pi*r*LENGTHS_UM;volume=np.pi*r*r*LENGTHS_UM
    return {'length_um':1200.,'area_um2':float(sum(area)),'volume_um3':float(sum(volume)),
            'capacitance_pF':float(sum(area)*cm*.01),'leak_conductance_nS':float(sum(area)*gm*10.),
            'section_lengths_um':LENGTHS_UM.tolist(),'section_radii_um':r.tolist(),
            'section_area_um2':area.tolist(),'section_volume_um3':volume.tolist(),
            'surface_to_volume_per_um':float(sum(area)/sum(volume)),
            'endcaps_included':False,'annular_ledges_included':False,
            'scope':'Synthetic prescribed ideal cylinders; no biological material or ATP.'}


def fvm_geometry(radii,nseg=48,ra=100.,gm=1e-4,cm=1.):
    """Vertex-centered conservative FVM, each region nseg equal intervals.

    nseg must be even >=2 to retain all named physical ports. Half of each
    adjacent cylinder interval's membrane/volume is assigned to its vertex;
    the shared junction accumulates contributions from all three regions.
    G uS, C nF, voltage mV, time ms and external input nA.
    """
    r=_radii(radii)
    if isinstance(nseg,bool) or int(nseg)!=nseg or int(nseg)<2 or int(nseg)%2:
        raise ValueError('Even integer nseg>=2 required for persistent midpoint vertices')
    n=int(nseg);ra,gm,cm=_positive(ra,'ra'),_positive(gm,'gm'),_positive(cm,'cm')
    size=3*n+1;G=np.zeros((size,size));area=np.zeros(size);volume=np.zeros(size)
    chains=[np.arange(n+1),np.r_[n,np.arange(n+1,2*n+1)],np.r_[n,np.arange(2*n+1,3*n+1)]]
    links=[]
    for region,chain in enumerate(chains):
        dx=400./n;halfarea=np.pi*r[region]*dx;halfvolume=np.pi*r[region]**2*dx/2
        axial=100*np.pi*r[region]**2/(ra*dx)
        for left,right in zip(chain[:-1],chain[1:]):
            area[left]+=halfarea;area[right]+=halfarea
            volume[left]+=halfvolume;volume[right]+=halfvolume
            G[left,left]+=axial;G[right,right]+=axial
            G[left,right]-=axial;G[right,left]-=axial
            links.append((int(left),int(right),float(axial),region))
    leak=area*gm*.01;C=area*cm*1e-5
    G+=np.diag(leak)
    indices=np.array([0,n,2*n,3*n,n//2,n+n//2,2*n+n//2])
    return {'G_uS':G,'C_nF':C,'area_um2':area,'volume_um3':volume,
            'leak_uS':leak,'named_indices':indices,'chains':chains,'links':links,'nseg':n}


def _fvm_indices(ports,n):
    mapped=[]
    for region,x in _ports(ports):
        j=x*n/400.
        if abs(j-round(j))>1e-10:raise ValueError('FVM physical port not at existing vertex; relocation prohibited')
        j=int(round(j))
        mapped.append(j if region==0 else (n if j==0 else region*n+j))
    return np.array(mapped,int)


@lru_cache(maxsize=24)
def _fvm_modes(radii,n,ra,gm,cm):
    g=fvm_geometry(radii,n,ra,gm,cm)
    inv=1/np.sqrt(g['C_nF'])
    rate,U=eigh(inv[:,None]*g['G_uS']*inv[None,:],check_finite=False)
    if min(rate)<=0:raise ArithmeticError('Passive decay spectrum must be strictly positive')
    return inv,rate,U


def fvm_transfer(radii,frequencies_Hz,inputs=DEFAULT_INPUTS,outputs=NAMES,ra=100.,gm=1e-4,cm=1.,nseg=48):
    """Independent finite-volume harmonic response[F,O,I], MOhm=mV/nA."""
    r=tuple(_radii(radii));f=_frequencies(frequencies_Hz)
    # Validate before cache lookup, including bool rejection and fixed ports.
    fvm_geometry(r,nseg,ra,gm,cm);n=int(nseg)
    src=_fvm_indices(inputs,n);out=_fvm_indices(outputs,n)
    inv,rates,U=_fvm_modes(r,n,float(ra),float(gm),float(cm))
    co=(U[out,:]*inv[out,None]);ci=(U[src,:]*inv[src,None])
    coefficients=np.einsum('om,im->oim',co,ci)
    answer=np.empty((len(f),len(out),len(src)),complex)
    for start in range(0,len(f),512):
        denominator=1/(rates[None,:]+2j*np.pi*f[start:start+512,None]/1000.)
        answer[start:start+len(denominator)]=np.einsum('fm,oim->foi',denominator,coefficients,optimize=True)
    return answer


def gaussian_mode(decay_per_ms,times_ms,center_ms,sigma_ms):
    """Exact exponential/Gaussian convolution (-infinity,t], in ms."""
    t=np.asarray(times_ms,float)
    if t.ndim!=1 or not np.all(np.isfinite(t)):raise ValueError('Finite vector times required')
    center=float(center_ms)
    if not np.isfinite(center):raise ValueError('Finite pulse center required')
    sigma=_positive(sigma_ms,'sigma_ms')
    lam=np.asarray(decay_per_ms)[None,:];x=t[:,None]-center
    z=(lam*sigma*sigma-x)/(np.sqrt(2)*sigma)
    X,L=np.broadcast_to(x,z.shape),np.broadcast_to(lam,z.shape)
    result=np.empty(z.shape);positive=z>=0
    result[positive]=np.exp(-.5*(X[positive]/sigma)**2)*erfcx(z[positive])
    result[~positive]=np.exp(-L[~positive]*X[~positive]+.5*(L[~positive]*sigma)**2)*erfc(z[~positive])
    return sigma*np.sqrt(np.pi/2)*result


def fvm_gaussian(radii,times_ms,pulses,nseg=48,inputs=DEFAULT_INPUTS,outputs=NAMES,ra=100.,gm=1e-4,cm=1.):
    """Modal FVM delta-V[T,O] mV for Gaussian pulse sum from -infinity.

    pulses dictionaries have port (index into inputs), amplitude_nA>0,
    center_ms finite, sigma_ms>0. No rest potential, noise or normalization.
    """
    r=tuple(_radii(radii));g=fvm_geometry(r,nseg,ra,gm,cm);n=int(nseg)
    src=_fvm_indices(inputs,n);out=_fvm_indices(outputs,n)
    inv,rates,U=_fvm_modes(r,n,float(ra),float(gm),float(cm))
    t=np.asarray(times_ms,float)
    if t.ndim!=1 or not np.all(np.isfinite(t)):raise ValueError('Finite vector times required')
    answer=np.zeros((len(t),len(out)))
    for pulse in pulses:
        port=pulse['port']
        if isinstance(port,bool) or int(port)!=port or not 0<=int(port)<len(src):raise ValueError('Pulse port index invalid')
        amp=_positive(pulse['amplitude_nA'],'amplitude_nA')
        coeff=(U[out,:]*inv[out,None])*(U[src[int(port)],:]*inv[src[int(port)]])[None,:]
        answer+=amp*gaussian_mode(rates,t,pulse['center_ms'],pulse['sigma_ms'])@coeff.T
    return answer
