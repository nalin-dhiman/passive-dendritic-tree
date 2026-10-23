"""Passive tree finite-volume implementation. Units: um, ms, mV, nA, uS, nF.

This solver implements a mathematical geometry, not anatomical completeness or
active-membrane certification. It does not redistribute junction surface area,
add spines, or infer absent dendrites. All input nodes must form one rooted tree.
"""
from dataclasses import dataclass
import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import splu


def _positive_finite(value, name):
    value = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(value)) or np.any(value <= 0):
        raise ValueError(f'{name} must be positive and finite')
    return value


def _ports(ports, n):
    raw = np.asarray(ports)
    if raw.ndim != 1 or not len(raw) or not np.all(np.isfinite(raw)) or not np.all(raw == np.floor(raw)):
        raise ValueError('Ports must be a nonempty one-dimensional integer array')
    ports = raw.astype(int)
    if np.any(ports < 0) or np.any(ports >= n) or len(np.unique(ports)) != len(ports):
        raise ValueError('Ports must be distinct valid node indices')
    return ports


def frustum_geometry(length, r0, r1):
    length = _positive_finite(length, 'Length')
    r0 = _positive_finite(r0, 'Radius')
    r1 = _positive_finite(r1, 'Radius')
    length, r0, r1 = np.broadcast_arrays(length, r0, r1)
    volume=np.pi*length*(r0*r0+r0*r1+r1*r1)/3
    area=np.pi*(r0+r1)*np.hypot(length,r1-r0)
    return area, volume


def axial_resistance_Mohm(length, r0, r1, Ra=100.):
    _positive_finite(length, 'Length'); _positive_finite(r0, 'Radius'); _positive_finite(r1, 'Radius'); _positive_finite(Ra, 'Axial resistivity')
    return .01*Ra*np.asarray(length)/(np.pi*np.asarray(r0)*np.asarray(r1))


def read_swc(path):
    a=np.loadtxt(path,comments='#')
    if a.ndim!=2 or a.shape[1]!=7: raise ValueError('SWC requires seven columns')
    if not np.all(np.isfinite(a)):raise ValueError('Nonfinite SWC entries')
    if not np.all(a[:,[0,1,6]]==np.floor(a[:,[0,1,6]])) or np.any(a[:,0]<=0):raise ValueError('SWC identifiers, types and parents must be integers')
    ids=a[:,0].astype(int)
    if len(np.unique(ids))!=len(ids): raise ValueError('Duplicate node identifiers')
    if np.any(a[:,5]<=0) or not np.all(np.isfinite(a)): raise ValueError('Invalid geometry')
    keep=np.isin(a[:,1].astype(int),[1,3,4]); a=a[keep]
    if len(a)<2 or a[0,1]!=1 or np.count_nonzero(a[:,1]==1)!=1:raise ValueError('A single root soma marker and dendritic cables are required')
    idmap={int(row[0]):i for i,row in enumerate(a)}
    parent=np.array([-1 if row[6]<0 else idmap.get(int(row[6]),-999) for row in a])
    if np.count_nonzero(parent==-1)!=1 or parent[0]!=-1: raise ValueError('Expected soma root first')
    if np.any(parent[1:]<0): raise ValueError('Disconnected retained tree')
    if np.any(parent[1:]>=np.arange(1,len(a))): raise ValueError('Require topologically sorted SWC')
    lengths=np.linalg.norm(a[1:,2:5]-a[parent[1:],2:5],axis=1)
    if np.any(lengths<=0): raise ValueError('Zero length edges are unsupported')
    return a,parent


@dataclass
class PassiveTree:
    parent: np.ndarray
    conductance: np.ndarray
    area: np.ndarray
    capacitance: np.ndarray
    membrane_conductance: np.ndarray
    original_to_mesh: np.ndarray
    xyz: np.ndarray
    dendritic_area: float
    dendritic_volume: float
    soma_area: float
    soma_volume: float
    total_length: float

    def diagonal(self,frequency_Hz=0.):
        if not np.isscalar(frequency_Hz) or not np.isfinite(frequency_Hz):raise ValueError('Frequency must be a finite scalar')
        d=self.membrane_conductance.astype(complex)+2j*np.pi*frequency_Hz/1000*self.capacitance
        d=d.copy(); d[1:]+=self.conductance[1:]
        np.add.at(d,self.parent[1:],self.conductance[1:])
        return d

    def matrix(self,frequency_Hz=0.):
        n=len(self.parent); rows=np.arange(1,n); p=self.parent[1:]
        return coo_matrix((np.r_[self.diagonal(frequency_Hz),-self.conductance[1:],-self.conductance[1:]],
                           (np.r_[np.arange(n),rows,p],np.r_[np.arange(n),p,rows])),shape=(n,n)).tocsc()

    def impedance(self,ports,frequency_Hz=0.):
        ports=_ports(ports,len(self.parent))
        b=np.zeros((len(self.parent),len(ports)),complex); b[ports,np.arange(len(ports))]=1
        x=splu(self.matrix(frequency_Hz)).solve(b)
        return x[ports], x, b

    def hines_solve(self,rhs,frequency_Hz=0.):
        """Independent tree elimination, without assembling a sparse matrix."""
        diagonal=self.diagonal(frequency_Hz)
        y=np.asarray(rhs,dtype=complex).copy()
        if y.ndim not in (1,2) or y.shape[0]!=len(self.parent) or not np.all(np.isfinite(y)):raise ValueError('Invalid right-hand side')
        vector=y.ndim==1
        if vector:y=y[:,None]
        for i in range(len(self.parent)-1,0,-1):
            p=self.parent[i]; g=self.conductance[i]
            diagonal[p]-=g*g/diagonal[i]
            y[p]+=g*y[i]/diagonal[i]
        x=np.empty_like(y); x[0]=y[0]/diagonal[0]
        for i in range(1,len(self.parent)):
            x[i]=(y[i]+self.conductance[i]*x[self.parent[i]])/diagonal[i]
        return x[:,0] if vector else x


def build_tree(swc,parent,subdivisions=1,radius_scale=None,Ra=100.,gm=1e-4,cm=1.):
    """Refine every source edge; preserve all source points as physical nodes.

    Soma attachments use constant CHILD radius, rather than introducing a cone
    from the large soma radius. Membrane belongs to exact half-frustum control
    volumes. A single spherical soma area is added to the root only.
    """
    a=np.asarray(swc,dtype=float); parent=np.asarray(parent)
    if a.ndim!=2 or a.shape[1]!=7 or len(a)<2 or not np.all(np.isfinite(a)):raise ValueError('Expected a finite seven-column morphology')
    if parent.shape!=(len(a),) or not np.all(np.isfinite(parent)) or not np.all(parent==np.floor(parent)):raise ValueError('Invalid parent array')
    parent=parent.astype(int)
    if parent[0]!=-1 or np.any(parent[1:]<0) or np.any(parent[1:]>=np.arange(1,len(a))):raise ValueError('Expected a single root in topological order')
    if a[0,1]!=1 or np.count_nonzero(a[:,1]==1)!=1 or not np.all(np.isin(a[:,1],[1,3,4])):raise ValueError('Expected one soma followed by dendritic nodes')
    for value,name in [(Ra,'Ra'),(gm,'gm'),(cm,'cm')]:
        if not np.isscalar(value):raise ValueError(f'{name} must be a scalar')
        _positive_finite(value,name)
    radius=_positive_finite(a[:,5],'Radius').copy()
    if radius_scale is not None:
        scale=_positive_finite(radius_scale,'Radius scale')
        if scale.shape not in [(),(len(a),)]:raise ValueError('Radius scale must be scalar or one factor per source node')
        radius*=scale
        _positive_finite(radius,'Scaled radius')
    if not np.isscalar(subdivisions) or not np.isfinite(subdivisions):raise ValueError('Positive integer subdivision required')
    k=int(subdivisions)
    if k<1 or k!=subdivisions: raise ValueError('Positive integer subdivision required')
    n=1+(len(a)-1)*k
    par=np.full(n,-1,int); g=np.zeros(n); area=np.zeros(n)
    xyz=np.zeros((n,3)); xyz[0]=a[0,2:5]; mapping=np.empty(len(a),int); mapping[0]=0
    vtotal=0.; atotal=0.; ltotal=0.; nexti=1
    for j in range(1,len(a)):
        p=parent[j]; start=mapping[p]
        L=np.linalg.norm(a[j,2:5]-a[p,2:5]); ltotal+=L
        _positive_finite(L,'Edge length')
        rp=radius[j] if a[p,1]==1 else radius[p]
        radii=np.linspace(rp,radius[j],k+1); length=L/k
        for q in range(k):
            i=nexti; nexti+=1; par[i]=start
            r0,r1=radii[q:q+2]; rm=(r0+r1)/2
            a0,_=frustum_geometry(length/2,r0,rm)
            a1,_=frustum_geometry(length/2,rm,r1)
            av,vv=frustum_geometry(length,r0,r1)
            area[start]+=a0;area[i]+=a1
            atotal+=av;vtotal+=vv
            g[i]=1/axial_resistance_Mohm(length,r0,r1,Ra)
            frac=(q+1)/k;xyz[i]=(1-frac)*a[p,2:5]+frac*a[j,2:5]
            start=i
        mapping[j]=start
    soma_area=4*np.pi*radius[0]**2;soma_volume=4*np.pi*radius[0]**3/3
    area[0]+=soma_area
    return PassiveTree(par,g,area,cm*area*1e-5,gm*area*.01,mapping,xyz,
                       float(atotal),float(vtotal),float(soma_area),float(soma_volume),float(ltotal))


def transient(tree,ports,input_node,dt=.025,horizon=250.,amplitude=.01,on=50.,off=150.):
    """Crank-Nicolson with interval-integrated piecewise constant current."""
    ports=_ports(ports,len(tree.parent))
    for value,name in [(dt,'dt'),(horizon,'horizon')]:
        if not np.isscalar(value):raise ValueError(f'{name} must be scalar')
        _positive_finite(value,name)
    if not isinstance(input_node,(int,np.integer)) or not 0<=input_node<len(tree.parent):raise ValueError('Invalid input node')
    if not all(np.isscalar(v) and np.isfinite(v) for v in (amplitude,on,off)) or not 0<=on<=off<=horizon:raise ValueError('Invalid current pulse')
    C=diags(tree.capacitance);G=tree.matrix(0).real
    A=(C/dt+G/2).tocsc();B=(C/dt-G/2).tocsc();factor=splu(A)
    nsteps=round(horizon/dt)
    if abs(nsteps*dt-horizon)>1e-10:raise ValueError('Horizon must align to dt')
    stride=round(.1/dt)
    if stride<1 or abs(stride*dt-.1)>1e-10:raise ValueError('dt must divide 0.1 ms')
    v=np.zeros(len(tree.parent));record=[v[ports].copy()];times=[0.];snapshots={}
    current_integral=0.
    for step in range(nsteps):
        t0=step*dt;t1=(step+1)*dt
        overlap=max(0.,min(t1,off)-max(t0,on));avg=amplitude*overlap/dt
        rhs=B@v;rhs[input_node]+=avg;v=factor.solve(rhs)
        current_integral+=amplitude*overlap
        if (step+1)%stride==0:record.append(v[ports].copy());times.append(t1)
        if any(abs(t1-target)<dt/4 for target in (50,100,150,200,250)):snapshots[str(round(t1,5))]=v.copy()
    return np.asarray(times),np.asarray(record),snapshots,current_integral


def norm2(a):return float(np.linalg.norm(a,2))


def equivalence_metrics(z0,zc):
    delta=zc-z0;den=norm2(z0)
    relative=np.linalg.solve(z0.T,delta.T).T
    return np.array([np.linalg.norm(delta[0])/den,norm2(delta)/den,norm2(relative)])
