"""Independently verify reconstructed-arbor outputs using NumPy and the standard library.

Usage: python reproduction/verify_allen_arithmetic.py --results DIRECTORY --output JSON --require-transients
No electrical engine, acquisition function or metric producer is imported.
"""
from pathlib import Path
import argparse,csv,json,hashlib
import numpy as np

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results',dest='output_root',type=Path,required=True);p.add_argument('--output',dest='report',type=Path,required=True);p.add_argument('--require-transients',action='store_true');a=p.parse_args()
O=a.output_root;R=Path(__file__).resolve().parents[1];checks=[]
def check(name,ok,value=None):
 checks.append({'check':name,'pass':bool(ok),'value':float(value) if value is not None else None})
def close(name,x,y,atol=1e-10):
 e=float(np.max(np.abs(np.asarray(x)-np.asarray(y))));check(name,e<=atol,e)
def spec(x):return np.linalg.svd(x,compute_uv=False)[...,0]
def readcsv(name):return list(csv.DictReader((O/name).open()))
with np.load(O/'impedance_matrices.npz',allow_pickle=False) as z: Z={k:z[k].copy() for k in z.files}
with np.load(R/'reproduction/reference/allen_impedance.npz',allow_pickle=False) as z: REF={k:z[k].copy() for k in z.files}
freq=Z.pop('frequencies_Hz');REF.pop('frequencies_Hz')
check('16_matrix_groups',len(Z)==16)
check('128_frequency_conditions',sum(len(v) for v in Z.values())==128)
worst=0
for key,z in Z.items():
 e=max(spec(z-REF[key])/spec(REF[key]));worst=max(worst,float(e));check(key+':saved',e<=1e-8,e)
 close(key+':reciprocity',z,z.swapaxes(-1,-2),1e-7)
 h=(z+z.conj().swapaxes(-1,-2))/2
 check(key+':passive',np.linalg.eigvalsh(h).min()>0)
for table,secondary in [('electrical_equivalence.csv',False),('secondary_row_normalization.csv',True)]:
 rows=readcsv(table);check(table+':72_rows',len(rows)==72)
 for row in rows:
  f=list(freq).index(float(row['frequency_Hz']));name=row['candidate'];values=[]
  for k in [8,4]:
   b=Z['baseline_k'+str(k)][f];d=Z[name+'_k'+str(k)][f]-b
   relative=spec(d@np.linalg.inv(b))
   if secondary:
    rs=np.linalg.norm(d,axis=1)/np.maximum(np.linalg.norm(b,axis=1),1)
    values.append([rs[0],rs.max(),relative])
   else:values.append([np.linalg.norm(d[0])/spec(b),spec(d)/spec(b),relative])
  j=['root_gain','all_port_gain','response_relative'].index(row['criterion']);raw=values[0][j];alpha=abs(raw-values[1][j]);label=table+':'+name+':'+row['frequency_Hz']+':'+row['criterion']
  close(label+':raw',raw,float(row['error']),1e-11);close(label+':allowance',alpha,float(row['engineering_discrepancy_allowance']),1e-11)
  check(label+':verdict',('PASS' if raw+alpha<=.05 else 'FAIL')==row['verdict'])
rows=readcsv('allowance_sensitivity.csv');check('144_allowance_sensitivities',len(rows)==144);check('allowance_verdicts_preserved',all(r['verdict_unchanged']=='True' for r in rows))
# Geometry/material arithmetic, independent of the finite-volume engine.
with np.load(O/'retained_geometry.npz',allow_pickle=False) as z:G={k:z[k].copy() for k in z.files}
swc=G['swc'];par=G['parent'];ports=G['ports'];source=R/'reproduction/allen/data/588402092.swc'
check('source_sha256',hashlib.sha256(source.read_bytes()).hexdigest()=='8d3c5fc12fbee445d8cdbe3a2281029e4f87b219689856477e82e60f46d29bb9')
check('2534_source_points',len(swc)==2534);check('source_port_identities',list(swc[ports,0].astype(int))==[1,1621,1783,2165,1216,182,708])
mat=readcsv('material_integrals.csv');check('16_material_conditions',len(mat)==16)
for row in mat:
 scale=G['scale_'+row['candidate']];r=swc[:,5]*scale;length=np.linalg.norm(swc[1:,2:5]-swc[par[1:],2:5],axis=1);r1=r[1:];r0=np.where(par[1:]==0,r1,r[par[1:]])
 area=np.sum(np.pi*(r0+r1)*np.hypot(length,r1-r0));vol=np.sum(np.pi*length*(r0*r0+r0*r1+r1*r1)/3)
 label=row['candidate']+':k'+row['mesh'];close(label+':volume',vol,float(row['dendritic_volume_um3']),1e-8);close(label+':area',area,float(row['dendritic_area_um2']),1e-8)
 close(label+':sphere',4*np.pi*swc[0,5]**3/3,float(row['soma_volume_um3']),1e-9)
 close(label+':capacitance',(area+4*np.pi*swc[0,5]**2)*.01,float(row['whole_capacitance_pF']),1e-9)
# Full-node harmonic field Kirchhoff balance, independently assembled coefficients.
with np.load(O/'harmonic_full_fields.npz',allow_pickle=False) as z:F={k:z[k].copy() for k in z.files}
k=8;n=1+(len(swc)-1)*k;mapping=np.arange(len(swc))*k;mapping[1:]=1+(np.arange(1,len(swc))-1)*k+k-1
for name in ['baseline','uniform','selective']:
 scale=G['scale_'+name];r=swc[:,5]*scale;parent=np.full(n,-1,int);conductance=np.zeros(n);area=np.zeros(n)
 for j in range(1,len(swc)):
  first=1+(j-1)*k;end=first+k;p0=mapping[par[j]];parent[first]=p0;parent[first+1:end]=np.arange(first,end-1)
  length=np.linalg.norm(swc[j,2:5]-swc[par[j],2:5])/k;r0=r[j] if par[j]==0 else r[par[j]];radii=np.linspace(r0,r[j],k+1);lo,hi=radii[:-1],radii[1:];mid=(lo+hi)/2
  idx=np.arange(first,end);conductance[idx]=np.pi*lo*hi/length
  la=np.pi*(lo+mid)*np.hypot(length/2,mid-lo);ra=np.pi*(mid+hi)*np.hypot(length/2,hi-mid)
  np.add.at(area,parent[idx],la);area[idx]+=ra
 area[0]+=4*np.pi*swc[0,5]**2
 v=F[name+'_phasor_per_nA'];lhs=(area*1e-6+2j*np.pi*10/1000*area*1e-5)[:,None]*v
 axial=conductance[1:,None]*(v[1:]-v[parent[1:]])
 lhs[1:]+=axial;np.add.at(lhs,parent[1:],-axial)
 rhs=np.zeros_like(v);rhs[mapping[ports[[3,4]]],np.arange(2)]=1
 error=float(np.max(abs(lhs-rhs)));check(name+':whole_field_KCL',error<1e-8,error)
 close(name+':port_full_field',v[mapping[ports]],Z[name+'_k8'][4][:,[3,4]],1e-7)
# Native trace and discrepancy checks preserve the expected failed temporal gate.
max_trace=None
if a.require_transients:
 with np.load(R/'reproduction/reference/allen_transients.npz',allow_pickle=False) as z:TR={k:z[k].copy() for k in z.files}
 trace_names=[f'transient_{name}_P{source}_dt{dt}.npz' for name in ['baseline','uniform','selective'] for source in [0,3] for dt in [.1,.05,.025]]
 check('18_native_transient_files',set(p.name for p in O.glob('transient_*.npz'))==set(trace_names))
 max_trace=0
 for filename in trace_names:
  with np.load(O/filename,allow_pickle=False) as z:t=z['time_ms'];v=z['voltage_deviation_mV'];charge=z['input_charge_nA_ms']
  stem=filename[:-4].replace('transient_','');close(filename+':time',t,TR[stem+'_time_ms'],0);e=float(np.max(abs(v-TR[stem+'_voltage_deviation_mV'])));max_trace=max(max_trace,e);check(filename+':voltage',e<1e-7,e);close(filename+':charge',charge,1,1e-10)
 for row in readcsv('temporal_convergence.csv'):
  dt=float(row['dt_ms']);name=row['candidate'];source=int(row['input_port']);coarse=dt*2
  with np.load(O/f'transient_{name}_P{source}_dt{dt}.npz',allow_pickle=False) as z:a1=z['voltage_deviation_mV']
  with np.load(O/f'transient_{name}_P{source}_dt{coarse}.npz',allow_pickle=False) as z:b1=z['voltage_deviation_mV']
  error=float(np.max(abs(a1-b1)));close(name+':P'+str(source)+':dt'+str(dt)+':temporal',error,float(row['max_common_time_difference_mV']),1e-12)
 gate=json.loads((O/'transient_gate.json').read_text());check('unchanged_temporal_failure',gate['status']=='FAIL' and gate['criterion_mV']==.0025 and gate['worst_finest_pair_mV']>.05)
result={'status':'PASS' if all(r['pass'] for r in checks) else 'FAIL','check_count':len(checks),'failures':[r for r in checks if not r['pass']],'worst_saved_matrix_relative_error':worst,'worst_saved_trace_absolute_mV':max_trace,'producer_imports':0,'transients_checked':a.require_transients,'checks':checks}
a.report.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='checks'},indent=2));assert result['status']=='PASS'
