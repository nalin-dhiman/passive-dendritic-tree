#!/usr/bin/env python3
"""Reproduce the fixed passive Y-cable comparison from a clean checkout.

Usage: python reproduction/y_cable/run.py --output NEW_DIRECTORY
Exit zero means numerical checks and the saved-data comparisons reproduced;
individual electrical-preservation criteria may correctly remain FAIL.
"""
import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"
import argparse, csv, hashlib, json, math, sys, time, traceback
from pathlib import Path
import numpy as np
import engine as E
import independent_reference as I
import metrics as M

HERE = Path(__file__).resolve().parent
REFERENCE = HERE.parent / "reference"


def native(value):
    if isinstance(value, np.ndarray): return native(value.tolist())
    if isinstance(value, np.generic): return native(value.item())
    if isinstance(value, dict): return {str(k): native(v) for k,v in value.items()}
    if isinstance(value, (tuple,list)): return [native(v) for v in value]
    if value is None or isinstance(value,(str,bool,int)): return value
    if isinstance(value,float) and math.isfinite(value): return value
    raise TypeError("Unsupported or nonfinite JSON value: " + repr(value))


def write_json(path, payload):
    with Path(path).open("x") as out:
        out.write(json.dumps(native(payload), indent=2, allow_nan=False) + "\n")


def save_npz(path, **arrays):
    with Path(path).open("xb") as out: np.savez_compressed(out, **arrays)

def condition_payload(geometry,setting,checks,details,material):
    """The production writer boundary, directly exercised by software contracts."""
    return {'label':'SYNTHETIC_EXECUTED','geometry':geometry,'setting':setting,
        'status':'PASS' if all(checks.values()) else 'FAIL','checks':checks,'details':details,'material':material,
        'scope':'Ideal passive cylinders; seven finite ports and sampled frequencies. No biological/ATP/active claim.'}


def append_ledger(root,payload):
    with (root/'processed_data'/'execution_ledger.jsonl').open('a') as out:out.write(json.dumps(native(payload),allow_nan=False)+'\n')


def acquire(root,geometry,setting,protocol):
    key=geometry['id']+'_'+setting['id'];radii=geometry['radii'];kw={k:setting[k] for k in ('ra','gm','cm')}
    frequencies=np.arange(protocol['scope']['frequency_Hz']['start'],protocol['scope']['frequency_Hz']['end']+1,protocol['scope']['frequency_Hz']['step'],dtype=float)
    ports=tuple(protocol['scope']['ports']);c=protocol['criteria'];start=time.monotonic()
    Z,J,K=E.continuum_state(radii,frequencies,inputs=ports,outputs=ports,**kw)
    independent=I.solve(frequencies,radii_um=radii,**kw)
    reference=independent['voltage'];reference_J=np.stack((independent['J_start'],-independent['J_end']),axis=2)
    finite={n:E.fvm_transfer(radii,frequencies,inputs=ports,outputs=ports,nseg=n,**kw) for n in c['fvm_meshes']}
    errors={str(n):M.relative_row_error(value,Z) for n,value in finite.items()}
    materials=E.material(radii,cm=kw['cm'],gm=kw['gm'])
    # Independent explicit cylinder formulas, rather than engine material reuse.
    area=float(800*np.pi*sum(radii));volume=float(400*np.pi*sum(float(r)**2 for r in radii))
    expected={'area_um2':area,'volume_um3':volume,'capacitance_pF':area*kw['cm']*.01,'leak_conductance_nS':area*kw['gm']*10.}
    identities=[]
    for n in c['fvm_meshes']:
        g=E.fvm_geometry(radii,nseg=n,**kw)
        observed={'area_um2':sum(g['area_um2']),'volume_um3':sum(g['volume_um3']),
            'capacitance_pF':sum(g['C_nF'])*1000,'leak_conductance_nS':sum(g['leak_uS'])*1000}
        identities.append({'mesh':n,**{k:abs(observed[k]/expected[k]-1) for k in expected}})
    norms=np.linalg.svd(Z,compute_uv=False)[:,0]
    reciprocity=M.relative_row_error(Z,Z.swapaxes(1,2))
    passive=np.linalg.eigvalsh((Z+Z.conj().swapaxes(1,2))/2)[:,0]/norms
    conditions={'continuum':float(max(np.linalg.cond(Z))),
        'reference':float(max(np.linalg.cond(reference))),**{'FVM'+str(n):float(max(np.linalg.cond(value))) for n,value in finite.items()}}
    details={'continuum_reference_relative':M.relative_row_error(Z,reference),
        'edge_current_reference_absolute':float(np.max(abs(J-reference_J))),
        'production_KCL_absolute':float(np.max(abs(K))),
        'reference_KCL_absolute':float(np.max(abs(independent['KCL_residual']))),
        'continuity_relative':float(np.max(abs(independent['continuity_residual_mV']))/max(float(np.max(abs(reference))),1.)),
        'reciprocity_relative':reciprocity,'passivity_min_relative_eigenvalue':float(min(passive)),
        'condition_numbers':conditions,'FVM_relative_row_errors':errors,
        'FVM_error_ratios':[errors[str(b)]/errors[str(a)] if errors[str(a)] else (0. if errors[str(b)]==0 else float('inf')) for a,b in zip(c['fvm_meshes'][:-1],c['fvm_meshes'][1:])],
        'geometry_identities':identities,'independent_material_relative':max(abs(materials[k]/v-1) for k,v in expected.items()),
        'wall_seconds':time.monotonic()-start}
    checks={'finite':all(np.all(np.isfinite(v)) for v in (Z,J,reference,reference_J,*finite.values())),
        'continuum_reference':details['continuum_reference_relative']<=c['continuum_reference_relative_max'],
        'reference_edge_currents':details['edge_current_reference_absolute']<=c['current_balance_max'],
        'production_KCL':details['production_KCL_absolute']<=c['current_balance_max'],
        'reference_KCL':details['reference_KCL_absolute']<=c['current_balance_max'],
        'continuity':details['continuity_relative']<=c['continuity_relative_max'],
        'reciprocity':reciprocity<=c['reciprocity_relative_max'],
        'passivity':min(passive)>=c['passivity_min_relative_eigenvalue'],
        'conditioning':max(conditions.values())<=c['condition_number_max'],
        'fine_FVM':errors[str(c['fvm_meshes'][-1])]<=c['fvm_fine_relative_row_error_max'],
        'FVM_contraction':max(details['FVM_error_ratios'])<=c['fvm_max_error_ratio_max'],
        'geometry':details['independent_material_relative']<=c['geometry_relative_error_max'] and all(v<=c['geometry_relative_error_max'] for row in identities for k,v in row.items() if k!='mesh')}
    # Save primary arrays and verdict metadata BEFORE enforcing any stop.
    save_npz(root/'raw_data'/('harmonic_'+key+'.npz'),frequency_Hz=frequencies,radii_um=np.asarray(radii),Z=Z,
        edge_currents=J,reference_Z=reference,reference_currents=reference_J,
        **{'FVM'+str(n):value for n,value in finite.items()},production_KCL_residual=K,
        reference_KCL_residual=independent['KCL_residual'],reference_continuity_residual_mV=independent['continuity_residual_mV'],
        namedports=np.array(ports))
    payload=condition_payload(geometry['id'],setting['id'],checks,details,materials)
    write_json(root/'processed_data'/('harmonic_'+key+'.json'),payload)
    append_ledger(root,{'kind':'condition','geometry':geometry['id'],'setting':setting['id'],'status':payload['status']})
    if payload['status']!='PASS':raise RuntimeError('Mandatory numerical FAIL: '+key)
    return {'Z':Z,'fine':finite[c['fvm_meshes'][-1]],'material':materials,'frequency':frequencies}


def compare(root,geometry,setting,baseline,candidate,protocol):
    key=geometry['id']+'_'+setting['id'];arrays=M.comparison_arrays(baseline['Z'],candidate['Z'],baseline['fine'],candidate['fine'])
    reduction=1-candidate['material']['volume_um3']/baseline['material']['volume_um3']
    payload=M.comparison_payload(geometry['id'],setting['id'],arrays,reduction,protocol)
    payload['absolute_voltage_difference_max_mV_at_current_budget']=float(np.max(arrays['absolute_row_gain_MOhm'])*protocol['witness']['current_L2_nA'])
    save_npz(root/'raw_data'/('comparison_'+key+'.npz'),frequency_Hz=baseline['frequency'],baseline_Z=baseline['Z'],candidate_Z=candidate['Z'],**arrays)
    write_json(root/'processed_data'/('comparison_'+key+'.json'),payload)
    append_ledger(root,{'kind':'comparison','geometry':geometry['id'],'setting':setting['id'],'status':payload['numerical_status'],
        'GAIN':payload['GAIN'],'RESPONSE_RELATIVE':payload['RESPONSE_RELATIVE']})
    if payload['numerical_status']!='PASS':raise RuntimeError('Mandatory allocation FAIL: '+key)
    if geometry['id']=='unchanged' and protocol['criteria']['identity_exact_required']:
        if not np.array_equal(baseline['Z'],candidate['Z']) or baseline['material']!=candidate['material']:
            raise RuntimeError('Identity control FAIL: '+key)
    if geometry['id']=='disruptive' and protocol['criteria']['disruptive_gain_fail_required'] and payload['GAIN']!='FAIL':
        raise RuntimeError('Disruptive control FAIL: '+key)
    witnesses=[]
    for kind in ('response_relative','root_null'):
        arrays_w,metadata=M.witness_data(baseline['Z'],candidate['Z'],baseline['frequency'],protocol,kind)
        path='raw_data/witness_'+key+'_'+kind+'.npz';save_npz(root/path,**arrays_w)
        metadata.update(geometry=geometry['id'],setting=setting['id'],evidence=path)
        write_json(root/'processed_data'/('witness_'+key+'_'+kind+'.json'),metadata);witnesses.append(metadata)
        if metadata['status']!='PASS':raise RuntimeError('Witness invariant FAIL: '+key+' '+kind)
    return payload,witnesses



def _numeric_close(observed, expected, label, tolerance=1e-9):
    if isinstance(expected, bool) or isinstance(expected,str) or expected is None:
        if observed != expected: raise AssertionError(f"Reference mismatch: {label}: {observed!r} != {expected!r}")
    elif isinstance(expected,(int,float)):
        if not np.isclose(float(observed),float(expected),atol=tolerance,rtol=tolerance):
            raise AssertionError(f"Reference mismatch: {label}: {observed} != {expected}")
    elif isinstance(expected,list):
        if len(observed)!=len(expected): raise AssertionError(f"Reference shape mismatch: {label}")
        for i,(a,b) in enumerate(zip(observed,expected)): _numeric_close(a,b,f"{label}[{i}]",tolerance)
    elif isinstance(expected,dict):
        for key,value in expected.items(): _numeric_close(observed[key],value,f"{label}.{key}",tolerance)


def verify_condition(key, acquired, payload, output, reference_z, expected):
    z0=reference_z['Z_'+key]
    error=float(np.max(abs(acquired['Z']-z0))/max(float(np.max(abs(z0))),1.))
    if error>1e-10: raise AssertionError(f"Saved impedance mismatch: {key}: {error}")
    actual=json.loads((output/'processed_data'/('harmonic_'+key+'.json')).read_text())
    _numeric_close(actual['material'],expected['material'],key+'.material')
    _numeric_close(actual['checks'],expected['numerical_checks'],key+'.checks')
    _numeric_close(actual['details']['FVM_relative_row_errors'],expected['fvm_relative_row_errors'],key+'.fvm',2e-8)
    # Recorded scientific verdicts and numerical values must both reproduce.
    _numeric_close(payload,expected['comparison'],key+'.comparison',2e-8)
    return error


def run(output):
    output=Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Refusing to overwrite a nonempty output directory: "+str(output))
    output.mkdir(parents=True,exist_ok=True)
    (output/'raw_data').mkdir();(output/'processed_data').mkdir()
    protocol=json.loads((HERE/'config.json').read_text())
    reference_info=json.loads((HERE/'source_provenance.json').read_text())
    for filename,sha in reference_info['reference_data_sha256'].items():
        if hashlib.sha256((REFERENCE/filename).read_bytes()).hexdigest()!=sha:
            raise ValueError('Saved reference checksum mismatch: '+filename)
    write_json(output/'config.json',protocol)
    expected=json.loads((REFERENCE/'y_expected.json').read_text())['conditions']
    rows=[];witnesses=[];baselines={};parity=[];completed=0;started=time.monotonic()
    try:
        with np.load(REFERENCE/'y_impedance_reference.npz') as reference_z:
            for geometry in protocol['geometries']:
                for setting in protocol['scope']['settings']:
                    key=geometry['id']+'_'+setting['id']
                    acquired=acquire(output,geometry,setting,protocol);completed+=1
                    if geometry['id']=='baseline':baselines[setting['id']]=acquired
                    row,records=compare(output,geometry,setting,baselines[setting['id']],acquired,protocol)
                    rows.append(row);witnesses.extend(records)
                    parity.append(verify_condition(key,acquired,row,output,reference_z,expected[key]))
        fields=['geometry','setting','volume_reduction','material_PASS','numerical_status','gain_root_raw','gain_ports_raw','response_relative_raw','gain_allocation_max','response_relative_allocation_max','gain_root_upper','gain_ports_upper','response_relative_upper','root_GAIN','GAIN','RESPONSE_RELATIVE','absolute_voltage_difference_max_mV_at_current_budget']
        with (output/'comparison.csv').open('x',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows({k:r[k] for k in fields} for r in rows)
        summary={}
        for geometry in protocol['geometries']:
            selected=[r for r in rows if r['geometry']==geometry['id']]
            summary[geometry['id']]={
                'volume_reduction':selected[0]['volume_reduction'],
                'root_gain_verdict':'PASS' if all(r['root_GAIN']=='PASS' for r in selected) else 'FAIL',
                'all_port_gain_verdict':'PASS' if all(r['GAIN']=='PASS' for r in selected) else 'FAIL',
                'response_relative_verdict':'PASS' if all(r['RESPONSE_RELATIVE']=='PASS' for r in selected) else 'FAIL',
                'root_gain_maximum_with_allowance':max(r['gain_root_upper'] for r in selected),
                'all_port_gain_maximum_with_allowance':max(r['gain_ports_upper'] for r in selected),
                'response_relative_maximum_with_allowance':max(r['response_relative_upper'] for r in selected)}
        write_json(output/'witnesses.json',{'records':witnesses})
        write_json(output/'summary.json',summary)
        result={'reproduction_status':'PASS','numerical_checks':'PASS','saved_data_comparison':'PASS','conditions':completed,
            'frequencies_per_condition':101,'reference_impedance_max_normalized_difference':max(parity),
            'witnesses':len(witnesses),'elapsed_seconds':time.monotonic()-started,
            'scientific_scope':'Passive seven-port sampled harmonic responses. Scientific criterion FAIL is preserved and is not a software/reproduction failure.'}
        write_json(output/'reproduction_status.json',result)
        print(json.dumps(result,indent=2));return 0
    except Exception as error:
        write_json(output/'reproduction_status.json',{'reproduction_status':'FAIL','reason':str(error),'completed_conditions':completed,
            'elapsed_seconds':time.monotonic()-started,'traceback':traceback.format_exc()})
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    return run(args.output)

if __name__=='__main__':raise SystemExit(main())
