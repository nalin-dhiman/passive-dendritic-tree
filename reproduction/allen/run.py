#!/usr/bin/env python3
"""Recompute the fixed reconstructed-arbor electrical comparison from the SWC.

The transient accuracy failure is an expected result, not a successful accuracy
certification. Exit zero means the requested results were faithfully reproduced.
"""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import csv
import hashlib
import json
import platform
try:
    import resource
except ImportError:  # Windows does not supply the Unix resource module.
    resource = None
import sys
import time
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

try:
    from .engine import (PassiveTree, axial_resistance_Mohm, build_tree,
                         equivalence_metrics, frustum_geometry, norm2,
                         read_swc, transient)
except ImportError:
    from engine import (PassiveTree, axial_resistance_Mohm, build_tree,
                        equivalence_metrics, frustum_geometry, norm2,
                        read_swc, transient)

HERE = Path(__file__).resolve().parent
REFERENCE = HERE.parent / 'reference'


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False,
        default=lambda obj: obj.item() if isinstance(obj, np.generic) else obj) + '\n')


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def geometry(source, config):
    """Select ports and the largest basal subtree using geometry only."""
    swc, parent = read_swc(source)
    n = len(swc)
    kinds = swc[:, 1].astype(int)
    children = [[] for _ in range(n)]
    distance = np.zeros(n)
    primary = np.zeros(n, int)
    edge_volume = np.zeros(n)
    for i in range(1, n):
        p = parent[i]
        children[p].append(i)
        length = np.linalg.norm(swc[i, 2:5] - swc[p, 2:5])
        distance[i] = distance[p] + length
        primary[i] = i if p == 0 else primary[p]
        start_radius = swc[i, 5] if p == 0 else swc[p, 5]
        edge_volume[i] = frustum_geometry(length, start_radius, swc[i, 5])[1]
    basal = [i for i in children[0] if kinds[i] == 3]
    basal.sort(key=lambda i: (-sum(edge_volume[primary == i]), int(swc[i, 0])))
    if len(basal) < 3:
        raise ValueError('The specified comparison requires three basal primary subtrees')
    selected = basal[0]
    mask = primary == selected
    fraction = sum(edge_volume[mask]) / sum(edge_volume)
    reduction = config['volume_reduction_fraction']
    if fraction <= reduction:
        raise ValueError('Selected subtree cannot supply the prescribed reduction')
    beta = float(np.sqrt(1 - reduction / fraction))
    scales = {'baseline': np.ones(n), 'unchanged': np.ones(n),
        'uniform': np.where(kinds == 1, 1., config['uniform_radius_factor']),
        'selective': np.where(mask, beta, 1.)}
    tips = [i for i in range(n) if kinds[i] == 4 and not children[i]]
    tip = max(tips, key=lambda i: (distance[i], -int(swc[i, 0])))
    path = []
    i = tip
    while i != 0:
        path.append(i)
        i = parent[i]
    path.reverse()
    apical = [min(path, key=lambda i: abs(distance[i] - q * distance[tip]))
              for q in (.25, .5, .9)]
    basal_ports = []
    for branch in basal[:3]:
        tips = [i for i in range(n) if primary[i] == branch and not children[i]]
        basal_ports.append(max(tips, key=lambda i: (distance[i], -int(swc[i, 0]))))
    ports = np.asarray([0] + apical + basal_ports)
    if len(set(ports)) != 7:
        raise ValueError('Seven distinct physical ports required')
    rows = [{'port': f'P{j}', 'source_node_id': int(swc[i, 0]), 'type': int(kinds[i]),
        'path_distance_um': float(distance[i]), 'x_um': float(swc[i, 2]),
        'y_um': float(swc[i, 3]), 'z_um': float(swc[i, 4])}
        for j, i in enumerate(ports)]
    details = {'source_sha256': sha256(source), 'nodes_included': n,
        'dendritic_edges': n - 1,
        'source_node_count_total': int(len(np.loadtxt(source, comments='#'))),
        'basal_primary_source_ids_ranked': [int(swc[i, 0]) for i in basal],
        'selected_basal_primary_source_id': int(swc[selected, 0]),
        'selected_fraction_of_modeled_dendritic_volume': float(fraction),
        'selective_radius_factor': beta, 'ports': rows}
    return swc, parent, ports, scales, primary, details


def analytic_fixtures():
    checks = []
    def check(name, error, tolerance):
        checks.append({'test': name, 'error': float(error), 'limit': tolerance,
                       'pass': bool(error <= tolerance)})
    for length, r0, r1 in [(20, .2, .8), (250, .5, .5), (1, .1, 2.)]:
        area, volume = frustum_geometry(length, r0, r1)
        radius = lambda x: r0 + (r1 - r0) * x / length
        v = quad(lambda x: np.pi * radius(x)**2, 0, length, epsabs=1e-11)[0]
        a = quad(lambda x: 2 * np.pi * radius(x) * np.sqrt(1 + ((r1-r0)/length)**2), 0, length)[0]
        resistance = quad(lambda x: 1. / (np.pi * radius(x)**2), 0, length, epsabs=1e-10)[0]
        check(f'frustum_volume_{length}', abs(volume/v-1), 1e-11)
        check(f'frustum_area_{length}', abs(area/a-1), 1e-11)
        check(f'frustum_axial_R_{length}', abs(axial_resistance_Mohm(length,r0,r1)/resistance-1), 1e-11)
    n = 2049
    length, radius = 250., .5
    dx = length / (n-1)
    parent = np.arange(n)-1
    conductance = np.r_[0, np.full(n-1, 1/axial_resistance_Mohm(dx,radius,radius))]
    area = np.full(n, 2*np.pi*radius*dx)
    area[[0,-1]] /= 2
    tree = PassiveTree(parent, conductance, area, area*1e-5, area*1e-6,
        np.arange(n), np.zeros((n,3)), sum(area), np.pi*radius**2*length, 0, 0, length)
    for frequency in [0,10,100]:
        computed,_,_ = tree.impedance([0,n-1],frequency)
        resistance = axial_resistance_Mohm(length,radius,radius)
        admittance = 2*np.pi*radius*length*1e-6 + 2j*np.pi*frequency/1000*(2*np.pi*radius*length*1e-5)
        gamma = np.sqrt(resistance*admittance)
        zc = np.sqrt(resistance/admittance)
        exact = zc*np.array([[1/np.tanh(gamma),1/np.sinh(gamma)],
                             [1/np.sinh(gamma),1/np.tanh(gamma)]])
        check(f'continuum_cylinder_{frequency}Hz',norm2(computed-exact)/norm2(exact),2e-6)
    radii = np.linspace(.2,.8,n)
    g = 1/axial_resistance_Mohm(dx,radii[:-1],radii[1:])
    diagonal = np.r_[g[0],g[:-1]+g[1:],g[-1]]
    matrix = diags([-g,diagonal,-g],[-1,0,1],shape=(n,n)).tocsc()
    rhs = np.zeros(n-2)
    rhs[0] = g[0]
    v = np.r_[1,spsolve(matrix[1:-1,1:-1],rhs),0]
    current = g[0]*(1-v[1])
    check('taper_DC_end_current',abs(current/(1/axial_resistance_Mohm(length,.2,.8))-1),1e-7)
    return checks


def secondary_metrics(z0, zc):
    delta = zc-z0
    rows = np.linalg.norm(delta,axis=1)/np.maximum(np.linalg.norm(z0,axis=1),1.)
    relative = norm2(np.linalg.solve(z0.T,delta.T).T)
    return np.asarray([rows[0],max(rows),relative]), rows


def analyze(matrices, frequencies, output, config):
    primary_rows, secondary_rows, all_rows, sensitivity = [], [], [], []
    margin = config['electrical_margin']
    for name in ['unchanged','uniform','selective']:
        for q, frequency in enumerate(frequencies):
            for family, evaluator, table in [('primary',equivalence_metrics,primary_rows),
                                              ('secondary',lambda a,b:secondary_metrics(a,b)[0],secondary_rows)]:
                values = {k:evaluator(matrices[('baseline',k)][q],matrices[(name,k)][q]) for k in [2,4,8]}
                for j, criterion in enumerate(['root_gain','all_port_gain','response_relative']):
                    allowance = float(abs(values[8][j]-values[4][j]))
                    two = allowance+float(abs(values[4][j]-values[2][j]))
                    raw = float(values[8][j])
                    table.append({'candidate':name,'frequency_Hz':frequency,'criterion':criterion,
                        'error':raw,'engineering_discrepancy_allowance':allowance,
                        'error_plus_allowance':raw+allowance,'threshold':margin,
                        'verdict':'PASS' if raw+allowance<=margin else 'FAIL'})
                    sensitivity.append({'normalization':family,'candidate':name,'frequency_Hz':frequency,
                        'criterion':criterion,'raw':raw,'implemented_one_increment_allowance':allowance,
                        'two_increment_allowance':two,'implemented_verdict':'PASS' if raw+allowance<=margin else 'FAIL',
                        'two_increment_verdict':'PASS' if raw+two<=margin else 'FAIL',
                        'verdict_unchanged':bool((raw+allowance<=margin)==(raw+two<=margin))})
            _,rows = secondary_metrics(matrices[('baseline',8)][q],matrices[(name,8)][q])
            all_rows.extend({'candidate':name,'frequency_Hz':frequency,'port':j,'row_relative_error':float(value)} for j,value in enumerate(rows))
    write_csv(output/'electrical_equivalence.csv',primary_rows)
    write_csv(output/'secondary_row_normalization.csv',secondary_rows)
    write_csv(output/'secondary_port_errors.csv',all_rows)
    write_csv(output/'allowance_sensitivity.csv',sensitivity)
    maxima = {}
    for family,table in [('primary',primary_rows),('secondary',secondary_rows)]:
        maxima[family] = {name:{criterion:max([r for r in table if r['candidate']==name and r['criterion']==criterion],key=lambda r:r['error_plus_allowance'])
            for criterion in ['root_gain','all_port_gain','response_relative']} for name in ['uniform','selective']}
    write_json(output/'metric_maxima.json',maxima)
    write_json(output/'allowance_sensitivity.json',{'comparisons':len(sensitivity),
        'unchanged_verdicts':sum(r['verdict_unchanged'] for r in sensitivity),
        'all_verdicts_unchanged':all(r['verdict_unchanged'] for r in sensitivity),
        'implemented_allowance':'abs(metric8-metric4)',
        'alternative_allowance':'abs(metric8-metric4)+abs(metric4-metric2)',
        'interpretation':'Empirical numerical discrepancy indicators, not rigorous bounds or anatomical uncertainty.'})
    return maxima


def harmonic_outputs(tree, ports, matrices, frequencies, output):
    times = np.linspace(0,300,1201)
    rows = []
    i = list(frequencies).index(10)
    for name in ['baseline','uniform','selective']:
        for source in [3,4]:
            voltage = matrices[(name,8)][i][:,source]*.001
            for t in times:
                wave = np.real(voltage*np.exp(2j*np.pi*10*t/1000))
                rows.append({'candidate':name,'input_port':source,'time_ms':t,'frequency_Hz':10,
                    'current_amplitude_nA':.001,**{f'P{j}_voltage_deviation_mV':float(wave[j]) for j in range(7)}})
    write_csv(output/'harmonic_10Hz_two_inputs.csv',rows)
    # Full-node values are directly solved; no interpolation from seven ports.
    fields = {}
    for name in ['baseline','uniform','selective']:
        t = tree[(name,8)]
        pp = t.original_to_mesh[ports]
        _, full, _ = t.impedance(pp,10)
        fields[name+'_phasor_per_nA'] = full[:,[3,4]]
    fields['xyz_um'] = tree[('baseline',8)].xyz
    fields['input_ports'] = np.array([3,4])
    np.savez_compressed(output/'harmonic_full_fields.npz',**fields)
    # The fixed algebraic construction uses the frequency with largest selective
    # primary distributed gain, then projects into the root-difference nullspace.
    scores = [equivalence_metrics(matrices[('baseline',8)][q],matrices[('selective',8)][q])[1] for q in range(len(frequencies))]
    q = int(np.argmax(scores))
    z0,zc = matrices[('baseline',8)][q],matrices[('selective',8)][q]
    delta = zc-z0
    _,_,vh = np.linalg.svd(delta)
    current = vh.conj().T[:,0]
    row = delta[0]
    current -= row.conj()*((row@current)/(row@row.conj()))
    current /= np.linalg.norm(current)
    np.savez_compressed(output/'root_null_witness.npz',frequency_Hz=frequencies[q],current=current,
        baseline=z0@current,candidate=zc@current,difference=delta@current)
    write_json(output/'root_null_witness.json',{'frequency_Hz':frequencies[q],
        'unit_current_norm_nA':float(np.linalg.norm(current)),
        'root_difference_mV_per_unit_current':float(abs((delta@current)[0])),
        'largest_port_difference_mV_per_unit_current':float(max(abs(delta@current))),
        'construction':'Leading right singular current projected into nullspace of root difference; Euclidean unit norm.',
        'display_current_norm_nA':.001})
    witness_rows=[]
    for t in times:
        values=np.real(.001*(delta@current)*np.exp(2j*np.pi*frequencies[q]*t/1000))
        witness_rows.append({'time_ms':t,'frequency_Hz':frequencies[q],'current_L2_norm_nA':.001,
            **{f'P{j}_difference_mV':float(values[j]) for j in range(7)}})
    write_csv(output/'root_null_harmonic_responses.csv',witness_rows)


def reproduce(output, include_transients=False, verify_reference=True):
    start, cpu = time.monotonic(), time.process_time()
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Output must be new or empty; existing results are never overwritten')
    output.mkdir(parents=True,exist_ok=True)
    config = json.loads((HERE/'config.json').read_text())
    source = HERE/'data/588402092.swc'
    source_info = json.loads((HERE/'data/SOURCE.json').read_text())
    if sha256(source)!=source_info['source_sha256']:
        raise ValueError('Original morphology hash mismatch')
    swc,parent,ports,scales,primary,geom = geometry(source,config)
    write_json(output/'geometry_and_ports.json',geom)
    write_csv(output/'physical_ports.csv',geom['ports'])
    np.savez_compressed(output/'retained_geometry.npz',swc=swc,parent=parent,ports=ports,primary=primary,
        **{'scale_'+name:scale for name,scale in scales.items()})
    fixtures = analytic_fixtures()
    write_csv(output/'analytic_fixture_checks.csv',fixtures)
    if not all(row['pass'] for row in fixtures):
        write_json(output/'numerical_gate.json',{'pass':False,'reason':'Analytic fixture failure'})
        return 1
    frequencies,meshes = config['frequencies_Hz'],config['meshes']
    matrices,trees,materials,checks = {},{},{},[]
    for name,scale in scales.items():
        for k in meshes:
            tree = build_tree(swc,parent,k,scale,Ra=config['Ra_ohm_cm'],gm=config['gm_S_cm2'],cm=config['cm_uF_cm2'])
            trees[(name,k)] = tree
            pp = tree.original_to_mesh[ports]
            materials[(name,k)] = {'candidate':name,'mesh':k,'nodes':len(tree.parent),
                'dendritic_area_um2':tree.dendritic_area,'dendritic_volume_um3':tree.dendritic_volume,
                'soma_area_um2':tree.soma_area,'soma_volume_um3':tree.soma_volume,
                'length_um':tree.total_length,'whole_modeled_area_um2':float(sum(tree.area)),
                'whole_modeled_volume_um3':tree.soma_volume+tree.dendritic_volume,
                'whole_capacitance_pF':float(sum(tree.capacitance)*1000),
                'membrane_leak_nS':float(sum(tree.membrane_conductance)*1000)}
            values=[]
            for f in frequencies:
                z,x,b = tree.impedance(pp,f)
                independent = tree.hines_solve(b,f)[pp]
                residual = tree.matrix(f)@x-b
                hermitian = (z+z.conj().T)/2
                row={'candidate':name,'mesh':k,'frequency_Hz':f,
                    'independent_relative_error':norm2(z-independent)/norm2(z),
                    'reciprocity_relative_error':norm2(z-z.T)/norm2(z),
                    'KCL_relative_residual':norm2(residual)/norm2(b),
                    'Hermitian_min_eigenvalue_Mohm':float(np.linalg.eigvalsh(hermitian)[0])}
                row['pass']=bool(max(row[key] for key in ['independent_relative_error','reciprocity_relative_error','KCL_relative_residual'])<=config['algebraic_relative_max'] and row['Hermitian_min_eigenvalue_Mohm']>0)
                checks.append(row)
                values.append(z)
            matrices[(name,k)]=np.asarray(values)
    write_csv(output/'material_integrals.csv',list(materials.values()))
    write_csv(output/'algebraic_checks.csv',checks)
    np.savez_compressed(output/'impedance_matrices.npz',frequencies_Hz=frequencies,
        **{f'{name}_k{k}':value for (name,k),value in matrices.items()})
    convergence,invariants=[],[]
    for name in scales:
        base=materials[(name,1)]
        for k in meshes[1:]:
            for key in ['dendritic_area_um2','dendritic_volume_um3','whole_capacitance_pF']:
                error=abs(materials[(name,k)][key]/base[key]-1)
                invariants.append({'candidate':name,'mesh':k,'quantity':key,
                    'relative_difference':error,'pass':bool(error<=config['geometry_relative_max'])})
        previous=None
        for coarse,fine in zip(meshes[:-1],meshes[1:]):
            errors=np.array([norm2(a-b)/norm2(b) for a,b in zip(matrices[(name,coarse)],matrices[(name,fine)])])
            for q,f in enumerate(frequencies):
                ratio=None if previous is None else float(errors[q]/previous[q])
                contraction=ratio is None or previous[q]<=config['contraction_floor'] or ratio<=config['contraction_max']
                convergence.append({'candidate':name,'coarse':coarse,'fine':fine,'frequency_Hz':f,
                    'relative_operator_difference':float(errors[q]),'contraction_ratio':ratio,
                    'contraction_pass':bool(contraction),'last_pair_gate_pass':bool(errors[q]<=config['spatial_max'] and contraction) if fine==8 else None})
            previous=errors
    write_csv(output/'spatial_convergence.csv',convergence)
    write_csv(output/'mesh_invariants.csv',invariants)
    identity=max(norm2(matrices[('baseline',k)][q]-matrices[('unchanged',k)][q]) for k in meshes for q in range(len(frequencies)))
    required=[row for row in convergence if row['fine']==8 and row['candidate']!='unchanged']
    gate={'pass':all(row['pass'] for row in checks+invariants) and all(row['last_pair_gate_pass'] for row in required) and identity==0,
        'fixture_pass':True,'all_algebraic_checks_pass':all(row['pass'] for row in checks),
        'all_geometry_invariants_pass':all(row['pass'] for row in invariants),
        'unchanged_absolute_difference':identity,'last_pair_spatial_gate_pass':all(row['last_pair_gate_pass'] for row in required),
        'last_pair_worst_relative_impedance_difference':max(row['relative_operator_difference'] for row in required),
        'last_pair_worst_contraction':max(row['contraction_ratio'] for row in required)}
    write_json(output/'numerical_gate.json',gate)
    if not gate['pass']:
        write_json(output/'reproduction_verification.json',{'status':'FAIL','reason':'Harmonic numerical gate failed; dependent work stopped'})
        return 1
    maxima=analyze(matrices,frequencies,output,config)
    harmonic_outputs(trees,ports,matrices,frequencies,output)
    temporal=[]
    transient_runs=0
    if include_transients:
        for name in ['baseline','uniform','selective']:
            tree=trees[(name,8)]
            pp=tree.original_to_mesh[ports]
            for source_port in config['transient']['input_ports']:
                previous=previous_error=None
                for dt in config['transient']['dt_ms']:
                    times,voltage,snapshots,charge=transient(tree,pp,pp[source_port],dt=dt,
                        horizon=config['transient']['horizon_ms'],amplitude=config['transient']['amplitude_nA'],
                        on=config['transient']['on_ms'],off=config['transient']['off_ms'])
                    np.savez_compressed(output/f'transient_{name}_P{source_port}_dt{dt}.npz',time_ms=times,
                        voltage_deviation_mV=voltage,input_port=source_port,input_charge_nA_ms=charge,
                        **{'snapshot_'+key:value for key,value in snapshots.items()})
                    transient_runs += 1
                    if previous is not None:
                        error=float(np.max(abs(voltage-previous)))
                        ratio=None if previous_error is None else error/previous_error
                        temporal.append({'candidate':name,'input_port':source_port,'dt_ms':dt,
                            'max_common_time_difference_mV':error,'contraction_ratio':ratio,
                            'pass':bool((ratio is None or ratio<=config['transient']['contraction_max']) and
                                (dt!=.025 or error<=config['transient']['max_final_voltage_difference_mV']))})
                        previous_error=error
                    previous=voltage
        write_csv(output/'temporal_convergence.csv',temporal)
        transient_gate={'status':'PASS' if all(row['pass'] for row in temporal) else 'FAIL',
            'pass':all(row['pass'] for row in temporal),'runs_completed':transient_runs,
            'worst_finest_pair_mV':max(row['max_common_time_difference_mV'] for row in temporal if row['dt_ms']==.025),
            'criterion_mV':config['transient']['max_final_voltage_difference_mV']}
    else:
        transient_gate={'status':'NOT_RUN','runs_completed':0,'reason':'Use --include-transients to reproduce the full step comparison.'}
    write_json(output/'transient_gate.json',transient_gate)
    verification=verify_outputs(output,include_transients) if verify_reference else {'status':'NOT_CHECKED'}
    write_json(output/'reproduction_verification.json',verification)
    summary={'reproduction_status':verification['status'],'harmonic_scientific_gate':'PASS',
        'transient_scientific_gate':transient_gate['status'],'transient_failure_is_preserved':transient_gate['status']=='FAIL',
        'harmonic_conditions':len(checks),'transient_runs':transient_gate['runs_completed'],
        'source_sha256':sha256(source),'wall_seconds':time.monotonic()-start,'cpu_seconds':time.process_time()-cpu,
        'max_rss_kB':None if resource is None else resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024 if sys.platform=='darwin' else 1),
        'memory_measurement':'Peak resident set in KiB; unavailable when the platform lacks resource.getrusage',
        'python':sys.version,
        'platform':platform.platform(),'parameters':config,
        'scope':'Specified retained passive geometry, seven collocated ports, sampled frequencies; expected step-accuracy failure remains FAIL.'}
    write_json(output/'summary.json',summary)
    print(json.dumps(summary,indent=2))
    return 0 if verification['status']=='PASS' else 1


def verify_outputs(output, include_transients=False):
    output=Path(output)
    config=json.loads((HERE/'config.json').read_text())
    expected=json.loads((REFERENCE/'allen_expected.json').read_text())
    current=np.load(output/'impedance_matrices.npz')
    reference=np.load(REFERENCE/'allen_impedance.npz')
    if set(current.files)!=set(reference.files):
        raise ValueError('Harmonic matrix inventory changed')
    errors=[]
    for name in reference.files:
        if name=='frequencies_Hz':
            if not np.array_equal(current[name],reference[name]):raise ValueError('Frequency grid changed')
            continue
        if current[name].shape!=reference[name].shape:
            raise ValueError('Harmonic matrix shape changed: '+name)
        for z,ref in zip(current[name],reference[name]):errors.append(norm2(z-ref)/norm2(ref))
    geom=json.loads((output/'geometry_and_ports.json').read_text())
    exact_keys=['source_sha256','nodes_included','dendritic_edges','source_node_count_total',
                'basal_primary_source_ids_ranked','selected_basal_primary_source_id']
    geometry_matches=all(geom[key]==expected['geometry'][key] for key in exact_keys)
    geometry_matches=geometry_matches and [row['source_node_id'] for row in geom['ports']]==[row['source_node_id'] for row in expected['geometry']['ports']]
    geometry_error=max(abs(geom[key]-expected['geometry'][key]) for key in ['selected_fraction_of_modeled_dendritic_volume','selective_radius_factor'])
    maxima=json.loads((output/'metric_maxima.json').read_text())
    max_metric_error=0.
    verdicts_match=True
    for family,old_family in [('primary','primary_maxima'),('secondary','secondary_maxima')]:
        for candidate in ['uniform','selective']:
            for new_name,old_name in [('root_gain','root_gain' if family=='primary' else 'root_row_relative_gain'),
                                      ('all_port_gain','all_port_gain' if family=='primary' else 'max_port_row_relative_gain'),
                                      ('response_relative','response_relative')]:
                a=maxima[family][candidate][new_name]
                b=expected[old_family][candidate][old_name]
                max_metric_error=max(max_metric_error,abs(a['error_plus_allowance']-float(b['error_plus_allowance'])))
                verdicts_match=verdicts_match and a['verdict']==b['verdict']
    trace_error=0.
    charge_error=0.
    observed_temporal_failure=False
    if include_transients:
        refs=np.load(REFERENCE/'allen_transients.npz')
        expected_names={name.removesuffix('_time_ms') for name in refs.files if name.endswith('_time_ms')}
        actual_names={file.stem.removeprefix('transient_') for file in output.glob('transient_*.npz')}
        if actual_names!=expected_names or len(expected_names)!=18:
            raise ValueError('Transient inventory incomplete: expected all 18 fixed executions')
        for file in sorted(output.glob('transient_*.npz')):
            current_trace=np.load(file)
            key=file.stem.replace('transient_','')
            if current_trace['voltage_deviation_mV'].shape!=refs[key+'_voltage_deviation_mV'].shape:
                raise ValueError('Transient voltage shape changed: '+key)
            if not np.array_equal(current_trace['time_ms'],refs[key+'_time_ms']):raise ValueError('Native transient output times changed')
            trace_error=max(trace_error,float(np.max(abs(current_trace['voltage_deviation_mV']-refs[key+'_voltage_deviation_mV']))))
            charge_error=max(charge_error,abs(float(current_trace['input_charge_nA_ms'])-float(refs[key+'_input_charge_nA_ms'])))
        gate=json.loads((output/'transient_gate.json').read_text())
        observed_temporal_failure=gate['status']=='FAIL' and gate['runs_completed']==18 and abs(gate['worst_finest_pair_mV']-expected['transient_gate']['worst_finest_pair_mV'])<config['reference_trace_max_absolute_mV']
    gate=json.loads((output/'numerical_gate.json').read_text())
    passed=geometry_matches and geometry_error<1e-12 and gate['pass'] and max(errors)<=config['reference_max_relative_matrix_error'] and max_metric_error<1e-8 and verdicts_match and (not include_transients or (trace_error<=config['reference_trace_max_absolute_mV'] and charge_error<=1e-12 and observed_temporal_failure))
    return {'status':'PASS' if passed else 'FAIL','matrix_comparisons':len(errors),
        'largest_relative_saved_matrix_difference':max(errors),'geometry_identity_match':bool(geometry_matches),
        'largest_geometry_selection_difference':geometry_error,'largest_metric_upper_difference':max_metric_error,
        'electrical_verdicts_match':bool(verdicts_match),'transients_included':include_transients,
        'largest_saved_voltage_difference_mV':trace_error,'largest_charge_difference_nA_ms':charge_error,
        'expected_transient_failure_reproduced':observed_temporal_failure,
        'interpretation':'Reproduction PASS means numerical agreement with saved results, including the failed transient accuracy criterion.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--include-transients',action='store_true',help='Execute all 18 steps and retain their expected numerical accuracy FAIL.')
    parser.add_argument('--verify-only',action='store_true',help='Check an existing output against numerical reference fixtures.')
    args=parser.parse_args()
    if args.verify_only:
        result=verify_outputs(args.output,args.include_transients)
        print(json.dumps(result,indent=2))
        return 0 if result['status']=='PASS' else 1
    return reproduce(args.output,args.include_transients)


if __name__=='__main__':
    raise SystemExit(main())
