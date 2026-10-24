"""Finite-port passive comparison; no import launches a scientific execution."""
import numpy as np


def row_norm(value):
    return np.linalg.norm(np.asarray(value), axis=-1)


def relative_row_error(value, reference):
    """Maximum row-2 discrepancy with the frozen 1 MOhm engineering floor."""
    return float(np.max(row_norm(np.asarray(value)-reference)/np.maximum(row_norm(reference),1.)))


def response_relative(delta, baseline):
    """R = delta @ inv(baseline), evaluated through a linear solve."""
    return np.linalg.solve(np.swapaxes(baseline,-1,-2),np.swapaxes(delta,-1,-2)).swapaxes(-1,-2)


def comparison_arrays(baseline, candidate, baseline_fine, candidate_fine):
    baseline,candidate,baseline_fine,candidate_fine=map(np.asarray,(baseline,candidate,baseline_fine,candidate_fine))
    if baseline.ndim!=3 or baseline.shape[1:]!=(7,7) or any(x.shape!=baseline.shape for x in (candidate,baseline_fine,candidate_fine)):
        raise ValueError('Four matching [frequency,7,7] matrices required')
    if not all(np.all(np.isfinite(x)) for x in (baseline,candidate,baseline_fine,candidate_fine)):
        raise ValueError('Finite comparison matrices required')
    delta=candidate-baseline
    scale=np.maximum(row_norm(baseline),1.)
    A=delta/scale[:,:,None]
    R=response_relative(delta,baseline)
    Rfine=response_relative(candidate_fine-baseline_fine,baseline_fine)
    gain_allocation=4*np.max((row_norm(baseline-baseline_fine)+row_norm(candidate-candidate_fine))/scale,axis=1)
    response_allocation=4*np.linalg.svd(R-Rfine,compute_uv=False)[:,0]
    return dict(b_fj=scale,A=A,R=R,
        gain_root_by_frequency=row_norm(A)[:,0],
        gain_ports_by_frequency=np.max(row_norm(A),axis=1),
        response_relative_by_frequency=np.linalg.svd(R,compute_uv=False)[:,0],
        gain_allocation_by_frequency=gain_allocation,
        response_relative_allocation_by_frequency=response_allocation,
        RMS_operator_by_frequency=np.linalg.svd(A,compute_uv=False)[:,0]/np.sqrt(7),
        absolute_row_gain_MOhm=row_norm(delta),
        per_source_column_normalized_error=np.abs(A))


def comparison_payload(geometry, setting, arrays, volume_reduction, protocol):
    c=protocol['criteria'];g=arrays['gain_allocation_by_frequency'];r=arrays['response_relative_allocation_by_frequency']
    root=float(np.max(arrays['gain_root_by_frequency']+g))
    ports=float(np.max(arrays['gain_ports_by_frequency']+g))
    response=float(np.max(arrays['response_relative_by_frequency']+r))
    numerical={'gain_allocation':np.max(g)<=c['allocation_fraction_of_margin_max']*c['gain_margin'],
               'response_relative_allocation':np.max(r)<=c['allocation_fraction_of_margin_max']*c['response_relative_margin']}
    return dict(label='SYNTHETIC_EXECUTED',geometry=geometry,setting=setting,
        numerical_status='PASS' if all(numerical.values()) else 'FAIL',numerical_checks=numerical,
        volume_reduction=float(volume_reduction),material_PASS=volume_reduction>=c['volume_reduction_min'],
        gain_root_raw=float(np.max(arrays['gain_root_by_frequency'])),
        gain_ports_raw=float(np.max(arrays['gain_ports_by_frequency'])),
        response_relative_raw=float(np.max(arrays['response_relative_by_frequency'])),
        gain_allocation_max=float(np.max(g)),response_relative_allocation_max=float(np.max(r)),
        gain_root_upper=root,gain_ports_upper=ports,response_relative_upper=response,
        root_GAIN='PASS' if root<=c['gain_margin'] else 'FAIL',
        GAIN='PASS' if ports<=c['gain_margin'] else 'FAIL',
        RESPONSE_RELATIVE='PASS' if response<=c['response_relative_margin'] else 'FAIL',
        gain_max_frequency_index=int(np.argmax(arrays['gain_ports_by_frequency']+g)),
        response_max_frequency_index=int(np.argmax(arrays['response_relative_by_frequency']+r)),
        scope='Finite sampled ports/frequencies; empirical discrepancy allocations, not rigorous continuum bounds.')


def canonical_current(vector, amplitude):
    x=np.asarray(vector,complex).copy();norm=np.linalg.norm(x)
    if norm==0:x=np.eye(7,dtype=complex)[:,0];norm=1.
    x*=amplitude/norm
    pivot=int(np.argmax(abs(x)))
    x*=np.exp(-1j*np.angle(x[pivot]))
    return x


def witness_data(baseline,candidate,frequency_Hz,protocol,kind):
    """Frozen algebraic witnesses; simultaneous signed electrode currents."""
    baseline,candidate=np.asarray(baseline),np.asarray(candidate)
    delta=candidate-baseline;frequency_Hz=np.asarray(frequency_Hz)
    amp=float(protocol['witness']['current_L2_nA'])
    if kind=='response_relative':
        R=response_relative(delta,baseline)
        norm=np.linalg.svd(R,compute_uv=False)[:,0]
        k=int(np.argmax(norm));_,s,Vh=np.linalg.svd(R[k])
        current=canonical_current(np.eye(7)[:,0] if np.max(abs(delta))==0 else np.linalg.solve(baseline[k],Vh.conj().T[:,0]),amp)
        selected_row=-1;target=float(s[0])
    elif kind=='root_null':
        choices=[];projectors=[]
        for a,d in zip(delta[:,0,:],delta):
            square=float(np.vdot(a,a).real)
            P=np.eye(7,dtype=complex) if square==0 else np.eye(7)-np.outer(a.conj(),a)/square
            rows=d[1:]@P
            choices.append(row_norm(rows));projectors.append(P)
        flat=int(np.argmax(np.asarray(choices)));k,j=np.unravel_index(flat,np.asarray(choices).shape)
        selected_row=int(j+1);row=delta[k,selected_row]@projectors[k]
        current=canonical_current(row.conj(),amp);target=float(np.linalg.norm(row))
    else:raise ValueError('Unknown witness kind')
    freq=float(frequency_Hz[k]);base=baseline[k]@current;changed=candidate[k]@current;difference=changed-base
    samples=int(protocol['witness']['cycles'])*(int(protocol['witness']['period_samples'])-1)+1
    duration=(1000/freq)*protocol['witness']['cycles'] if freq else 2000.
    times=np.linspace(0,duration,samples)
    phase=np.exp(2j*np.pi*freq*times/1000)
    values=dict(frequency_index=np.array(k),frequency_Hz=np.array(freq),selected_output_index=np.array(selected_row),
        input_phasor_nA=current,Vbaseline_phasor_mV=base,Vcandidate_phasor_mV=changed,Vdelta_phasor_mV=difference,
        times_ms=times,current_nA=np.real(phase[:,None]*current),Vbaseline_mV=np.real(phase[:,None]*base),
        Vcandidate_mV=np.real(phase[:,None]*changed),Vdelta_mV=np.real(phase[:,None]*difference))
    ratio=float(np.linalg.norm(difference)/np.linalg.norm(base)) if np.linalg.norm(base) else 0.
    c=protocol['criteria']
    root_tolerance=c['current_balance_max']*max(1.,amp*np.linalg.norm(delta[k,0]))
    checks={'current_budget':abs(np.linalg.norm(current)-amp)<=c['geometry_relative_error_max']*amp,
            'DC_real':bool(freq!=0 or np.max(abs(current.imag))<=c['current_balance_max'])}
    if kind=='response_relative':
        checks['relative_witness']=abs(ratio-target)<=c['continuum_reference_relative_max']
    else:
        checks['root_difference_null']=abs(difference[0])<=root_tolerance
        checks['omitted_witness']=abs(abs(difference[selected_row])-amp*target)<=c['continuum_reference_relative_max']*max(1.,amp*target)
    metadata=dict(label='SYNTHETIC_EXECUTED',kind=kind,frequency_index=k,frequency_Hz=freq,selected_output_index=selected_row,
        target_gain_or_ratio=target,actual_response_ratio=ratio,current_L2_nA=float(np.linalg.norm(current)),
        root_delta_absolute_mV=float(abs(difference[0])),max_delta_absolute_mV=float(np.max(abs(difference))),
        checks=checks,status='PASS' if all(checks.values()) else 'FAIL',
        time_scope='Steady-state phasor waveform; no startup/transient integration. DC display spans2000ms.',
        degenerate_zero_operator=bool(np.max(abs(delta))==0),
        stimulus_scope='Finite seven-port signed/phase-shifted electrode-current engineering witness, not biological synapses.')
    return values,metadata
