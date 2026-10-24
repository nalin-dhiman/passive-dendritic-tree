#!/usr/bin/env python3
"""Export a retained Allen morphology and saved harmonic comparator matrices.

Usage: python scripts/export_geometry.py --source-dir PATH --out docs/data
The source directory must contain retained_geometry.npz, geometry_and_port_lock.json,
impedance_matrices.npz and source_input/588402092.swc. No source paths are published.
"""
import argparse, hashlib, json
from pathlib import Path
import numpy as np

def export(source, out):
    out.mkdir(parents=True, exist_ok=True)
    a=np.load(source/'retained_geometry.npz'); lock=json.loads((source/'geometry_and_port_lock.json').read_text())
    swc=a['swc']; parent=a['parent']; ports=a['ports']; primary=a['primary']
    source_hash=hashlib.sha256((source/'source_input/588402092.swc').read_bytes()).hexdigest()
    if source_hash!=lock['source_sha256']: raise ValueError('SWC source hash mismatch')
    selected=np.where(swc[:,0].astype(int)==lock['selected_basal_primary_source_id'])[0][0]
    labels=['Soma','Proximal apical','Middle apical','Distal apical','Selected basal tip','Second basal tip','Third basal tip']
    model={'schemaVersion':1,'id':'allen','name':'Mouse cortical dendritic arbor','specimenId':588402092,
      'source':{'provider':'Allen Institute for Brain Science','url':'https://celltypes.brain-map.org/experiment/morphology/588402092',
      'reconstructionId':675965554,'sourceFileId':675965556,'swcUrl':'https://api.brain-map.org/api/v2/well_known_file_download/675965556',
      'termsUrl':'https://alleninstitute.org/legal/terms-use/','citationPolicyUrl':'https://alleninstitute.org/legal/citation-policy/','use':'Noncommercial research and education; Allen terms apply to morphology data',
      'swcSha256':source_hash,'coordinates':'micrometres','radius':'micrometres','excludedSwcTypes':[2],
      'includedSwcTypes':[1,3,4],'attribution':'Allen Institute for Brain Science, Allen Cell Types Database'},
      'somaMode':'fixed sphere','somaAttachments':'cylinders at child radius','internalEdges':'linear-radius frusta',
      'selectiveFactor':lock['selective_radius_factor'],'selectedRootSourceId':lock['selected_basal_primary_source_id'],
      'nodes':[{'id':int(row[0]),'type':int(row[1]),'x':float(row[2]),'y':float(row[3]),'z':float(row[4]),'r':float(row[5]),'parent':int(parent[i]),'selected':bool(primary[i]==selected)} for i,row in enumerate(swc)],
      'ports':[{'id':f'P{i}','label':labels[i],'sourceNodeId':int(swc[j,0]),'sourceIndex':int(j),'x':float(swc[j,2]),'y':float(swc[j,3]),'z':float(swc[j,4])} for i,j in enumerate(ports)]}
    (out/'allen_588402092.json').write_text(json.dumps(model,separators=(',',':'))+'\n')
    data=np.load(source/'impedance_matrices.npz'); refs={'frequenciesHz':data['frequencies_Hz'].tolist(),'units':'MOhm','sourceSwcSha256':source_hash,'matrices':{}}
    for preset in ['baseline','uniform','selective']:
      for k in [4,8]:
        z=data[f'{preset}_k{k}']; refs['matrices'][f'{preset}_k{k}']={'re':z.real.reshape(8,49).tolist(),'im':z.imag.reshape(8,49).tolist()}
        metrics=[]
        for z0,zc in zip(data[f'baseline_k{k}'],z):
          delta=zc-z0;norm=float(np.linalg.norm(z0,2));relative=np.linalg.solve(z0.T,delta.T).T
          metrics.append({'somaticGain':float(np.linalg.norm(delta[0])/norm),'distributedGain':float(np.linalg.norm(delta,2)/norm),'responseRelative':float(np.linalg.norm(relative,2))})
        refs['matrices'][f'{preset}_k{k}']['metrics']=metrics
    (out/'harmonic_reference.json').write_text(json.dumps(refs,separators=(',',':'))+'\n')
    manifests={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('*.json') if p.name!='checksums.json'}
    (out/'checksums.json').write_text(json.dumps(manifests,indent=2)+'\n')
    print(json.dumps({'sourceNodes':len(swc),'ports':len(ports),'sourceSha256':source_hash,'exports':manifests},indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-dir',type=Path,required=True);p.add_argument('--out',type=Path,default=Path('docs/data'));a=p.parse_args();export(a.source_dir,a.out)
