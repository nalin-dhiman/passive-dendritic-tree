import assert from 'node:assert/strict';
import fs from 'node:fs';
import {buildHarmonicTree,solveHarmonicTree,spectralNorm,invertComplex,multiplyComplex,comparisonMetrics,computeHarmonic,makeHarmonicY,validateGeometry,FREQUENCIES} from '../docs/js/harmonic.mjs';
const geometry=JSON.parse(fs.readFileSync(new URL('../docs/data/allen_588402092.json',import.meta.url)));
const reference=JSON.parse(fs.readFileSync(new URL('../docs/data/harmonic_reference.json',import.meta.url)));
let checks=0,worstParity=0,worstKCL=0,worstReciprocity=0;
const check=(name,condition)=>{assert.ok(condition,name);checks++;};
const close=(name,a,b,tol)=>check(name,Math.abs(a-b)<=tol);

// Independent one-compartment RC: conductance 0.1 µS and capacitance 0.2 nF.
for(const f of [0,1,10,100]){
 const t={n:1,parent:Int32Array.of(-1),axial:Float64Array.of(0),diag:Float64Array.of(.1),capacitance:Float64Array.of(.2)};
 const r=solveHarmonicTree(t,f,[0]),omega=2*Math.PI*f/1000,y=.2*omega,den=.01+y*y;
 close('RC real',r.re[0],.1/den,1e-13);close('RC imaginary',r.im[0],-y/den,1e-13);
}

// Exact two-segment resistive network: sealed nodes coupled with 2 µS,
// leak 1 µS at each node; inverse [[3,2],[2,3]]/5 MOhm.
const two={n:2,parent:Int32Array.of(-1,0),axial:Float64Array.of(0,2),diag:Float64Array.of(3,3),capacitance:Float64Array.of(0,0)};
const twoZ=solveHarmonicTree(two,0,[0,1]);[.6,.4,.4,.6].forEach((v,i)=>close('two node independent inverse',twoZ.zRe[i],v,1e-14));

// Matrix norms include repeated singular values and genuinely complex entries.
close('repeated spectrum norm',spectralNorm(Float64Array.of(1,0,0,1),new Float64Array(4),2),1,1e-14);
close('rectangular row norm',spectralNorm(Float64Array.of(3,0),Float64Array.of(0,4),1,2),5,1e-13);
for(let trial=0;trial<20;trial++){
 const ar=Float64Array.from({length:4},(_,i)=>Math.sin(3.1*(i+1)*(trial+1))),ai=Float64Array.from({length:4},(_,i)=>Math.cos(1.7*(i+1)*(trial+2)));
 const g00=ar[0]**2+ai[0]**2+ar[2]**2+ai[2]**2,g11=ar[1]**2+ai[1]**2+ar[3]**2+ai[3]**2;
 const gr=ar[0]*ar[1]+ai[0]*ai[1]+ar[2]*ar[3]+ai[2]*ai[3],gi=ar[0]*ai[1]-ai[0]*ar[1]+ar[2]*ai[3]-ai[2]*ar[3];
 const exact=Math.sqrt((g00+g11+Math.sqrt((g00-g11)**2+4*(gr**2+gi**2)))/2);
 close('complex spectral norm independent 2x2 eigenvalue',spectralNorm(ar,ai,2),exact,2e-13);
 const inv=invertComplex(ar,ai,2),identity=multiplyComplex(ar,ai,inv.re,inv.im,2);
 identity.re.forEach((v,i)=>close('complex inverse identity real',v,i%3===0?1:0,1e-12));identity.im.forEach(v=>close('complex inverse identity imaginary',v,0,1e-12));
}

// Geometry identity, exact material definition and source-point persistence.
check('retained all 2534 source nodes',geometry.nodes.length===2534);
check('no axon retained',geometry.nodes.every(n=>n.type!==2));
check('seven physical source IDs preserved',JSON.stringify(geometry.ports.map(p=>p.sourceNodeId))==='[1,1621,1783,2165,1216,182,708]');
check('frozen selected basal primary identity',geometry.selectedRootSourceId===724);
close('source selective radius factor',geometry.selectiveFactor,.8648954759986717,0);
check('source SHA exact',geometry.source.swcSha256==='8d3c5fc12fbee445d8cdbe3a2281029e4f87b219689856477e82e60f46d29bb9');
for(const k of [4,8])for(const preset of ['baseline','uniform','selective']){
 const tree=buildHarmonicTree(geometry,preset,k),ports=geometry.ports.map(p=>tree.sourceToMesh[p.sourceIndex]);
 check('expected refinement node count',tree.n===1+2533*k);
 for(let i=0;i<geometry.nodes.length;i++)for(let dim=0;dim<3;dim++)close('persistent physical source coordinate',tree.xyz[tree.sourceToMesh[i]*3+dim],geometry.nodes[i][['x','y','z'][dim]],1e-12);
 for(let fi=0;fi<FREQUENCIES.length;fi++){
  const f=FREQUENCIES[fi],sol=solveHarmonicTree(tree,f,ports),ref=reference.matrices[`${preset}_k${k}`];
  let delta=0,scale=0;
  for(let i=0;i<49;i++){delta+=Math.abs(sol.zRe[i]-ref.re[fi][i])**2+Math.abs(sol.zIm[i]-ref.im[fi][i])**2;scale+=ref.re[fi][i]**2+ref.im[fi][i]**2;}
  const expectedMetrics=ref.metrics[fi],computedMetrics=comparisonMetrics(reference.matrices[`baseline_k${k}`].re[fi],reference.matrices[`baseline_k${k}`].im[fi],sol.zRe,sol.zIm);
  for(const key of ['somaticGain','distributedGain','responseRelative'])close('seven-port norm against independent NumPy SVD',computedMetrics[key],expectedMetrics[key],1e-8);
  const err=Math.sqrt(delta/scale);worstParity=Math.max(worstParity,err);check(`saved matrix ${preset} k${k} ${f}Hz`,err<1e-9);
  let rec=0;for(let i=0;i<7;i++)for(let j=0;j<7;j++)rec=Math.max(rec,Math.hypot(sol.zRe[i*7+j]-sol.zRe[j*7+i],sol.zIm[i*7+j]-sol.zIm[j*7+i]));
  worstReciprocity=Math.max(worstReciprocity,rec);check('complex reciprocity',rec<1e-8);
  if(fi===4){
   for(let col=0;col<7;col++){
    const rr=new Float64Array(tree.n),ii=new Float64Array(tree.n),omega=2*Math.PI*f/1000;
    for(let i=0;i<tree.n;i++){const r=sol.re[i*7+col],v=sol.im[i*7+col];rr[i]=tree.leak[i]*r-omega*tree.capacitance[i]*v;ii[i]=tree.leak[i]*v+omega*tree.capacitance[i]*r;}
    for(let i=1;i<tree.n;i++){const p=tree.parent[i],g=tree.axial[i],r=g*(sol.re[i*7+col]-sol.re[p*7+col]),v=g*(sol.im[i*7+col]-sol.im[p*7+col]);rr[i]+=r;rr[p]-=r;ii[i]+=v;ii[p]-=v;}
    rr[ports[col]]-=1;const residual=Math.max(...rr.map((v,i)=>Math.hypot(v,ii[i])));worstKCL=Math.max(worstKCL,residual);check('independent full-field KCL',residual<1e-8);
   }
  }
 }
 if(preset==='baseline'){close('modeled dendritic baseline volume',tree.materials.dendriticVolumeUm3,834.65428117184,2e-10);close('fixed soma volume',tree.materials.somaVolumeUm3,996.0373715102695,1e-10);}
 else close('matched 5.91% dendritic volume reduction',1-tree.materials.dendriticVolumeUm3/834.65428117184,.0591,2e-13);
}

// API field phase, amplitude, identity copy and norm inequalities.
const base=computeHarmonic({model:'allen',preset:'baseline',inputPort:4,frequencyHz:10,includeSpectrum:true},geometry);
close('unchanged metric somatic',base.metrics.somaticGain,0,0);close('unchanged metric matrix',base.metrics.distributedGain,0,0);close('unchanged metric response',base.metrics.responseRelative,0,0);
check('actual eight-frequency sweep',JSON.stringify(base.spectrum.map(r=>r.frequencyHz))===JSON.stringify(FREQUENCIES));
const result=computeHarmonic({model:'allen',preset:'selective',inputPort:4,frequencyHz:10,amplitudePA:1,phaseDeg:0},geometry);
const shifted=computeHarmonic({model:'allen',preset:'selective',inputPort:4,frequencyHz:10,amplitudePA:2,phaseDeg:90},geometry);
for(const port of result.ports){const j=port.nodeIndex;close('phase shifted real',shifted.candidate.voltageRe[j],-2*result.candidate.voltageIm[j],1e-13);close('phase shifted imaginary',shifted.candidate.voltageIm[j],2*result.candidate.voltageRe[j],1e-13);}
check('root <= distributed <= response-relative',result.metrics.somaticGain<=result.metrics.distributedGain+1e-12&&result.metrics.distributedGain<=result.metrics.responseRelative+1e-12);
close('selected local harmonic difference',Math.hypot(result.difference.voltageRe[result.ports[4].nodeIndex],result.difference.voltageIm[result.ports[4].nodeIndex]),.17322297383687,1e-6);
close('selected soma harmonic difference',Math.hypot(result.difference.voltageRe[result.ports[0].nodeIndex],result.difference.voltageIm[result.ports[0].nodeIndex]),.00390077197574,1e-6);

// Synthetic Y retains constant radius on each edge and exact prescribed material.
const y=makeHarmonicY(),yb=buildHarmonicTree(y,'baseline',4),yu=buildHarmonicTree(y,'uniform',4),ys=buildHarmonicTree(y,'selective',4);
close('Y three-cylinder total length',yb.materials.lengthUm,1200,1e-10);close('Y cylinder volume',yb.materials.totalVolumeUm3,Math.PI*400*(1+2*.8**2),1e-9);
for(const t of [yu,ys])close('Y matched volume',1-t.materials.totalVolumeUm3/yb.materials.totalVolumeUm3,.0591,1e-12);
const yz=solveHarmonicTree(yb,10,y.ports.map(p=>yb.sourceToMesh[p.sourceIndex]));
// Identical daughters: root coupling to matching daughter locations is identical.
close('Y symmetry real',yz.zRe[4],yz.zRe[6],1e-9);close('Y symmetry imaginary',yz.zIm[4],yz.zIm[6],1e-9);
assert.throws(()=>computeHarmonic({model:'allen',frequencyHz:-1},geometry));checks++;
assert.throws(()=>buildHarmonicTree(geometry,'unknown'));checks++;
const bad=structuredClone(geometry);bad.nodes[1].parent=1;assert.throws(()=>validateGeometry(bad));checks++;
console.log(JSON.stringify({status:'PASS',checks,referenceMatrices:48,worstRelativeSavedMatrixError:worstParity,worstAbsoluteReciprocityErrorMOhm:worstReciprocity,worstAbsoluteKCLResidualNA:worstKCL},null,2));
