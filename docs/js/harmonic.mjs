/** Complex passive-cable finite volumes, in µm, ms, mV, nA, µS, nF.
 * A harmonic phasor is Re[V exp(i 2π f t/1000)], with t in ms.
 * This module computes periodic steady state, not a time-integrated transient.
 */
import {frustum} from './engine.mjs';
export const FREQUENCIES = Object.freeze([0,1,2,5,10,20,50,100]);
export const PASSIVE_PARAMETERS = Object.freeze({Ra:100,gm:1e-4,cm:1});
const models=new Map(), trees=new Map(), solutions=new Map();
const MAX_CACHE=48;
function remember(map,key,value){if(map.size>=MAX_CACHE)map.delete(map.keys().next().value);map.set(key,value);return value;}

/** Synthetic constant-radius Y with seven persistent collocated ports. */
export function makeHarmonicY(){
  const h=Math.sqrt(400**2-320**2), nodes=[
    {x:0,y:0,r:1,parent:-1},{x:0,y:200,r:1,parent:0},{x:0,y:400,r:1,parent:1},
    {x:-160,y:400+h/2,r:.8,parent:2,selected:true},{x:-320,y:400+h,r:.8,parent:3,selected:true},
    {x:160,y:400+h/2,r:.8,parent:2},{x:320,y:400+h,r:.8,parent:5}].map((n,i)=>({id:i+1,type:3,z:0,selected:false,...n}));
  for(let i=1;i<nodes.length;i++){nodes[i].edgeR0=i<3?1:.8;nodes[i].edgeR1=nodes[i].edgeR0;}
  return {schemaVersion:1,id:'y',name:'Ideal passive Y cable',source:{provider:'Synthetic geometry',coordinates:'micrometres',radius:'micrometres'},somaMode:'none',subdivisionMultiplier:32,
    selectiveFactor:Math.sqrt(1-.0591/(.8**2/(1+2*.8**2))),nodes,
    ports:nodes.map((n,i)=>({id:`P${i}`,label:['Root','Parent midpoint','Branch junction','Selected daughter midpoint','Selected daughter tip','Other daughter midpoint','Other daughter tip'][i],sourceNodeId:n.id,sourceIndex:i,x:n.x,y:n.y,z:0}))};
}
export async function loadModelGeometry(id='allen'){
  if(!['allen','y'].includes(id))throw Error('Unknown geometry.');
  if(models.has(id))return models.get(id);
  if(id==='y'){const y=makeHarmonicY();models.set(id,y);return y;}
  const response=await fetch(new URL('../data/allen_588402092.json',import.meta.url));
  if(!response.ok)throw Error(`Morphology download failed (${response.status}).`);
  const data=await response.json();validateGeometry(data);models.set(id,data);return data;
}
export function validateGeometry(geometry){
  const n=geometry?.nodes;
  if(!Array.isArray(n)||n.length<2||n.length>5000)throw Error('Geometry must contain 2–5000 source nodes.');
  n.forEach((p,i)=>{if(![p.x,p.y,p.z,p.r].every(Number.isFinite)||p.r<=0||!Number.isInteger(p.parent)||(i===0?p.parent!==-1:p.parent<0||p.parent>=i))throw Error(`Invalid geometry at node ${i}.`);});
  if(!Array.isArray(geometry.ports)||geometry.ports.length!==7||new Set(geometry.ports.map(p=>p.sourceIndex)).size!==7||geometry.ports.some(p=>!Number.isInteger(p.sourceIndex)||p.sourceIndex<0||p.sourceIndex>=n.length))throw Error('Seven distinct persistent source ports required.');
  if(!(geometry.selectiveFactor>0&&geometry.selectiveFactor<=1))throw Error('Invalid selective radius factor.');
  return geometry;
}

export function buildHarmonicTree(geometry,preset='baseline',subdivisions=4,parameters=PASSIVE_PARAMETERS){
  validateGeometry(geometry);
  if(!['baseline','uniform','selective'].includes(preset))throw Error('Unknown material preset.');
  if(![1,2,4,8].includes(subdivisions))throw Error('Choose 1, 2, 4 or 8 subdivisions.');
  const {Ra,gm,cm}={...PASSIVE_PARAMETERS,...parameters};
  if(![Ra,gm,cm].every(v=>Number.isFinite(v)&&v>0))throw Error('Positive finite passive parameters required.');
  const k=subdivisions*(geometry.subdivisionMultiplier||1), source=geometry.nodes,n=1+(source.length-1)*k;
  if(n>40000)throw Error('Refined geometry exceeds the browser resource limit; no geometry was discarded.');
  const parent=new Int32Array(n).fill(-1),axial=new Float64Array(n),area=new Float64Array(n),xyz=new Float64Array(3*n),sourceToMesh=new Int32Array(source.length);
  const scales=source.map(p=>preset==='uniform'&&!(geometry.somaMode!=='none'&&p.type===1)?.97:preset==='selective'&&p.selected?geometry.selectiveFactor:1);
  xyz.set([source[0].x,source[0].y,source[0].z]);
  let next=1,volume=0,cableArea=0,lengthTotal=0;
  for(let j=1;j<source.length;j++){
    const b=source[j],a=source[b.parent],length=Math.hypot(b.x-a.x,b.y-a.y,b.z-a.z);if(!(length>0))throw Error('Zero-length edges unsupported.');
    const scale0=scales[b.parent],scale1=scales[j];
    const r0=(b.edgeR0??(a.type===1?b.r:a.r))*(b.edgeR0!==undefined?scale1:a.type===1?scale1:scale0), r1=(b.edgeR1??b.r)*scale1;
    const dl=length/k;let p=sourceToMesh[b.parent];lengthTotal+=length;
    for(let q=0;q<k;q++){
      const i=next++,u0=q/k,u1=(q+1)/k,rr0=r0+(r1-r0)*u0,rr1=r0+(r1-r0)*u1,rm=(rr0+rr1)/2;
      parent[i]=p;axial[i]=100*Math.PI*rr0*rr1/(Ra*dl);
      const left=frustum(dl/2,rr0,rm),right=frustum(dl/2,rm,rr1),whole=frustum(dl,rr0,rr1);
      area[p]+=left.area;area[i]+=right.area;volume+=whole.volume;cableArea+=whole.area;
      xyz.set([a.x+(b.x-a.x)*u1,a.y+(b.y-a.y)*u1,a.z+(b.z-a.z)*u1],i*3);p=i;
    }
    sourceToMesh[j]=p;
  }
  const sphere=geometry.somaMode==='fixed sphere',somaArea=sphere?4*Math.PI*source[0].r**2:0,somaVolume=sphere?4*Math.PI*source[0].r**3/3:0;
  area[0]+=somaArea;
  const capacitance=area.map(a=>a*cm*1e-5),leak=area.map(a=>a*gm*.01),diag=leak.slice();
  for(let i=1;i<n;i++){diag[i]+=axial[i];diag[parent[i]]+=axial[i];}
  const materials={dendriticVolumeUm3:volume,dendriticAreaUm2:cableArea,somaVolumeUm3:somaVolume,somaAreaUm2:somaArea,totalVolumeUm3:volume+somaVolume,totalAreaUm2:cableArea+somaArea,lengthUm:lengthTotal,capacitancePF:capacitance.reduce((a,b)=>a+b,0)*1000,leakNS:leak.reduce((a,b)=>a+b,0)*1000};
  return {n,parent,axial,area,capacitance,leak,diag,xyz,sourceToMesh,materials,subdivisions,actualSubdivisions:k,parameters:{Ra,gm,cm}};
}

/** Symmetric complex tree elimination; all RHS columns share one factorization. */
export function solveHarmonicTree(tree,frequencyHz,ports){
  if(!Number.isFinite(frequencyHz)||frequencyHz<0||frequencyHz>1000)throw Error('Frequency must be within 0–1000 Hz.');
  const n=tree.n,m=ports.length,dr=tree.diag.slice(),di=tree.capacitance.map(c=>2*Math.PI*frequencyHz/1000*c),br=new Float64Array(n*m),bi=new Float64Array(n*m);
  ports.forEach((p,j)=>{if(!Number.isInteger(p)||p<0||p>=n)throw Error('Invalid persistent port.');br[p*m+j]=1;});
  for(let i=n-1;i>0;i--){const p=tree.parent[i],g=tree.axial[i],den=dr[i]**2+di[i]**2,ar=g*dr[i]/den,ai=-g*di[i]/den;
    if(!(den>0)||!Number.isFinite(den))throw Error('Invalid complex tree pivot.');dr[p]-=g*ar;di[p]-=g*ai;
    for(let j=0;j<m;j++){const a=i*m+j,b=p*m+j;br[b]+=ar*br[a]-ai*bi[a];bi[b]+=ar*bi[a]+ai*br[a];}}
  const vr=new Float64Array(n*m),vi=new Float64Array(n*m);
  for(let i=0;i<n;i++){const p=tree.parent[i],g=tree.axial[i],den=dr[i]**2+di[i]**2;
    for(let j=0;j<m;j++){const a=i*m+j,b=p*m+j,rr=br[a]+(p<0?0:g*vr[b]),ii=bi[a]+(p<0?0:g*vi[b]);vr[a]=(rr*dr[i]+ii*di[i])/den;vi[a]=(ii*dr[i]-rr*di[i])/den;}}
  const zr=new Float64Array(m*m),zi=new Float64Array(m*m);ports.forEach((p,i)=>{zr.set(vr.subarray(p*m,(p+1)*m),i*m);zi.set(vi.subarray(p*m,(p+1)*m),i*m);});
  return {re:vr,im:vi,zRe:zr,zIm:zi};
}

/** Largest singular value via full symmetric Jacobi diagonalization of A* A.
 * The equivalent real Hermitian representation avoids a fragile power-method
 * stopping rule when leading singular values are repeated or nearly equal.
 */
export function spectralNorm(re,im,rows,cols=rows){
  if(re.length!==rows*cols||im.length!==re.length)throw Error('Matrix dimensions do not match.');
  let scale=0;for(let i=0;i<re.length;i++)scale=Math.max(scale,Math.hypot(re[i],im[i]));if(scale===0)return 0;
  const n=2*cols,a=new Float64Array(n*n);
  for(let i=0;i<cols;i++)for(let j=0;j<cols;j++){let r=0,v=0;for(let k=0;k<rows;k++){const x=k*cols+i,y=k*cols+j;r+=(re[x]*re[y]+im[x]*im[y])/(scale*scale);v+=(re[x]*im[y]-im[x]*re[y])/(scale*scale);}a[i*n+j]=r;a[(i+cols)*n+j+cols]=r;a[i*n+j+cols]=-v;a[(i+cols)*n+j]=v;}
  for(let sweep=0;sweep<100;sweep++){
    let off=0,maxd=0;for(let i=0;i<n;i++)maxd=Math.max(maxd,Math.abs(a[i*n+i]));
    for(let p=0;p<n-1;p++)for(let q=p+1;q<n;q++){
      const apq=a[p*n+q];off=Math.max(off,Math.abs(apq));if(Math.abs(apq)<=1e-15*Math.max(1,maxd))continue;
      const app=a[p*n+p],aqq=a[q*n+q],tau=(aqq-app)/(2*apq),t=(tau>=0?1:-1)/(Math.abs(tau)+Math.sqrt(1+tau*tau)),c=1/Math.sqrt(1+t*t),s=t*c;
      for(let k=0;k<n;k++)if(k!==p&&k!==q){const akp=a[k*n+p],akq=a[k*n+q],v=c*akp-s*akq,w=s*akp+c*akq;a[k*n+p]=a[p*n+k]=v;a[k*n+q]=a[q*n+k]=w;}
      a[p*n+p]=app-t*apq;a[q*n+q]=aqq+t*apq;a[p*n+q]=a[q*n+p]=0;
    }
    if(off<=2e-14*Math.max(1,maxd))break;if(sweep===99)throw Error('Spectral-norm iteration did not converge.');
  }
  let largest=0;for(let i=0;i<n;i++)largest=Math.max(largest,a[i*n+i]);return scale*Math.sqrt(Math.max(0,largest));
}

/** Partial-pivoted complex Gaussian inverse, used only for seven-port matrices. */
export function invertComplex(re,im,n){
  const ar=re.slice(),ai=im.slice(),xr=new Float64Array(n*n),xi=new Float64Array(n*n);for(let i=0;i<n;i++)xr[i*n+i]=1;
  for(let k=0;k<n;k++){
    let p=k,best=0;for(let i=k;i<n;i++){const v=Math.hypot(ar[i*n+k],ai[i*n+k]);if(v>best){p=i;best=v;}}if(!(best>0))throw Error('Singular impedance matrix.');
    if(p!==k)for(let j=0;j<n;j++)for(const a of [ar,ai,xr,xi]){const t=a[k*n+j];a[k*n+j]=a[p*n+j];a[p*n+j]=t;}
    const r=ar[k*n+k],v=ai[k*n+k],den=r*r+v*v;
    for(let j=0;j<n;j++)for(const [a,b] of [[ar,ai],[xr,xi]]){const rr=a[k*n+j],ii=b[k*n+j];a[k*n+j]=(rr*r+ii*v)/den;b[k*n+j]=(ii*r-rr*v)/den;}
    for(let i=0;i<n;i++)if(i!==k){const r=ar[i*n+k],v=ai[i*n+k];for(let j=0;j<n;j++)for(const [a,b] of [[ar,ai],[xr,xi]]){a[i*n+j]-=r*a[k*n+j]-v*b[k*n+j];b[i*n+j]-=r*b[k*n+j]+v*a[k*n+j];}}
  }
  return {re:xr,im:xi};
}

export function multiplyComplex(ar,ai,br,bi,n){const re=new Float64Array(n*n),im=new Float64Array(n*n);for(let i=0;i<n;i++)for(let j=0;j<n;j++)for(let k=0;k<n;k++){const a=i*n+k,b=k*n+j;re[i*n+j]+=ar[a]*br[b]-ai[a]*bi[b];im[i*n+j]+=ar[a]*bi[b]+ai[a]*br[b];}return {re,im};}
export function comparisonMetrics(z0r,z0i,zcr,zci){
  const n=Math.sqrt(z0r.length),dr=zcr.map((v,i)=>v-z0r[i]),di=zci.map((v,i)=>v-z0i[i]),base=spectralNorm(z0r,z0i,n);
  let root=0;for(let j=0;j<n;j++)root+=dr[j]**2+di[j]**2;
  const inverse=invertComplex(z0r,z0i,n),relative=multiplyComplex(dr,di,inverse.re,inverse.im,n);
  return {somaticGain:Math.sqrt(root)/base,distributedGain:spectralNorm(dr,di,n)/base,responseRelative:spectralNorm(relative.re,relative.im,n),normalization:'Global baseline seven-port spectral norm; response-relative uses D Z0^-1'};
}

function cachedTree(geometry,preset,k,parameters){const key=JSON.stringify([geometry.id,preset,k,parameters]);if(trees.has(key))return trees.get(key);return remember(trees,key,buildHarmonicTree(geometry,preset,k,parameters));}
function cachedSolution(geometry,preset,k,parameters,f){const key=JSON.stringify([geometry.id,preset,k,parameters,f]);if(solutions.has(key))return solutions.get(key);const t=cachedTree(geometry,preset,k,parameters),ports=geometry.ports.map(p=>t.sourceToMesh[p.sourceIndex]);return remember(solutions,key,solveHarmonicTree(t,f,ports));}

export function computeHarmonic(payload={},geometry=models.get(payload.model||'allen')){
  const {model='allen',preset='uniform',frequencyHz=10,amplitudePA=1,phaseDeg=0,inputPort=3,subdivisions=4,includeSpectrum=false}=payload;
  const parameters={...PASSIVE_PARAMETERS,...payload.parameters};
  if(!geometry||geometry.id!==model)throw Error('Load the requested geometry before computing.');
  if(!Number.isInteger(inputPort)||inputPort<0||inputPort>6)throw Error('Choose one of the seven persistent input ports.');
  if(!Number.isFinite(amplitudePA)||Math.abs(amplitudePA)>1000||!Number.isFinite(phaseDeg)||Math.abs(phaseDeg)>36000)throw Error('Current must be finite and within ±1000 pA.');
  if(!Number.isFinite(frequencyHz)||frequencyHz<0||frequencyHz>100)throw Error('Explorer frequency range is 0–100 Hz.');
  const baseTree=cachedTree(geometry,'baseline',subdivisions,parameters),candidateTree=cachedTree(geometry,preset,subdivisions,parameters),base=cachedSolution(geometry,'baseline',subdivisions,parameters,frequencyHz),candidate=cachedSolution(geometry,preset,subdivisions,parameters,frequencyHz);
  const c=Math.cos(phaseDeg*Math.PI/180)*amplitudePA/1000,s=Math.sin(phaseDeg*Math.PI/180)*amplitudePA/1000;
  function field(solution){const re=new Float64Array(baseTree.n),im=new Float64Array(baseTree.n);for(let i=0;i<baseTree.n;i++){const j=i*7+inputPort;re[i]=solution.re[j]*c-solution.im[j]*s;im[i]=solution.re[j]*s+solution.im[j]*c;}return {voltageRe:re,voltageIm:im,zRe:solution.zRe.slice(),zIm:solution.zIm.slice()};}
  const baseline=field(base),changed=field(candidate),difference={voltageRe:changed.voltageRe.map((v,i)=>v-baseline.voltageRe[i]),voltageIm:changed.voltageIm.map((v,i)=>v-baseline.voltageIm[i])};
  const result={model,preset,frequencyHz,amplitudePA,phaseDeg,inputPort,parameters,geometry,ports:geometry.ports.map(p=>({...p,nodeIndex:baseTree.sourceToMesh[p.sourceIndex]})),
    mesh:{subdivisions,actualSubdivisions:baseTree.actualSubdivisions,nodeCount:baseTree.n},fullField:{xyz:baseTree.xyz,parent:baseTree.parent,sourceToMesh:baseTree.sourceToMesh},
    baseline,candidate:changed,difference,metrics:comparisonMetrics(base.zRe,base.zIm,candidate.zRe,candidate.zIm),materials:{baseline:baseTree.materials,candidate:candidateTree.materials},
    reductions:{dendriticVolumeFraction:1-candidateTree.materials.dendriticVolumeUm3/baseTree.materials.dendriticVolumeUm3,totalVolumeFraction:1-candidateTree.materials.totalVolumeUm3/baseTree.materials.totalVolumeUm3,areaFraction:1-candidateTree.materials.totalAreaUm2/baseTree.materials.totalAreaUm2},
    scope:'Periodic steady-state passive voltages relative to resting potential; no spikes or time-integrated transients. Seven-port errors describe the specified ports.'};
  if(includeSpectrum)result.spectrum=FREQUENCIES.map(f=>{const a=cachedSolution(geometry,'baseline',subdivisions,parameters,f),b=cachedSolution(geometry,preset,subdivisions,parameters,f);return {frequencyHz:f,...comparisonMetrics(a.zRe,a.zIm,b.zRe,b.zIm)};});
  return result;
}
