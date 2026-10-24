import {loadModelGeometry,computeHarmonic} from './harmonic.mjs';
self.onmessage=async({data})=>{
  const {type='compute',id,payload={}}=data;
  try{
    const geometry=await loadModelGeometry(payload.model||'allen');
    if(type==='geometry'){self.postMessage({type:'geometry',id,geometry});return;}
    const result=computeHarmonic(payload,geometry);
    // Do not transfer cached buffers: subsequent controls reuse the factors.
    self.postMessage({type:'result',id,result});
  }catch(error){self.postMessage({type:'error',id,message:error?.message||String(error)});}
};
