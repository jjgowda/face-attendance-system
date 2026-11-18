// Stream public model files with real progress; keep the existing face model/encodings.
export async function loadFaceModels(faceapi,{report=()=>{},timeoutMs=60000}={}) {
 const controller=new AbortController(),timer=setTimeout(()=>controller.abort(),timeoutMs);
 const specs=[['tinyFaceDetector','tiny_face_detector_model'],['faceLandmark68TinyNet','face_landmark_68_tiny_model'],['faceRecognitionNet','face_recognition_model']];
 try{
  report({phase:'download',percent:0});
  const pending=specs.filter(([name])=>!faceapi.nets[name].isLoaded);
  const manifests=await Promise.all(pending.map(async([name,file])=>{const r=await fetch(`/models/${file}-weights_manifest.json`,{signal:controller.signal});if(!r.ok)throw Error('Model download failed.');return {name,manifest:await r.json()};}));
  const bytes=weight=>weight.shape.reduce((a,b)=>a*b,1)*(({uint8:1,uint16:2,float32:4,int32:4,bool:1})[weight.quantization?.dtype||weight.dtype]||4);
  const total=manifests.reduce((sum,m)=>sum+m.manifest.reduce((s,g)=>s+g.weights.reduce((n,w)=>n+bytes(w),0),0),0);let loaded=0;
  const progress=count=>{loaded+=count;report({phase:'download',percent:Math.min(100,Math.floor(loaded/Math.max(total,1)*100)),loaded,total});};
  const download=async url=>{
   const r=await fetch(url,{signal:controller.signal});if(!r.ok)throw Error('Model download failed.');
   if(!r.body){const buffer=await r.arrayBuffer();progress(buffer.byteLength);return buffer;}
   const reader=r.body.getReader(),chunks=[];let length=0;
   while(true){const {value,done}=await reader.read();if(done)break;chunks.push(value);length+=value.byteLength;progress(value.byteLength);}
   const combined=new Uint8Array(length);let offset=0;for(const chunk of chunks){combined.set(chunk,offset);offset+=chunk.length;}return combined.buffer;
  };
  const loadWeights=faceapi.tf.io.weightsLoaderFactory(paths=>Promise.all(paths.map(download)));
  await Promise.all(manifests.map(async({name,manifest})=>{const weights=await loadWeights(manifest,'/models');if(controller.signal.aborted){Object.values(weights).forEach(w=>w.dispose());throw Error('Download timed out.');}faceapi.nets[name].loadFromWeightMap(weights);}));
  report({phase:'ready',percent:100});
 }catch(error){controller.abort();throw Error(error.name==='AbortError'?'Face scanner download took too long. Close this window and retry on a stronger connection.':error.message||'Could not load face scanner. Please retry.');}
 finally{clearTimeout(timer);}
}
