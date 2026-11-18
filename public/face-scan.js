// Wait for usable face samples, with one completion per scan and cancellation on close.
export async function collectFaceSamples({count,detect,isActive,report,timeoutMs=45000,pause=ms=>new Promise(r=>setTimeout(r,ms)),clock=Date.now}) {
 const started=clock();let samples=[];
 while(isActive() && clock()-started<timeoutMs) {
  const faces=await detect();if(!isActive())return null;
  let hint='Look at the camera in good light.';
  if(faces.length>1)hint='Keep only your face in the frame.';
  else if(faces.length===1 && faces[0].width<95)hint='Move a little closer to the camera.';
  else if(faces.length===1) {
   const descriptor=Array.from(faces[0].descriptor);
   if(samples.length && Math.sqrt(descriptor.reduce((sum,v,i)=>sum+(v-samples[0][i])**2,0))>.45)samples=[];
   samples.push(descriptor);report('Hold still — scanning your face…',samples.length/count);
   if(samples.length===count)return samples;
   await pause(300);continue;
  }
  samples=[];report(hint,0);await pause(250);
 }
 if(!isActive())return null;
 throw Error('No clear face found. Close this window and try again in better light.');
}
