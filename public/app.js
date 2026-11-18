import {collectFaceSamples} from '/face-scan.js';
import {loadFaceModels} from '/face-loader.js';
const page = document.documentElement.dataset.page;
let pendingRequests = 0;
function requestLoading(start) { pendingRequests = Math.max(0,pendingRequests+(start?1:-1)); document.body.dataset.loading=String(pendingRequests>0); }
const $ = s=>document.querySelector(s);
const state={config:null,employees:[],models:null,library:null,stream:null,generation:0,admin:null,permissionPromise:null,permissionBusy:false,modelProgress:null};
const time = value=>value?new Date(value).toLocaleTimeString('en-IN',{timeZone:'Asia/Kolkata',hour:'2-digit',minute:'2-digit'}):'—';
const message = (text,error=false)=>{const el=$('#notice');el.hidden=false;el.textContent=text;el.classList.toggle('error',error);};
async function api(path,data) {requestLoading(true);try{const r=await fetch('/api/'+path,{method:data===undefined?'GET':'POST',headers:data===undefined?{}:{'Content-Type':'application/json'},body:data===undefined?undefined:JSON.stringify(data)});let b;try{b=await r.json();}catch{throw Error('Unable to reach the attendance service. Please retry.');}if(!r.ok)throw Object.assign(Error(b.error||'Request failed.'),{status:r.status});return b;}finally{requestLoading(false);}}
async function loadEmployees() {
 const register=$('#registerEmployee');register.disabled=true;
 try{const b=await api('employees');state.employees=b.employees;register.replaceChildren(new Option('Choose your name',''));
 for(const e of b.employees){const option=new Option(e.name+(e.registered?' — Already registered':''),e.id);option.disabled=!!e.registered;register.add(option);}
 }finally{register.disabled=false;}
}
function tab(register) {$('#attendanceForm').hidden=register;$('#registerForm').hidden=!register;$('#attendanceTab').classList.toggle('active',!register);$('#registerTab').classList.toggle('active',register);$('#attendanceTab').setAttribute('aria-selected',String(!register));$('#registerTab').setAttribute('aria-selected',String(register));$('#notice').hidden=true;}
$('#attendanceTab').onclick=()=>tab(false);$('#registerTab').onclick=()=>tab(true);
for(const radio of document.querySelectorAll('[name=action]'))radio.onchange=()=>{$('#scanButton').replaceChildren(document.createTextNode(`Check ${radio.value==='in'?'in':'out'}`));};
function cameraError(e) {if(e.name==='NotAllowedError')return 'Camera permission was denied. Allow camera access in your browser settings and retry.';if(e.name==='NotFoundError')return 'No camera found on this device.';if(e.name==='NotReadableError')return 'The camera is busy. Close other camera apps and retry.';return e.message||'Could not start camera.';}
function reportModelProgress(progress){
 state.modelProgress=progress;const dialog=$('#cameraDialog');
 if(dialog.open && dialog.dataset.scanning==='true' && progress.phase!=='ready'){
  $('#cameraStatus').textContent=progress.phase==='library'?'Loading face scanner…':`Loading face scanner… ${progress.percent}%`;
  $('#cameraProgress span').style.width=`${progress.percent||0}%`;
 }
}
async function models() {
 if(!window.faceapi){
  if(!state.library)state.library=new Promise((resolve,reject)=>{
   reportModelProgress({phase:'library',percent:0});const script=document.createElement('script');script.src='/vendor/face-api.js';
   const timer=setTimeout(()=>{script.remove();state.library=null;reject(Error('Scanner download took too long. Close and retry with a stronger connection.'));},20000);
   script.onload=()=>{clearTimeout(timer);resolve();};script.onerror=()=>{clearTimeout(timer);script.remove();state.library=null;reject(Error('Could not load the face scanner. Check your connection and retry.'));};document.head.append(script);
  });
  await state.library;
 }
 if(!state.models)state.models=loadFaceModels(faceapi,{report:reportModelProgress}).catch(error=>{state.models=null;throw error;});
 return state.models;
}
function warmScanner(){if(page!=='admin' && !navigator.connection?.saveData)void models().catch(()=>{});}
$('#registerEmployee').addEventListener('change',warmScanner);
async function showPermissions() {
 try {
  const checks=await Promise.all([navigator.permissions.query({name:'camera'}),navigator.permissions.query({name:'geolocation'})]);
  if(checks[0].state==='granted' && (!state.config.office.enabled || checks[1].state==='granted')){warmScanner();return;}
 }catch{}
 if(!$('#permissionDialog').open)$('#permissionDialog').showModal();
}
async function preparePermissions() {
 if(state.permissionBusy)return state.permissionPromise;
 state.permissionBusy=true;warmScanner();
 state.permissionPromise=(async()=>{
  const dialog=$('#permissionDialog'),label=$('#permissionText'),button=$('#enablePermissions');
  label.textContent='Choose Allow when your browser asks for camera and location access.';
  button.disabled=true;button.textContent='Requesting access…';button.classList.add('is-loading');dialog.setAttribute('aria-busy','true');
  const errors=[];
  try{if(!navigator.mediaDevices?.getUserMedia)throw Error('Use Safari or Chrome over HTTPS to allow camera access.');const stream=await navigator.mediaDevices.getUserMedia({video:true,audio:false});stream.getTracks().forEach(t=>t.stop());}catch(e){errors.push(cameraError(e));}
  if(state.config.office.enabled){try{await location();}catch(e){errors.push(e.message);}}
  dialog.removeAttribute('aria-busy');state.permissionBusy=false;button.disabled=false;button.classList.remove('is-loading');
  if(errors.length){label.textContent=errors.join(' ');button.textContent='Try again';}else{button.textContent='Enable camera & location';if(dialog.open)dialog.close();}
 })();
 return state.permissionPromise;
}
$('#enablePermissions').onclick=()=>void preparePermissions();
$('#closePermissions').onclick=()=>$('#permissionDialog').close();
$('#skipPermissions').onclick=()=>$('#permissionDialog').close();
function stopCamera(){state.generation++;state.stream?.getTracks().forEach(t=>t.stop());state.stream=null;$('#video').srcObject=null;}
$('#closeCamera').onclick=()=>$('#cameraDialog').close();$('#cameraDialog').addEventListener('close',stopCamera);
async function openCamera(mode,onCapture) {
 const generation=++state.generation,dialog=$('#cameraDialog');
 $('#cameraTitle').textContent=mode==='register'?'Register your face':'Verifying your face';
 $('#cameraStatus').textContent='Starting camera…';$('#cameraProgress span').style.width='0%';dialog.dataset.scanning='true';dialog.showModal();
 try {
  if(state.permissionBusy)await state.permissionPromise;
  if(generation!==state.generation)return;
  if(!navigator.mediaDevices?.getUserMedia)throw Error('Open this website in Safari or Chrome over HTTPS to use the camera.');
  const modelReady=models();modelReady.catch(()=>{});
  const stream=await navigator.mediaDevices.getUserMedia({video:{facingMode:'user',width:{ideal:640},height:{ideal:480}},audio:false});
  if(generation!==state.generation){stream.getTracks().forEach(t=>t.stop());return;}
  state.stream=stream;const video=$('#video');video.srcObject=stream;await video.play();
  if(state.modelProgress?.phase!=='ready')reportModelProgress(state.modelProgress||{phase:'download',percent:0});await modelReady;
  if(generation!==state.generation)return;
  $('#cameraProgress span').style.width='0%';$('#cameraStatus').textContent='Look at the camera…';
  void automaticScan(mode,onCapture,generation);
 }catch(e){if(generation!==state.generation)return;$('#cameraStatus').textContent=cameraError(e);dialog.dataset.scanning='false';stopCamera();}
}
async function automaticScan(mode,onCapture,generation) {
 const active=()=>generation===state.generation && $('#cameraDialog').open;
 try {
  const canvas=document.createElement('canvas'),video=$('#video');
  const descriptors=await collectFaceSamples({count:mode==='register'?3:2,isActive:active,
   report:(text,progress)=>{$('#cameraStatus').textContent=text;$('#cameraProgress span').style.width=`${progress*100}%`;},
   detect:async()=>{
    if(!video.videoWidth)return [];
    canvas.width=640;canvas.height=Math.round(640*video.videoHeight/video.videoWidth);canvas.getContext('2d').drawImage(video,0,0,canvas.width,canvas.height);
    const faces=await faceapi.detectAllFaces(canvas,new faceapi.TinyFaceDetectorOptions({inputSize:320,scoreThreshold:.65})).withFaceLandmarks(true).withFaceDescriptors();
    return faces.map(face=>({width:face.detection.box.width,descriptor:face.descriptor}));
   }});
  if(!descriptors||!active())return;
  $('#cameraStatus').textContent=mode==='register'?'Saving your registration…':'Checking your location…';
  await onCapture(descriptors,active);
  if(active())$('#cameraDialog').close();
 }catch(e){if(active()){$('#cameraStatus').textContent=e.message;$('#cameraDialog').dataset.scanning='false';stopCamera();}}
}
async function location() {if(!state.config.office.enabled)return null;if(!navigator.geolocation)throw Error('This browser does not support location. Use Safari or Chrome.');return new Promise((resolve,reject)=>navigator.geolocation.getCurrentPosition(p=>resolve({lat:p.coords.latitude,lng:p.coords.longitude,accuracy:p.coords.accuracy,timestamp:p.timestamp}),e=>reject(Error(e.code===1?'Location access was denied. Allow location in browser settings and retry.':'Could not get an accurate location. Move near a window and retry.')),{enableHighAccuracy:true,maximumAge:0,timeout:20000}));}
$('#attendanceForm').onsubmit=async e=>{e.preventDefault();const action=$('input[name=action]:checked').value;await openCamera('verify',async(descriptors,active)=>{const loc=await location();if(!active())return;$('#cameraStatus').textContent='Identifying your face…';const {challenge}=await api('challenge',{});if(!active())return;const result=await api('attendance',{descriptors,challenge,action,location:loc});message(`${result.message}, ${result.name}.\n${time(result.time)} IST${result.locker?' · Locker '+result.locker:''}`);});};
$('#registerForm').onsubmit=async e=>{
 e.preventDefault();const details={employeeId:$('#registerEmployee').value,registrationCode:$('#registrationCode').value.trim(),consent:$('#consent').checked};
 try{await api('register/check',details);}catch(error){message(error.message,true);return;}
 await openCamera('register',async(descriptors,active)=>{if(!active())return;const result=await api('register',{...details,descriptors});localStorage.setItem('r3-employee',result.id);window.location.assign('/?registered=1');});
};
async function openAdmin(){ $('#employeeView').hidden=true;$('#adminView').hidden=false;$('#adminAuth').hidden=false;$('#adminDashboard').hidden=true;$('#authMessage').textContent='';const setup=state.config.setupRequired;$('#authTitle').textContent=setup?'Set up your admin account':'Admin sign in';$('#authDescription').textContent=setup?'Enter your one-time setup code and choose a password of at least 12 characters.':'Manage your team, attendance and lockers.';$('#setupField').hidden=!setup;$('#setupCode').required=setup;$('#adminPassword').minLength=setup?12:1;$('#adminPassword').autocomplete=setup?'new-password':'current-password';try{await loadAdmin();}catch(e){if(e.status!==401)$('#authMessage').textContent=e.message;}}

$('#adminAuth').onsubmit=async e=>{e.preventDefault();const button=$('#adminAuth button');button.disabled=true;try{await api(state.config.setupRequired?'admin/setup':'admin/login',{password:$('#adminPassword').value,setupCode:$('#setupCode').value.trim()});state.config.setupRequired=false;$('#adminPassword').value='';$('#setupCode').value='';await loadAdmin();}catch(e){$('#authMessage').textContent=e.message;}finally{button.disabled=false;}};
function text(tag,value,className){const e=document.createElement(tag);e.textContent=value;if(className)e.className=className;return e;}
async function loadAdmin(){const panel=$('#adminDashboard');panel.setAttribute('aria-busy','true');try{const day=$('#attendanceDate').value||state.config.day;const data=await api('admin/data?day='+encodeURIComponent(day));state.admin=data;$('#adminAuth').hidden=true;$('#adminDashboard').hidden=false;$('#attendanceDate').value=day;$('#teamCount').textContent=`${data.employees.filter(e=>e.active).length} / 5`;$('#inCount').textContent=data.attendance.length;$('#outCount').textContent=data.attendance.filter(e=>e.check_out).length;
 const tbody=$('#attendanceRows');tbody.replaceChildren();if(!data.attendance.length){const tr=document.createElement('tr'),td=text('td','No attendance recorded for this date.','empty');td.colSpan=4;tr.append(td);tbody.append(tr);}for(const a of data.attendance){const tr=document.createElement('tr');for(const value of [a.name,time(a.check_in),time(a.check_out),a.locker||'—'])tr.append(text('td',value));tbody.append(tr);}
 const cards=$('#employeeCards');cards.replaceChildren();if(!data.employees.length)cards.append(text('p','No employees registered yet. Add an employee to create a personal registration code.','empty'));for(const employee of data.employees){const row=text('div','','employee-row'),person=text('div','','person');person.append(text('strong',employee.name),text('small',employee.code+(employee.active?' · Active':' · Inactive')+(employee.registered?' · Face registered':' · Registration pending')));const locker=document.createElement('input');locker.value=employee.locker;locker.placeholder='Locker no.';locker.maxLength=30;locker.setAttribute('aria-label',`Locker for ${employee.name}`);const save=text('button','Save locker','secondary');save.onclick=()=>adminAction(save,()=>api('admin/employee',{id:employee.id,locker:locker.value,active:!!employee.active}));const toggle=text('button',employee.active?'Deactivate':'Activate','secondary');toggle.onclick=()=>adminAction(toggle,()=>api('admin/employee',{id:employee.id,locker:locker.value,active:!employee.active}));const del=text('button','Delete & reset face','quiet danger');del.onclick=()=>{if(confirm(`Delete ${employee.name}'s face registration and attendance history? They can then register again.`))adminAction(del,()=>api('admin/delete-employee',{id:employee.id}));};row.append(person);if(!employee.registered){const invitation=text('div','','employee-invitation');invitation.append(text('small','Registration code'),text('code',employee.registration_code||'Not issued'));const renew=text('button','New code','secondary');renew.onclick=()=>adminAction(renew,()=>api('admin/registration-code',{id:employee.id}));invitation.append(renew);row.append(invitation);}row.append(locker,save,toggle,del);cards.append(row);}
 $('#officeLat').value=data.office.lat;$('#officeLng').value=data.office.lng;$('#officeRadius').value=data.office.radius;$('#geoEnabled').checked=data.office.enabled;
}finally{panel.setAttribute('aria-busy','false');}}
async function adminAction(button,fn){button.disabled=true;button.classList.add('is-loading');button.setAttribute('aria-busy','true');try{await fn();await loadAdmin();}catch(e){alert(e.message);}finally{button.disabled=false;button.classList.remove('is-loading');button.removeAttribute('aria-busy');}}
$('#addEmployeeForm').onsubmit=async event=>{event.preventDefault();try{const result=await api('admin/add-employee',{name:$('#newEmployeeName').value.trim(),code:$('#newEmployeeCode').value.trim()});$('#addEmployeeForm').reset();await loadAdmin();await loadEmployees();$('#addEmployeeMessage').textContent=`${result.name} added. Registration code: ${result.registrationCode}`;}catch(error){$('#addEmployeeMessage').textContent=error.message;}};
$('#attendanceDate').onchange=()=>loadAdmin().catch(e=>alert(e.message));
$('#settingsForm').onsubmit=async e=>{e.preventDefault();const button=$('#settingsForm button');button.disabled=true;try{const office={lat:Number($('#officeLat').value),lng:Number($('#officeLng').value),radius:Number($('#officeRadius').value),enabled:$('#geoEnabled').checked};await api('admin/settings',{office});state.config.office=office;$('#settingsMessage').textContent='Settings saved.';}catch(e){$('#settingsMessage').textContent=e.message;}finally{button.disabled=false;}};
$('#adminLogout').onclick=async()=>{try{await api('admin/logout',{});await openAdmin();}catch(e){alert(e.message);}};
$('#exportCsv').onclick=()=>{const cell=v=>'"'+String(v??'').replace(/^[=+@-]/,"'$&").replaceAll('"','""')+'"';const rows=[['Employee ID','Name','Date (IST)','Check in (IST)','Check out (IST)','Locker'],...state.admin.attendance.map(a=>[a.code,a.name,a.day,time(a.check_in),time(a.check_out),a.locker])];const blob=new Blob(['\ufeff'+rows.map(r=>r.map(cell).join(',')).join('\r\n')],{type:'text/csv;charset=utf-8'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=`R3-Square-attendance-${state.admin.day}.csv`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
$('#today').textContent=new Date().toLocaleDateString('en-IN',{timeZone:'Asia/Kolkata',weekday:'long',day:'numeric',month:'short',year:'numeric'});
async function init(){try{state.config=await api('config');if(page!=='admin')void showPermissions();if(page!=='attendance')await loadEmployees();if(window.location.hash==='#admin'){window.location.replace('/admin');return;}if(page==='admin')await openAdmin();else if(new URLSearchParams(window.location.search).has('registered'))message('Your face is registered. You can now check in.');}catch(e){message(e.message,true);$('#registerEmployee').replaceChildren(new Option('Could not load team — refresh to retry',''));}}
// Keep each action responsive while its camera or network request is starting.
for (const [id,label] of [['attendanceForm','Opening camera…'],['registerForm','Checking registration code…'],['adminAuth','Signing in…'],['settingsForm','Saving…'],['addEmployeeForm','Adding employee…']]) {
 const form=$('#'+id),submit=form.onsubmit;
 form.onsubmit=async event=>{
  event.preventDefault();if(form.dataset.busy==='true')return;
  const button=form.querySelector('button[type=submit]'),original=button.textContent;
  form.dataset.busy='true';form.setAttribute('aria-busy','true');button.classList.add('is-loading');button.disabled=true;button.textContent=label;
  try{await submit(event);}finally{form.dataset.busy='false';form.removeAttribute('aria-busy');button.classList.remove('is-loading');button.disabled=false;button.textContent=original;}
 };
}
init();
