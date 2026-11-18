import { validDescriptor, distance, faceMatch, identifyFace, locationCheck, istDay } from './core.js';
const json = (data,status=200,headers={})=>new Response(JSON.stringify(data),{status,headers:{'Content-Type':'application/json','Cache-Control':'no-store',...headers}});
const fail = (message,status=400)=>{throw Object.assign(new Error(message),{status});};
const now = ()=>Date.now();
async function setting(db,key,fallback=null) {const row=await db.prepare('SELECT value FROM settings WHERE key=?').bind(key).first();return row?JSON.parse(row.value):fallback;}
async function set(db,key,value) {await db.prepare('INSERT INTO settings(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value').bind(key,JSON.stringify(value)).run();}
async function hash(value) {return [...new Uint8Array(await crypto.subtle.digest('SHA-256',new TextEncoder().encode(value)))].map(x=>x.toString(16).padStart(2,'0')).join('');}
async function passwordHash(password,salt) {const key=await crypto.subtle.importKey('raw',new TextEncoder().encode(password),'PBKDF2',false,['deriveBits']);return [...new Uint8Array(await crypto.subtle.deriveBits({name:'PBKDF2',hash:'SHA-256',salt:new TextEncoder().encode(salt),iterations:100000},key,256))].map(x=>x.toString(16).padStart(2,'0')).join('');}
async function equal(a,b) {return await hash(String(a))===await hash(String(b));}
async function rate(db,key,max) {const bucket=Math.floor(now()/600000); const row=await db.prepare('INSERT INTO rate_limits(key,count,expires) VALUES(?,1,?) ON CONFLICT(key) DO UPDATE SET count=count+1 RETURNING count').bind(`${key}:${bucket}`,now()+600000).first();if(row.count>max)fail('Too many attempts. Please try again in 10 minutes.',429);}
async function admin(req,db) {const token=(req.headers.get('Cookie')||'').match(/(?:^|;\s*)r3_session=([^;]+)/)?.[1];if(!token)fail('Please sign in as admin.',401);const s=await db.prepare('SELECT token FROM sessions WHERE token=? AND expires>?').bind(await hash(token),now()).first();if(!s)fail('Your admin session expired. Please sign in.',401);}
async function makeSession(db) {const token=crypto.randomUUID()+crypto.randomUUID();await db.prepare('INSERT INTO sessions VALUES(?,?)').bind(await hash(token),now()+43200000).run();return `r3_session=${token}; HttpOnly; Secure; SameSite=Strict; Path=/; Max-Age=43200`;}
function samples(body,n=2) {if(!Array.isArray(body.descriptors)||body.descriptors.length!==n||!body.descriptors.every(validDescriptor))fail('Face capture was incomplete. Please try again.');if(body.descriptors.some(d=>distance(d,body.descriptors[0])>.5))fail('Face samples did not agree. Keep one person in view and retry.');return body.descriptors;}
async function body(req) {if(Number(req.headers.get('Content-Length'))>32000)fail('Request too large.',413);let raw=await req.text();if(raw.length>32000)fail('Request too large.',413);try{return JSON.parse(raw);}catch{fail('Invalid request.');}}

async function registrationTarget(db,b) {
 const employee=await db.prepare('SELECT id,name,descriptors,registration_code FROM employees WHERE id=? AND active=1').bind(String(b.employeeId||'')).first();
 if(!employee)fail('Choose your name from the employee list.');
 if(JSON.parse(employee.descriptors).length)fail('Your face is already registered. Please use attendance.',409);
 if(!employee.registration_code || !await equal(String(b.registrationCode||'').trim().toUpperCase(),employee.registration_code))fail('Registration code is incorrect for this employee. Ask your admin for your personal code.',403);
 return employee;
}
async function api(req,env,url) {
 const db=env.DB, path=url.pathname, method=req.method;
 if(method!=='GET' && req.headers.get('Origin') && req.headers.get('Origin')!==url.origin)fail('Invalid request origin.',403);
 const ip=req.headers.get('CF-Connecting-IP')||'local';
 if(path==='/api/config'&&method==='GET') {const office=await setting(db,'office');return json({office,setupRequired:!(await setting(db,'admin')),day:istDay()});}
 if(path==='/api/employees'&&method==='GET') {const {results}=await db.prepare('SELECT id,name,code,locker,(json_array_length(descriptors)>0) AS registered FROM employees WHERE active=1 ORDER BY name').all();return json({employees:results});}
 if(path==='/api/register/check'&&method==='POST') {
  await rate(db,`register-check:${ip}`,20);await registrationTarget(db,await body(req));return json({ok:true});
 }
 if(path==='/api/register'&&method==='POST') {
  await rate(db,`register:${ip}`,15);const b=await body(req),employee=await registrationTarget(db,b);
  if(b.consent!==true)fail('Please agree to face verification for attendance.');const descriptors=samples(b,3);
  const result=await db.prepare('UPDATE employees SET descriptors=?,registration_code=NULL WHERE id=? AND active=1 AND json_array_length(descriptors)=0 AND registration_code=?').bind(JSON.stringify(descriptors),employee.id,employee.registration_code).run();
  if(!result.meta.changes)fail('Your registration has already been completed. Use attendance.',409);
  return json({message:'Your face is registered. You can now check in.',id:employee.id},201);
 }
 if(path==='/api/challenge'&&method==='POST') {
  await rate(db,`challenge:${ip}`,40);const b=await body(req);let employeeId='*';
  if(b.employeeId){const employee=await db.prepare('SELECT id FROM employees WHERE id=? AND active=1 AND json_array_length(descriptors)>0').bind(String(b.employeeId)).first();if(!employee)fail('This employee is not registered.');employeeId=employee.id;}
  else{const enrolled=await db.prepare('SELECT id FROM employees WHERE active=1 AND json_array_length(descriptors)>0 LIMIT 1').first();if(!enrolled)fail('No faces are registered yet. Please register first.');}
  const id=crypto.randomUUID();await db.prepare('INSERT INTO challenges VALUES(?,?,?)').bind(id,employeeId,now()+120000).run();return json({challenge:id});
 }
 if(path==='/api/attendance'&&method==='POST') {
  await rate(db,`attendance:${ip}`,40);const b=await body(req);if(!['in','out'].includes(b.action))fail('Choose check in or check out.');const ds=samples(b);
  const c=await db.prepare('DELETE FROM challenges WHERE id=? AND expires>? RETURNING employee_id').bind(String(b.challenge||''),now()).first();if(!c)fail('Your scan expired. Please scan again.');
  let employee;
  if(c.employee_id==='*'){
   const {results}=await db.prepare('SELECT * FROM employees WHERE active=1 AND json_array_length(descriptors)>0').all();
   employee=identifyFace(ds,results.map(e=>({...e,templates:JSON.parse(e.descriptors)})));
   if(!employee)fail('Could not identify your face clearly. Face the camera in good light and try again. If you are new, register first.',403);
  }else{
   employee=await db.prepare('SELECT * FROM employees WHERE id=? AND active=1').bind(c.employee_id).first();if(!employee)fail('Employee is unavailable. Contact admin.');
   if(!faceMatch(ds,JSON.parse(employee.descriptors)))fail('Your face did not match. Improve the lighting and try again.',403);
  }
  const office=await setting(db,'office');const d=locationCheck(b.location,office);const day=istDay(),time=new Date().toISOString();
  if(b.action==='in') {const r=await db.prepare('INSERT INTO attendance(id,employee_id,day,check_in,in_distance) VALUES(?,?,?,?,?) ON CONFLICT(employee_id,day) DO NOTHING').bind(crypto.randomUUID(),employee.id,day,time,d).run();if(!r.meta.changes)fail('You have already checked in today.');}
  else {const r=await db.prepare('UPDATE attendance SET check_out=?,out_distance=? WHERE employee_id=? AND day=? AND check_out IS NULL AND check_in IS NOT NULL').bind(time,d,employee.id,day).run();if(!r.meta.changes)fail('Check in first, or you have already checked out today.');}
  return json({message:b.action==='in'?'Checked in successfully':'Checked out successfully',name:employee.name,time,locker:employee.locker,distance:d});
 }
 if(path==='/api/admin/setup'&&method==='POST') {
  await rate(db,`setup:${ip}`,8);const b=await body(req);if(await setting(db,'admin'))fail('Admin is already set up. Please sign in.',409);
  if(!env.SETUP_TOKEN||!await equal(b.setupCode||'',env.SETUP_TOKEN))fail('Setup code is incorrect.',403);
  if(typeof b.password!=='string'||b.password.length<12||b.password.length>128)fail('Use an admin password of 12–128 characters.');
  const salt=crypto.randomUUID(),value={salt,hash:await passwordHash(b.password,salt)};
  const r=await db.prepare("INSERT OR IGNORE INTO settings(key,value) VALUES('admin',?)").bind(JSON.stringify(value)).run();if(!r.meta.changes)fail('Admin is already set up.',409);
  await set(db,'registrationCode',crypto.randomUUID().slice(0,8).toUpperCase());return json({ok:true},200,{'Set-Cookie':await makeSession(db)});
 }
 if(path==='/api/admin/login'&&method==='POST') {await rate(db,`login:${ip}`,8);const b=await body(req),a=await setting(db,'admin');if(!a||typeof b.password!=='string'||b.password.length>128||!await equal(await passwordHash(b.password,a.salt),a.hash))fail('Password is incorrect.',401);return json({ok:true},200,{'Set-Cookie':await makeSession(db)});}
 if(path.startsWith('/api/admin/')) {
  await admin(req,db);
  if(path==='/api/admin/password'&&method==='POST') {
   await rate(db,`password:${ip}`,8);const b=await body(req),a=await setting(db,'admin');
   if(!a||typeof b.currentPassword!=='string'||b.currentPassword.length>128||!await equal(await passwordHash(b.currentPassword,a.salt),a.hash))fail('Current password is incorrect.',403);
   if(typeof b.newPassword!=='string'||b.newPassword.length<12||b.newPassword.length>128)fail('Use a new password of 12–128 characters.');
   if(b.newPassword!==b.confirmPassword)fail('New passwords do not match.');
   if(await equal(b.currentPassword,b.newPassword))fail('Choose a different password from your current one.');
   const salt=crypto.randomUUID(),updated={salt,hash:await passwordHash(b.newPassword,salt)};
   const result=await db.batch([db.prepare("UPDATE settings SET value=? WHERE key='admin' AND value=?").bind(JSON.stringify(updated),JSON.stringify(a)),db.prepare('DELETE FROM sessions')]);
   if(!result[0].meta.changes)fail('The password changed in another session. Please sign in again.',409);
   return json({ok:true},200,{'Set-Cookie':'r3_session=; HttpOnly; Secure; SameSite=Strict; Path=/; Max-Age=0'});
  }
  if(path==='/api/admin/logout'&&method==='POST') {const token=(req.headers.get('Cookie')||'').match(/(?:^|;\s*)r3_session=([^;]+)/)?.[1];await db.prepare('DELETE FROM sessions WHERE token=?').bind(await hash(token)).run();return json({ok:true},200,{'Set-Cookie':'r3_session=; HttpOnly; Secure; SameSite=Strict; Path=/; Max-Age=0'});}
  if(path==='/api/admin/data'&&method==='GET') {const day=url.searchParams.get('day')||istDay();if(!/^\d{4}-\d{2}-\d{2}$/.test(day))fail('Invalid date.');const {results:employees}=await db.prepare('SELECT id,name,code,locker,active,created_at,registration_code,(json_array_length(descriptors)>0) AS registered FROM employees ORDER BY name').all();const {results:attendance}=await db.prepare('SELECT a.*,e.name,e.code,e.locker FROM attendance a JOIN employees e ON e.id=a.employee_id WHERE day=? ORDER BY check_in DESC').bind(day).all();return json({employees,attendance,office:await setting(db,'office'),day});}
  if(path==='/api/admin/settings'&&method==='POST') {const b=await body(req),o=b.office;if(!o||![o.lat,o.lng,o.radius].every(Number.isFinite)||Math.abs(o.lat)>90||Math.abs(o.lng)>180||o.radius<50||o.radius>2000||typeof o.enabled!=='boolean')fail('Enter valid office coordinates and a radius between 50 and 2,000 m.');await set(db,'office',o);return json({ok:true});}
  if(path==='/api/admin/add-employee'&&method==='POST') {
   const b=await body(req),name=String(b.name||'').trim();let code=String(b.code||'').trim().toUpperCase();
   if(name.length<2||name.length>60)fail('Enter an employee name between 2 and 60 characters.');
   if(code&&!/^[A-Z0-9-]{2,20}$/.test(code))fail('Employee ID must use 2–20 letters, numbers or hyphens.');
   const duplicate=await db.prepare('SELECT id FROM employees WHERE lower(name)=lower(?) AND active=1').bind(name).first();if(duplicate)fail('That employee already exists.',409);
   if(!code){const {results}=await db.prepare('SELECT code FROM employees').all();const taken=new Set(results.map(e=>e.code));for(let i=1;i<=99;i++){const next='R3-'+String(i).padStart(2,'0');if(!taken.has(next)){code=next;break;}}if(!code)fail('Please enter an employee ID.');}
   const id=crypto.randomUUID(),registrationCode=crypto.randomUUID().slice(0,8).toUpperCase();
   try{const r=await db.prepare("INSERT INTO employees(id,code,name,descriptors,created_at,registration_code) SELECT ?,?,?,'[]',?,? WHERE (SELECT COUNT(*) FROM employees WHERE active=1)<5").bind(id,code,name,new Date().toISOString(),registrationCode).run();if(!r.meta.changes)fail('All five active employee places are filled.');}catch(e){if(String(e.message).includes('UNIQUE'))fail('That employee ID already exists.',409);throw e;}
   return json({id,name,code,registrationCode},201);
  }
  if(path==='/api/admin/registration-code'&&method==='POST') {
   const b=await body(req),code=crypto.randomUUID().slice(0,8).toUpperCase();const r=await db.prepare('UPDATE employees SET registration_code=? WHERE id=? AND json_array_length(descriptors)=0').bind(code,String(b.id||'')).run();if(!r.meta.changes)fail('This employee is already registered or unavailable.');return json({registrationCode:code});
  }
  if(path==='/api/admin/employee'&&method==='POST') {const b=await body(req);if(typeof b.locker!=='string'||b.locker.length>30||typeof b.active!=='boolean')fail('Invalid employee details.');if(b.locker.trim()&&b.active){const other=await db.prepare('SELECT id FROM employees WHERE locker=? AND active=1 AND id<>?').bind(b.locker.trim(),b.id).first();if(other)fail('That locker is assigned to another employee.');}if(b.active){const count=await db.prepare('SELECT COUNT(*) AS n FROM employees WHERE active=1 AND id<>?').bind(b.id).first();if(count.n>=5)fail('Only five active employees are allowed.');}await db.prepare('UPDATE employees SET locker=?,active=? WHERE id=?').bind(b.locker.trim(),b.active?1:0,String(b.id)).run();return json({ok:true});}
  if(path==='/api/admin/delete-employee'&&method==='POST') {const b=await body(req);await db.batch([db.prepare('DELETE FROM attendance WHERE employee_id=?').bind(b.id),db.prepare('DELETE FROM challenges WHERE employee_id=?').bind(b.id),db.prepare('DELETE FROM employees WHERE id=?').bind(b.id)]);return json({ok:true});}
 }
 return json({error:'Not found'},404);
}
export default {
 async fetch(req,env) {const url=new URL(req.url);let response;
 try{response=url.pathname.startsWith('/api/')?await api(req,env,url):await env.ASSETS.fetch(req);}catch(e){if(!e.status&&!/Allow location|Location is|location is|Your location|You are|Invalid location/.test(e.message))console.error('Request failed',e.message);response=json({error:e.status||/location|Location|You are/.test(e.message)?e.message:'Something went wrong. Please try again.'},e.status||(/location|Location|You are/.test(e.message)?400:500));}
 const headers=new Headers(response.headers);headers.set('X-Content-Type-Options','nosniff');headers.set('Referrer-Policy','same-origin');headers.set('Permissions-Policy','camera=(self), geolocation=(self), microphone=()');headers.set('X-Frame-Options','DENY');headers.set('Content-Security-Policy',"default-src 'self'; script-src 'self' 'unsafe-eval'; style-src 'self'; img-src 'self' data: blob:; connect-src 'self'; media-src 'self' blob:; worker-src 'self' blob:; object-src 'none'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'");return new Response(response.body,{status:response.status,headers});
 },
 async scheduled(event,env) {await env.DB.batch([env.DB.prepare('DELETE FROM challenges WHERE expires<?').bind(now()),env.DB.prepare('DELETE FROM sessions WHERE expires<?').bind(now()),env.DB.prepare('DELETE FROM rate_limits WHERE expires<?').bind(now())]);}
};
