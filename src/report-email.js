import {dailyReport} from './reports.js';
import {istDay} from './core.js';
export const REPORT_RECIPIENTS=['senthil@r3square.com','jeevan@r3square.com'];
export function reportSlot(cron){return cron==='30 6 * * *'?'12:00 PM':cron==='30 14 * * *'?'8:00 PM':null;}
const escape=value=>String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const duration=minutes=>minutes===null?'—':`${Math.floor(minutes/60)}h ${String(minutes%60).padStart(2,'0')}m`;
const clock=new Intl.DateTimeFormat('en-IN',{timeZone:'Asia/Kolkata',hour:'2-digit',minute:'2-digit'});
const formatTime=value=>value?clock.format(new Date(value)):'—';
export function buildDailyEmail(report,generatedAt,slot){
 const date=new Date(report.date+'T12:00:00+05:30').toLocaleDateString('en-IN',{timeZone:'Asia/Kolkata',day:'numeric',month:'short',year:'numeric'});
 const asOf=new Date(generatedAt).toLocaleString('en-IN',{timeZone:'Asia/Kolkata',hour:'2-digit',minute:'2-digit',day:'numeric',month:'short'});
 const rows=report.rows.map(row=>[row.name,row.status,formatTime(row.checkIn),formatTime(row.checkOut),duration(row.minutes)]);
 const heading=['Employee','Status','Check in (IST)','Check out (IST)','Completed hours'];
 const text=`R3 Square — Daily attendance\n${date} · ${slot} IST\nSnapshot as of ${asOf} IST\n\nChecked in: ${report.summary.checkedIn}\nCompleted: ${report.summary.completed}\nNo check-in: ${report.summary.noCheckIn}\nCompleted hours: ${duration(report.summary.totalMinutes)}\n\n`+[heading,...rows].map(row=>row.join(' | ')).join('\n')+'\n\nNo check-in means no scan was recorded; it does not mark absence. Open entries are excluded from completed hours.\nAdmin: https://employee.r3square.com/admin';
 const html=`<div style="font-family:Arial,sans-serif;color:#172b4d;max-width:760px;margin:auto"><h2>R3 Square — Daily attendance</h2><p>${escape(date)} · ${escape(slot)} IST</p><p style="color:#68788e">Snapshot as of ${escape(asOf)} IST</p><p><b>Checked in:</b> ${report.summary.checkedIn} &nbsp; <b>Completed:</b> ${report.summary.completed} &nbsp; <b>No check-in:</b> ${report.summary.noCheckIn}</p><p><b>Completed hours:</b> ${escape(duration(report.summary.totalMinutes))}</p><table style="border-collapse:collapse;width:100%;font-size:13px"><thead><tr>${heading.map(value=>`<th style="padding:10px;text-align:left;background:#eef4fc;border:1px solid #dfe6f0">${escape(value)}</th>`).join('')}</tr></thead><tbody>${rows.map(row=>`<tr>${row.map(value=>`<td style="padding:10px;border:1px solid #dfe6f0">${escape(value)}</td>`).join('')}</tr>`).join('')}</tbody></table><p style="color:#68788e;font-size:12px">No check-in means no scan was recorded; it does not mark absence. Hours include completed check-ins only.</p><p><a href="https://employee.r3square.com/admin">Open admin reports</a></p></div>`;
 const csvCell=value=>{let v=String(value??'');if(/^\s*[=+@-]/.test(v))v="'"+v;return '"'+v.replaceAll('"','""')+'"';};
 const csv='\ufeff'+[heading,...rows].map(row=>row.map(csvCell).join(',')).join('\r\n');
 return {subject:`R3 Square attendance — ${date} — ${slot} IST`,text,html,csv};
}
export async function sendDailyReport(env,{slot,scheduledTime=Date.now(),manual=false}={}){
 if(!env.REPORT_SENDER||!env.REPORT_EMAIL)return {ok:false,pending:true,message:'Email sender setup is pending. Recipient verification and sender-domain approval are required.'};
 const day=istDay(new Date(scheduledTime)),id=manual?'manual:'+crypto.randomUUID():day+':'+slot;
 const claim=await env.DB.prepare("INSERT OR IGNORE INTO report_email_log(id,report_day,slot,recipients,status,attempted_at) VALUES(?,?,?,?,'sending',?)").bind(id,day,slot,JSON.stringify(REPORT_RECIPIENTS),new Date().toISOString()).run();
 if(!claim.meta.changes)return {ok:true,duplicate:true};
 try{
  const [{results:employees},{results:records}]=await Promise.all([env.DB.prepare('SELECT id,name,code,active,created_at,(json_array_length(descriptors)>0) AS registered FROM employees ORDER BY name').all(),env.DB.prepare('SELECT employee_id,day,check_in,check_out FROM attendance WHERE day=?').bind(day).all()]);
  const email=buildDailyEmail(dailyReport(employees,records,day,day),scheduledTime,slot);
  const bytes=new TextEncoder().encode(email.csv);let binary='';for(const byte of bytes)binary+=String.fromCharCode(byte);
  await env.REPORT_EMAIL.send({from:{email:env.REPORT_SENDER,name:'R3 Square Attendance'},to:REPORT_RECIPIENTS,subject:email.subject,text:email.text,html:email.html,attachments:[{filename:`R3-Square-attendance-${day}.csv`,type:'text/csv',disposition:'attachment',content:btoa(binary)}]});
  await env.DB.prepare("UPDATE report_email_log SET status='sent',accepted_at=?,error=NULL WHERE id=?").bind(new Date().toISOString(),id).run();return {ok:true,message:'Report accepted for delivery to both recipients.'};
 }catch(error){const reason=String(error.code||'EMAIL_ERROR')+': '+String(error.message||'Email sending failed.');await env.DB.prepare("UPDATE report_email_log SET status='failed',error=? WHERE id=?").bind(reason.slice(0,500),id).run();return {ok:false,message:reason};}
}
