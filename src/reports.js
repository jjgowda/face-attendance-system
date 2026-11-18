import {istDay} from './core.js';
export function validReportDay(value){if(!/^\d{4}-\d{2}-\d{2}$/.test(value))return false;const d=new Date(value+'T00:00:00Z');return Number.isFinite(+d)&&d.toISOString().slice(0,10)===value&&+value.slice(0,4)>=1900&&+value.slice(0,4)<=2100;}
export function monthRange(month){if(!/^\d{4}-(0[1-9]|1[0-2])$/.test(month)||+month.slice(0,4)<1900||+month.slice(0,4)>2100)throw Error('Invalid month.');const [year,m]=month.split('-').map(Number);return {start:month+'-01',end:new Date(Date.UTC(year,m,1)).toISOString().slice(0,10),days:new Date(Date.UTC(year,m,0)).getUTCDate()};}
export function dayRecord(employee,record,day,today){
 let status,minutes=null;const created=employee.created_at&&Number.isFinite(Date.parse(employee.created_at))?new Date(Date.parse(employee.created_at)+19800000).toISOString().slice(0,10):null;
 if(record){const start=Date.parse(record.check_in),end=record.check_out?Date.parse(record.check_out):null;
  if(!Number.isFinite(start)||(record.check_out&&(!Number.isFinite(end)||end<start)))status='Needs review';
  else if(record.check_out){status='Completed';minutes=Math.floor((end-start)/60000);}
  else status=day===today?'Checked in':'Missing checkout';
 }else if(day>today)status='Upcoming';
 else if(created&&day<created)status='Not on roster';
 else if(!employee.registered)status='Not registered';
 else status='No check-in';
 return {employeeId:employee.id,name:employee.name,code:employee.code,active:!!employee.active,day,status,checkIn:record?.check_in||null,checkOut:record?.check_out||null,minutes};
}
export function dailyReport(employees,records,day,today){const map=new Map(records.map(r=>[r.employee_id,r]));const rows=employees.filter(e=>e.active||map.has(e.id)).map(e=>dayRecord(e,map.get(e.id),day,today));return {mode:'daily',date:day,today,rows,summary:{employees:rows.length,checkedIn:rows.filter(r=>r.checkIn).length,completed:rows.filter(r=>r.status==='Completed').length,open:rows.filter(r=>r.status==='Checked in'||r.status==='Missing checkout').length,noCheckIn:rows.filter(r=>r.status==='No check-in').length,notRegistered:rows.filter(r=>r.status==='Not registered').length,totalMinutes:rows.reduce((s,r)=>s+(r.minutes||0),0)}};}
export function monthlyReport(employees,records,month,today,employeeId=''){
 const range=monthRange(month),withRecords=new Set(records.map(r=>r.employee_id));const roster=employees.filter(e=>e.active||withRecords.has(e.id)||e.id===employeeId),recordMap=new Map(records.map(r=>[r.employee_id+':'+r.day,r]));
 const summaries=roster.map(employee=>{const days=Array.from({length:range.days},(_,i)=>{const day=month+'-'+String(i+1).padStart(2,'0');return dayRecord(employee,recordMap.get(employee.id+':'+day),day,today);});return {employeeId:employee.id,name:employee.name,code:employee.code,active:!!employee.active,presentDays:days.filter(r=>r.checkIn).length,completedDays:days.filter(r=>r.status==='Completed').length,missingCheckout:days.filter(r=>r.status==='Missing checkout').length,openDays:days.filter(r=>r.status==='Checked in').length,noCheckIn:days.filter(r=>r.status==='No check-in').length,totalMinutes:days.reduce((s,r)=>s+(r.minutes||0),0),days};});
 const selected=employeeId?summaries.find(e=>e.employeeId===employeeId):null,visible=selected?[selected]:summaries;
 return {mode:'monthly',month,today,employees:employees.map(e=>({id:e.id,name:e.name,code:e.code,active:!!e.active})),summary:{employees:visible.length,presentDays:visible.reduce((s,e)=>s+e.presentDays,0),completedDays:visible.reduce((s,e)=>s+e.completedDays,0),missingCheckout:visible.reduce((s,e)=>s+e.missingCheckout,0),noCheckIn:visible.reduce((s,e)=>s+e.noCheckIn,0),totalMinutes:visible.reduce((s,e)=>s+e.totalMinutes,0)},rows:summaries.map(({days,...e})=>e),selected:selected||null};
}
