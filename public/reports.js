export function createReports({request,today}){
 const $=s=>document.querySelector(s),panel=$('#reportsPanel');
 const state={mode:'daily',date:today,month:today.slice(0,7),employee:'',data:null,sequence:0};
 const el=(tag,value='',className='')=>{const node=document.createElement(tag);node.textContent=value;if(className)node.className=className;return node;};
 const dateLabel=day=>new Date(day+'T12:00:00+05:30').toLocaleDateString('en-IN',{timeZone:'Asia/Kolkata',day:'numeric',month:'short',year:'numeric'});
 const clock=value=>value?new Date(value).toLocaleTimeString('en-IN',{timeZone:'Asia/Kolkata',hour:'2-digit',minute:'2-digit'}):'—';
 const duration=minutes=>minutes===null?'—':`${Math.floor(minutes/60)}h ${String(minutes%60).padStart(2,'0')}m`;
 const slug=status=>status.toLowerCase().replaceAll(' ','-');
 function cell(row,label,value,className=''){const td=el('td','',className);td.dataset.label=label;td.append(typeof value==='string'?document.createTextNode(value):value);row.append(td);return td;}
 function person(name,code){const wrap=el('div','','report-person');wrap.append(el('strong',name),el('small',code));return wrap;}
 function badge(status){return el('span',status,'status-badge status-'+slug(status));}
 function empty(tbody,columns,message){tbody.replaceChildren();const row=el('tr'),td=el('td',message,'empty report-empty');td.colSpan=columns;row.append(td);tbody.append(row);}
 function dayRow(record,employee=true){const row=el('tr');cell(row,employee?'Employee':'Date',employee?person(record.name,record.code):dateLabel(record.day),'person-cell');cell(row,'Status',badge(record.status),'status-cell');cell(row,'Check in',clock(record.checkIn));cell(row,'Check out',clock(record.checkOut));cell(row,'Completed hours',duration(record.minutes));return row;}
 function summaries(data){const summary=data.summary,monthly=data.mode==='monthly',values=monthly?[
  [state.employee?'Days checked in':'Employee check-in days',String(summary.presentDays),state.employee?'Selected employee':'Across all employees'],
  ['Completed hours',duration(summary.totalMinutes),'Open entries are excluded'],
  ['Missing checkout',String(summary.missingCheckout),'Past days without an exit scan'],
  ['No check-in',String(summary.noCheckIn),'No scan recorded; not absence']
 ]:[
  ['Checked in',String(summary.checkedIn),`${summary.employees} employees in this report`],
  ['Completed',String(summary.completed),`${summary.open} open / missing checkout`],
  ['No check-in',String(summary.noCheckIn),summary.notRegistered?`${summary.notRegistered} awaiting registration`:'No scan recorded; not absence'],
  ['Completed hours',duration(summary.totalMinutes),'Open entries are excluded']
 ];const target=$('#reportSummary');target.replaceChildren();for(const [label,value,note]of values){const card=el('div','','report-stat');card.append(el('small',label),el('strong',value),el('span',note));target.append(card);}}
 function calendar(data){const target=$('#employeeCalendar');target.replaceChildren();for(const day of ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'])target.append(el('span',day,'calendar-weekday'));const offset=(new Date(data.month+'-01T00:00:00Z').getUTCDay()+6)%7;for(let i=0;i<offset;i++){const space=el('span','','calendar-spacer');space.setAttribute('aria-hidden','true');target.append(space);}for(const record of data.selected.days){const button=el('button','','calendar-day status-'+slug(record.status));button.type='button';button.append(el('strong',String(Number(record.day.slice(-2)))),el('small',record.minutes!==null?duration(record.minutes):record.status==='Upcoming'?'':record.status==='Completed'?'✓':record.status==='No check-in'?'—':record.status==='Not on roster'?'':record.status==='Not registered'?'—':'●'));button.setAttribute('aria-label',`${dateLabel(record.day)}: ${record.status}. Open daily report.`);button.title=record.status;if(record.day===today)button.classList.add('calendar-today');button.onclick=()=>{state.date=record.day;$('#reportDate').value=record.day;setMode('daily');};target.append(button);}}
 function render(data){summaries(data);if(data.mode==='daily'){
  $('#reportPeriod').textContent=dateLabel(data.date);const tbody=$('#dailyReportRows');tbody.replaceChildren();if(!data.rows.length)empty(tbody,5,'No employees are available for this date.');else for(const row of data.rows)tbody.append(dayRow(row));
 }else{
  const monthName=new Date(data.month+'-01T12:00:00+05:30').toLocaleDateString('en-IN',{timeZone:'Asia/Kolkata',month:'long',year:'numeric'});$('#reportPeriod').textContent=monthName+(data.selected?' · '+data.selected.name:' · All employees');
  const select=$('#reportEmployee');select.replaceChildren(new Option('All employees',''));for(const e of data.employees)select.add(new Option(e.name+(e.active?'':' (inactive)'),e.id));select.value=state.employee;
  $('#monthOverview').hidden=!!data.selected;$('#employeeMonthDetail').hidden=!data.selected;
  const tbody=$('#monthlyReportRows');tbody.replaceChildren();if(!data.rows.length)empty(tbody,6,'No employee records for this month.');else for(const e of data.rows){const row=el('tr');cell(row,'Employee',person(e.name,e.code),'person-cell');cell(row,'Days checked in',String(e.presentDays));cell(row,'Completed days',String(e.completedDays));cell(row,'Missing checkout',String(e.missingCheckout));cell(row,'Completed hours',duration(e.totalMinutes));const view=el('button','View month','secondary');view.type='button';view.setAttribute('aria-label','View month for '+e.name);view.onclick=()=>{state.employee=e.employeeId;void refresh();};cell(row,'',view,'row-action');tbody.append(row);}
  if(data.selected){$('#employeeMonthName').textContent=data.selected.name+' · '+data.selected.code;calendar(data);const rows=$('#employeeMonthRows');rows.replaceChildren();for(const day of data.selected.days)rows.append(dayRow(day,false));}
 }}
 async function refresh(){const sequence=++state.sequence;panel.setAttribute('aria-busy','true');$('#reportError').hidden=true;$('#reportLoading').hidden=false;$('#reportExport').disabled=true;$('#reportPrint').disabled=true;
  try{const query=state.mode==='daily'?`mode=daily&date=${encodeURIComponent(state.date)}`:`mode=monthly&month=${encodeURIComponent(state.month)}&employee=${encodeURIComponent(state.employee)}`;const data=await request('admin/reports?'+query);if(sequence!==state.sequence)return;state.data=data;render(data);$('#reportExport').disabled=false;$('#reportPrint').disabled=false;
  }catch(error){if(sequence!==state.sequence)return;if(error.status===404&&state.employee){state.employee='';return refresh();}state.data=null;$('#reportSummary').replaceChildren();for(const selector of ['#dailyReportRows','#monthlyReportRows','#employeeMonthRows'])$(selector).replaceChildren();$('#employeeMonthDetail').hidden=true;$('#reportError').textContent=error.message||'Could not load this report. Please retry.';$('#reportError').hidden=false;
  }finally{if(sequence===state.sequence){panel.setAttribute('aria-busy','false');$('#reportLoading').hidden=true;}}
 }
 function setMode(mode){state.mode=mode;const daily=mode==='daily';$('#dailyFilters').hidden=!daily;$('#monthlyFilters').hidden=daily;$('#dailyReportView').hidden=!daily;$('#monthlyReportView').hidden=daily;$('#dailyReportTab').setAttribute('aria-selected',String(daily));$('#monthlyReportTab').setAttribute('aria-selected',String(!daily));return refresh();}
 $('#reportDate').value=state.date;$('#reportMonth').value=state.month;$('#dailyReportTab').onclick=()=>setMode('daily');$('#monthlyReportTab').onclick=()=>setMode('monthly');
 for(const id of ['dailyReportTab','monthlyReportTab'])$('#'+id).onkeydown=event=>{if(!['ArrowLeft','ArrowRight'].includes(event.key))return;event.preventDefault();const target=id==='dailyReportTab'?'monthly':'daily';$('#'+target+'ReportTab').focus();void setMode(target);};
 $('#reportDate').onchange=()=>{if(!$('#reportDate').value){$('#reportDate').value=state.date;return;}state.date=$('#reportDate').value;void refresh();};
 function moveDay(amount){const day=new Date(state.date+'T12:00:00Z');day.setUTCDate(day.getUTCDate()+amount);state.date=day.toISOString().slice(0,10);$('#reportDate').value=state.date;void refresh();}
 $('#previousReportDay').onclick=()=>moveDay(-1);$('#nextReportDay').onclick=()=>moveDay(1);$('#reportToday').onclick=()=>{state.date=today;$('#reportDate').value=today;void refresh();};
 $('#reportMonth').onchange=()=>{if(!$('#reportMonth').value){$('#reportMonth').value=state.month;return;}state.month=$('#reportMonth').value;void refresh();};$('#reportEmployee').onchange=()=>{state.employee=$('#reportEmployee').value;void refresh();};
 $('#reportPrint').onclick=()=>window.print();
 $('#reportExport').onclick=()=>{const data=state.data;if(!data)return;let rows,filename;
  if(data.mode==='daily'){rows=[['Date (IST)','Employee ID','Employee','Status','Check in (IST)','Check out (IST)','Completed minutes','Completed hours'],...data.rows.map(r=>[r.day,r.code,r.name,r.status,clock(r.checkIn),clock(r.checkOut),r.minutes??'',duration(r.minutes)])];filename='R3-Square-daily-'+data.date;}
  else if(data.selected){const e=data.selected;rows=[['Date (IST)','Employee ID','Employee','Status','Check in (IST)','Check out (IST)','Completed minutes','Completed hours'],...e.days.map(r=>[r.day,e.code,e.name,r.status,clock(r.checkIn),clock(r.checkOut),r.minutes??'',duration(r.minutes)])];filename='R3-Square-monthly-'+e.code+'-'+data.month;}
  else{rows=[['Month','Employee ID','Employee','Days checked in','Completed days','Missing checkout','No check-in','Completed minutes','Completed hours'],...data.rows.map(e=>[data.month,e.code,e.name,e.presentDays,e.completedDays,e.missingCheckout,e.noCheckIn,e.totalMinutes,duration(e.totalMinutes)])];filename='R3-Square-monthly-'+data.month;}
  const cell=value=>{let v=String(value??'');if(/^\s*[=+@-]/.test(v))v="'"+v;return '"'+v.replaceAll('"','""')+'"';},url=URL.createObjectURL(new Blob(['\ufeff'+rows.map(row=>row.map(cell).join(',')).join('\r\n')],{type:'text/csv;charset=utf-8'})),link=document.createElement('a');link.href=url;link.download=filename+'.csv';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
 };
 return {refresh};
}
