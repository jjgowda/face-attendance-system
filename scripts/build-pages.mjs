import {readFile,writeFile} from 'node:fs/promises';
const template=await readFile(new URL('../src/page.html',import.meta.url),'utf8');
const pages=[
 {file:'index.html',page:'attendance',title:'Attendance',badge:'Attendance',heading:'Attendance',description:'Record your check-in or check-out.'},
 {file:'attendance.html',page:'attendance',title:'Attendance',badge:'Attendance',heading:'Attendance',description:'Record your check-in or check-out.'},
 {file:'register.html',page:'register',title:'Register your face',badge:'Registration',heading:'Face registration',description:'Set up face verification for your attendance.'},
 {file:'admin.html',page:'admin',title:'Admin',badge:'Admin',heading:'',description:''}
];
for(const config of pages){
 let html=template;
 for(const [key,value] of Object.entries(config))html=html.replaceAll('{{'+key.toUpperCase()+'}}',value);
 html=html.replace('{{FACE_SCRIPT}}',config.page==='admin'?'':'<script defer src="/vendor/face-api.js"></script>');
 if(config.page==='register')html=html.replace('<form id="attendanceForm">','<form id="attendanceForm" hidden>').replace('<form id="registerForm" hidden>','<form id="registerForm">');
 if(config.page==='admin')html=html.replace('<section id="employeeView" class="workspace">','<section id="employeeView" class="workspace" hidden>').replace('<section id="adminView" hidden>','<section id="adminView">');
 await writeFile(new URL('../public/'+config.file,import.meta.url),html);
}
console.log('Built attendance, registration and admin pages.');
