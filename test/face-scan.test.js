import test from 'node:test';
import assert from 'node:assert/strict';
import {collectFaceSamples} from '../public/face-scan.js';
const face=(value=.1,width=160)=>({descriptor:Array(128).fill(value),width});
function harness(frames,options={}){let n=0,t=0;const reports=[];return {reports,run:()=>collectFaceSamples({count:2,detect:async()=>frames[Math.min(n++,frames.length-1)],isActive:()=>true,report:(...r)=>reports.push(r),pause:async ms=>{t+=ms;},clock:()=>t,timeoutMs:2000,...options})};}
 test('automatically waits through empty and distant frames before collecting two samples',async()=>{const h=harness([[],[face(.1,60)],[face()],[face()]]);assert.equal((await h.run()).length,2);assert(h.reports.some(r=>r[0].includes('closer')));});
 test('multiple people discard partial captures',async()=>{const h=harness([[face()],[face(),face(.3)],[face(.2)],[face(.2)]]);const result=await h.run();assert.equal(result[0][0],.2);assert.equal(result[1][0],.2);});
 test('switching faces resets samples so identities cannot be mixed',async()=>{const result=await harness([[face(.1)],[face(.3)],[face(.3)]]).run();assert.equal(result[0][0],.3);assert.equal(result[1][0],.3);});
 test('closing scanner cancels in-flight detection without returning templates',async()=>{let active=true;const h=harness([[face()]],{isActive:()=>active,detect:async()=>{active=false;return [face()];}});assert.equal(await h.run(),null);});
 test('empty frames time out instead of looping forever',async()=>{await assert.rejects(harness([[]]).run(),/No clear face/);});
