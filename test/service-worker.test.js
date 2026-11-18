import test from 'node:test';
import assert from 'node:assert/strict';
import vm from 'node:vm';
import {readFileSync} from 'node:fs';
function handler(){const callbacks={};vm.runInNewContext(readFileSync(new URL('../public/sw.js',import.meta.url),'utf8'),{URL,self:{location:{origin:'https://example.com'},addEventListener:(name,fn)=>callbacks[name]=fn}});return callbacks.fetch;}
test('service worker never intercepts or queues API GETs or attendance POSTs',()=>{const fetch=handler();for(const [method,path] of [['GET','/api/employees'],['POST','/api/attendance'],['POST','/api/register']]){let intercepted=false;fetch({request:{method,url:'https://example.com'+path},respondWith:()=>intercepted=true});assert.equal(intercepted,false);}});
