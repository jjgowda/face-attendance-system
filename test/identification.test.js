import test from 'node:test';
import assert from 'node:assert/strict';
import {identifyFace} from '../src/core.js';
const face=v=>Array(128).fill(v),person=(id,v,active=1)=>({id,active,templates:[face(v)]});
test('identifies the correct employee without a selected name',()=>assert.equal(identifyFace([face(.4),face(.4)],[person('A',.1),person('B',.4)]).id,'B'));
test('rejects unknown faces, mixed samples and incomplete captures',()=>{const staff=[person('A',.1),person('B',.4)];assert.equal(identifyFace([face(.8),face(.8)],staff),null);assert.equal(identifyFace([face(.1),face(.4)],staff),null);assert.equal(identifyFace([face(.1)],staff),null);});
test('rejects ambiguous matches rather than assigning the wrong employee',()=>assert.equal(identifyFace([face(.1),face(.1)],[person('A',.1),person('B',.101)]),null));
test('does not identify inactive or unregistered employees',()=>assert.equal(identifyFace([face(.1),face(.1)],[person('A',.1,0),{id:'pending',templates:[]}]),null));
