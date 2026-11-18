export function validDescriptor(d) { return Array.isArray(d) && d.length === 128 && d.every(v => typeof v === 'number' && Number.isFinite(v) && Math.abs(v) <= 2); }
export function distance(a,b) { return Math.sqrt(a.reduce((s,v,i) => s + (v-b[i])**2,0)); }
export function faceMatch(samples, enrolled, threshold = 0.5) {
  return samples.length >= 2 && samples.every(s => validDescriptor(s) && Math.min(...enrolled.map(e=>distance(s,e))) <= threshold);
}
export function identifyFace(samples, candidates, threshold=.5, margin=.05) {
 if(samples.length<2 || !samples.every(validDescriptor))return null;
 const ranked=candidates.filter(e=>e.active!==0 && Array.isArray(e.templates) && e.templates.length && e.templates.every(validDescriptor)).map(employee=>({employee,score:Math.max(...samples.map(sample=>Math.min(...employee.templates.map(template=>distance(sample,template)))))})).sort((a,b)=>a.score-b.score);
 if(!ranked.length || ranked[0].score>threshold)return null;
 if(ranked[1] && ranked[1].score-ranked[0].score<margin)return null;
 return ranked[0].employee;
}
export function metres(lat1,lng1,lat2,lng2) {
 const r = Math.PI/180, a = Math.sin((lat2-lat1)*r/2)**2 + Math.cos(lat1*r)*Math.cos(lat2*r)*Math.sin((lng2-lng1)*r/2)**2;
 return 6371000 * 2 * Math.atan2(Math.sqrt(a),Math.sqrt(1-a));
}
export function locationCheck(location, office, now = Date.now()) {
 if (!office.enabled) return null;
 if (!location || ![location.lat,location.lng,location.accuracy,location.timestamp].every(Number.isFinite)) throw Error('Allow location access and try again.');
 if (Math.abs(location.lat)>90 || Math.abs(location.lng)>180 || location.accuracy<0) throw Error('Invalid location. Try again.');
 if (now-location.timestamp>120000 || location.timestamp>now+30000) throw Error('Your location is out of date. Please try again.');
 if (location.accuracy>150) throw Error('Location is not accurate enough. Move near a window or outside and retry.');
 const d = metres(office.lat,office.lng,location.lat,location.lng);
 if (d>office.radius) throw Error(`You are ${Math.round(d)} m from R3 Square. Please check in within ${office.radius} m of the office.`);
 return Math.round(d);
}
export function istDay(date = new Date()) { return new Intl.DateTimeFormat('en-CA',{timeZone:'Asia/Kolkata',year:'numeric',month:'2-digit',day:'2-digit'}).format(date); }
