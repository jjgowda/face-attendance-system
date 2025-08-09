# app.py
from datetime import datetime, timezone, timedelta
from pathlib import Path
import os, io, csv, sys, re
from typing import Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client

import cv2, numpy as np, face_recognition


# ──────────────────────────────────────────────────────────────
# 1) Environment & Supabase
# ──────────────────────────────────────────────────────────────
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")  # server: prefer service role key
if not SUPABASE_URL or not SUPABASE_KEY:
    print("❌ Missing SUPABASE_URL or SUPABASE_KEY in .env", file=sys.stderr)
    raise SystemExit(1)
sb: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


# ──────────────────────────────────────────────────────────────
# 2) FastAPI + CORS
# ──────────────────────────────────────────────────────────────
app = FastAPI(title="Face-Attendance-MVP")

# open CORS for dev; tighten for prod
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent

# IST helpers
IST = timezone(timedelta(hours=5, minutes=30))
def now_ist():
    return datetime.now(IST)
def today_ist_str():
    return now_ist().date().isoformat()

UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}$"
)


# ──────────────────────────────────────────────────────────────
# 3) Students index (id ↔ roll_no) + face store
# ──────────────────────────────────────────────────────────────
KNOWN_DIR = BASE_DIR / "known_faces"
KNOWN_DIR.mkdir(parents=True, exist_ok=True)
TOLERANCE = 0.60

known_encs: List[np.ndarray] = []
# labels correspond to filename stems (uuid OR roll_no)
known_labels: List[str] = []

id_by_roll: Dict[str, str] = {}     # roll_no -> id (uuid)
roll_by_id: Dict[str, str] = {}     # id (uuid) -> roll_no

def refresh_student_index():
    global id_by_roll, roll_by_id
    id_by_roll, roll_by_id = {}, {}
    try:
        resp = sb.table("students").select("id, roll_no").execute()
        for row in resp.data or []:
            sid = row.get("id")
            rno = row.get("roll_no")
            if sid and rno:
                id_by_roll[rno] = sid
                roll_by_id[sid] = rno
        print(f"Student index loaded: {len(id_by_roll)}")
    except Exception as e:
        print(f"⚠️ Failed to load student index: {e}", file=sys.stderr)

def load_all_known_faces():
    known_encs.clear()
    known_labels.clear()
    for p in KNOWN_DIR.iterdir():
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            try:
                img = face_recognition.load_image_file(p)
                encs = face_recognition.face_encodings(img)
                if encs:
                    known_encs.append(encs[0])
                    known_labels.append(p.stem)  # uuid OR roll_no
                else:
                    print(f"⚠️ No face found in {p.name} — skipping")
            except Exception as e:
                print(f"⚠️ Failed to load {p.name}: {e}")

def resolve_student_id(label: str) -> Optional[str]:
    """If label is UUID, return it; else treat as roll_no and map → UUID."""
    if UUID_RE.match(label or ""):
        return label
    return id_by_roll.get(label)

def resolve_roll_no(label: str) -> Optional[str]:
    """If label is roll, return it; if UUID, map to roll."""
    if UUID_RE.match(label or ""):
        return roll_by_id.get(label)
    return label

refresh_student_index()
load_all_known_faces()
print("Known labels:", known_labels if known_labels else "(none)")


# ──────────────────────────────────────────────────────────────
# 4) Serve Admin UI (your existing admin.html) at "/"
# ──────────────────────────────────────────────────────────────
@app.get("/", response_class=FileResponse)
def serve_admin():
    path = BASE_DIR / "admin.html"
    if not path.exists():
        return HTMLResponse("<h1>admin.html not found</h1>", status_code=404)
    return FileResponse(path)


# ──────────────────────────────────────────────────────────────
# 5) Separate Scanner UI at "/scan" (shows roll_no)
# ──────────────────────────────────────────────────────────────
SCAN_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Face Scan</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-slate-950 text-slate-100 min-h-screen">
  <div class="max-w-3xl mx-auto p-6">
    <header class="flex items-center justify-between mb-6">
      <h1 class="text-xl font-semibold">Face Scan</h1>
      <a href="/" class="text-sm px-3 py-1.5 rounded bg-white/10 hover:bg-white/15">Back to Admin</a>
    </header>

    <div class="grid md:grid-cols-2 gap-6">
      <div class="rounded-xl border border-white/10 p-4 bg-white/5">
        <video id="v" class="w-full rounded-lg aspect-video bg-black" autoplay muted playsinline></video>
        <div class="mt-4 flex gap-2">
          <button id="btn" class="px-4 py-2 rounded-lg bg-indigo-600 hover:bg-indigo-500">Snapshot & Recognize</button>
          <button id="toggle" class="px-4 py-2 rounded-lg bg-white/10 hover:bg-white/15">Start Camera</button>
        </div>
      </div>

      <div class="rounded-xl border border-white/10 p-4 bg-white/5">
        <h2 class="font-medium mb-3">Status</h2>
        <div id="chip" class="inline-flex items-center gap-2 text-sm px-3 py-1.5 rounded-full bg-amber-500/15 text-amber-200 border border-amber-500/30">
          Idle
        </div>
        <div class="mt-4 text-sm space-y-1">
          <div>Roll No: <span id="roll" class="font-semibold text-emerald-300">—</span></div>
          <div>Message: <span id="msg" class="text-slate-300">Ready</span></div>
          <div>Distance: <span id="dist" class="text-slate-300">—</span></div>
          <div>Time: <span id="when" class="text-slate-300">—</span></div>
        </div>
      </div>
    </div>
  </div>

<script>
  const v = document.getElementById('v');
  const btn = document.getElementById('btn');
  const tog = document.getElementById('toggle');
  const chip = document.getElementById('chip');
  const roll = document.getElementById('roll');
  const msg  = document.getElementById('msg');
  const dist = document.getElementById('dist');
  const when = document.getElementById('when');

  let stream = null;

  function setChip(txt, cls){
    chip.textContent = txt;
    chip.className = 'inline-flex items-center gap-2 text-sm px-3 py-1.5 rounded-full border ' + cls;
  }
  async function startCam(){
    try{
      stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' }, audio:false });
      v.srcObject = stream;
      setChip('Camera Ready', 'bg-emerald-500/15 text-emerald-200 border-emerald-500/30');
      msg.textContent = 'Camera started';
    }catch(e){
      setChip('Camera Error', 'bg-rose-500/15 text-rose-200 border-rose-500/30');
      msg.textContent = 'Camera error: ' + e;
    }
  }
  function stopCam(){
    if(stream){ stream.getTracks().forEach(t=>t.stop()); stream = null; }
    v.srcObject = null;
    setChip('Idle', 'bg-amber-500/15 text-amber-200 border-amber-500/30');
    msg.textContent = 'Camera stopped';
  }
  tog.onclick = ()=> { stream ? stopCam() : startCam(); };

  btn.onclick = async ()=>{
    if(!stream){ await startCam(); if(!stream) return; }
    const c = document.createElement('canvas');
    c.width = v.videoWidth || 640; c.height = v.videoHeight || 480;
    c.getContext('2d').drawImage(v, 0, 0);
    c.toBlob(async b => {
      const f = new FormData(); f.append('file', b, 'snap.jpg');
      setChip('Recognizing…', 'bg-indigo-500/15 text-indigo-200 border-indigo-500/30');
      msg.textContent = 'Recognizing…'; roll.textContent='—'; dist.textContent='—'; when.textContent='—';
      try{
        const r = await fetch('/recognize', { method:'POST', body:f });
        const j = await r.json();
        if(r.ok){
          roll.textContent = j.roll_no || 'Unknown';
          msg.textContent = j.message || '';
          dist.textContent = (j.distance!=null) ? String(j.distance.toFixed(4)) : '—';
          when.textContent = new Date().toLocaleString();
          if(j.roll_no){ setChip('Matched', 'bg-emerald-500/15 text-emerald-200 border-emerald-500/30'); }
          else { setChip('Unknown', 'bg-amber-500/15 text-amber-200 border-amber-500/30'); }
        }else{
          setChip('Error', 'bg-rose-500/15 text-rose-200 border-rose-500/30');
          msg.textContent = j.message || 'Error';
        }
      }catch(e){
        setChip('Error', 'bg-rose-500/15 text-rose-200 border-rose-500/30');
        msg.textContent = 'Network error';
      }
    }, 'image/jpeg', 0.9);
  };

  // auto-start camera on load (can be toggled)
  startCam();
</script>
</body>
</html>
"""

@app.get("/scan", response_class=HTMLResponse)
def scan_page():
    return SCAN_HTML


# ──────────────────────────────────────────────────────────────
# 6) ENROLL endpoints
# ──────────────────────────────────────────────────────────────
@app.post("/enroll")
async def enroll_face(
    student_id: str = Query(..., description="students.id (UUID)"),
    file: UploadFile = File(...)
):
    """Save face to known_faces/<student_id>.jpg and refresh encodings."""
    if not UUID_RE.match(student_id):
        raise HTTPException(400, "student_id must be a UUID (students.id)")
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty file")

    img = face_recognition.load_image_file(io.BytesIO(data))
    encs = face_recognition.face_encodings(img)
    if not encs:
        raise HTTPException(400, "No face found in image")

    cvimg = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if cvimg is None:
        raise HTTPException(400, "Invalid image")

    outp = KNOWN_DIR / f"{student_id}.jpg"
    cv2.imwrite(str(outp), cvimg)

    refresh_student_index()
    load_all_known_faces()
    return {"message": f"Enrolled face for {student_id}", "file": outp.name}

@app.post("/enroll_by_roll")
async def enroll_face_by_roll(
    roll_no: str = Query(..., description="students.roll_no"),
    file: UploadFile = File(...)
):
    """
    Save image using roll_no, or convert to UUID if found.
    Preferred: map roll_no -> students.id and save as <uuid>.jpg.
    If mapping fails, save as <roll_no>.jpg (recognizer will map later).
    """
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty file")

    img = face_recognition.load_image_file(io.BytesIO(data))
    encs = face_recognition.face_encodings(img)
    if not encs:
        raise HTTPException(400, "No face found in image")

    cvimg = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if cvimg is None:
        raise HTTPException(400, "Invalid image")

    sid = id_by_roll.get(roll_no)
    outp = KNOWN_DIR / (f"{sid}.jpg" if sid else f"{roll_no}.jpg")
    cv2.imwrite(str(outp), cvimg)

    refresh_student_index()
    load_all_known_faces()
    return {"message": f"Enrolled face for {roll_no}", "file": outp.name, "mapped_uuid": sid}


# ──────────────────────────────────────────────────────────────
# 7) RECOGNIZE: returns roll_no, writes attendance_daily
# ──────────────────────────────────────────────────────────────
@app.post("/recognize")
async def recognize(file: UploadFile = File(...)):
    data = await file.read()
    frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(400, "Bad image")

    try:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    except Exception:
        return JSONResponse({"message": "Invalid image data"}, status_code=400)

    # detect & encode largest face
    locs = face_recognition.face_locations(rgb, number_of_times_to_upsample=1, model="hog")
    if not locs:
        return JSONResponse({"message": "No face detected"})
    (top, right, bottom, left) = max(locs, key=lambda bb: (bb[2]-bb[0]) * (bb[1]-bb[3]))
    encs = face_recognition.face_encodings(rgb, [(top, right, bottom, left)])
    if not encs:
        return JSONResponse({"message": "No face encoding"})

    if not known_encs:
        return JSONResponse({"message": "No known faces loaded. Enroll faces first."})

    # match
    dists = face_recognition.face_distance(known_encs, encs[0])
    idx = int(np.argmin(dists))
    dist = float(dists[idx])
    if dist > TOLERANCE:
        return JSONResponse({"message": "Unknown face", "roll_no": None, "distance": dist}, status_code=200)

    label = known_labels[idx]               # uuid OR roll_no
    student_id = resolve_student_id(label)  # uuid
    roll_no = resolve_roll_no(label)        # roll

    if not student_id or not roll_no:
        return JSONResponse({
            "message": f"Matched '{label}', but couldn't map to students table. "
                       f"Enroll by UUID or ensure roll_no exists.",
            "roll_no": roll_no or None,
            "distance": dist
        }, status_code=200)

    # Upsert attendance for IST today (entry first, then exit)
    today = today_ist_str()
    now_hms = now_ist().time().strftime("%H:%M:%S")

    try:
        resp = (
            sb.table("attendance_daily")
              .select("id, entry_time, exit_time")
              .eq("student_id", student_id)
              .eq("att_date", today)
              .limit(1)
              .execute()
        )
        row = resp.data[0] if resp.data else None
    except Exception as e:
        return JSONResponse({"message": f"DB error (select): {e}"}, status_code=500)

    try:
        if row:
            sb.table("attendance_daily").update({"exit_time": now_hms}).eq("id", row["id"]).execute()
            msg = f"Exit marked for {roll_no} at {now_hms}"
        else:
            payload = {
                "student_id": student_id,
                "att_date": today,
                "entry_time": now_hms,
                "exit_time": None,
                "notes": None
            }
            sb.table("attendance_daily").insert(payload).execute()
            msg = f"Entry marked for {roll_no} at {now_hms}"
    except Exception as e:
        return JSONResponse({"message": f"DB error (upsert): {e}"}, status_code=500)

    return JSONResponse({"message": msg, "roll_no": roll_no, "distance": dist})


# ──────────────────────────────────────────────────────────────
# 8) CSV for today (optional)
# ──────────────────────────────────────────────────────────────
@app.get("/admin/download_csv")
def download_csv():
    today = today_ist_str()
    try:
        atts = (
            sb.table("attendance_daily")
              .select("student_id, entry_time, exit_time")
              .eq("att_date", today)
              .order("entry_time", desc=False)
              .execute()
              .data
            or []
        )
        ids = list({row["student_id"] for row in atts})
        studs = (
            sb.table("students")
              .select("id, full_name, roll_no")
              .in_("id", ids).execute().data
        ) if ids else []
        label = {s["id"]: (s.get("full_name") or s.get("roll_no") or s["id"]) for s in (studs or [])}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=["student_id","student_label","att_date","entry_time","exit_time"])
    w.writeheader()
    for r in atts:
        sid = r.get("student_id","")
        w.writerow({
            "student_id": sid,
            "student_label": label.get(sid, ""),
            "att_date": today,
            "entry_time": r.get("entry_time","") or "",
            "exit_time": r.get("exit_time","") or ""
        })
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="attendance_{today}.csv"'}
    )


# ──────────────────────────────────────────────────────────────
# 9) Run
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=9999, reload=True)
