CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
INSERT OR IGNORE INTO settings VALUES ('office', '{"lat":13.0942949,"lng":77.5505004,"radius":500,"enabled":true}');
CREATE TABLE IF NOT EXISTS employees (
 id TEXT PRIMARY KEY, code TEXT NOT NULL UNIQUE, name TEXT NOT NULL,
 descriptors TEXT NOT NULL, active INTEGER NOT NULL DEFAULT 1,
 locker TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS attendance (
 id TEXT PRIMARY KEY, employee_id TEXT NOT NULL REFERENCES employees(id), day TEXT NOT NULL,
 check_in TEXT NOT NULL, check_out TEXT, in_distance REAL, out_distance REAL,
 UNIQUE(employee_id,day)
);
CREATE INDEX IF NOT EXISTS attendance_day ON attendance(day);
CREATE TABLE IF NOT EXISTS challenges (id TEXT PRIMARY KEY, employee_id TEXT NOT NULL, expires INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS sessions (token TEXT PRIMARY KEY, expires INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS rate_limits (key TEXT PRIMARY KEY, count INTEGER NOT NULL, expires INTEGER NOT NULL);
