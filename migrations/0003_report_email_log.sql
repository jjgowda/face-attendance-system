CREATE TABLE IF NOT EXISTS report_email_log (
 id TEXT PRIMARY KEY, report_day TEXT NOT NULL, slot TEXT NOT NULL,
 recipients TEXT NOT NULL, status TEXT NOT NULL, attempted_at TEXT NOT NULL,
 accepted_at TEXT, error TEXT
);
CREATE INDEX IF NOT EXISTS report_email_recent ON report_email_log(attempted_at);
