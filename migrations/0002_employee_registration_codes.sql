ALTER TABLE employees ADD COLUMN registration_code TEXT;
CREATE UNIQUE INDEX IF NOT EXISTS employee_registration_code_unique ON employees(registration_code) WHERE registration_code IS NOT NULL;
