# R3 Square — browser face attendance

A blue-and-white mobile website for a team of five. No Python server or always-on computer is required. The legacy Python implementation is retained in `app.py` and documented in `docs/LEGACY-PYTHON.md`.

## Features

- Employee self-registration using an admin-issued registration code and three face samples.
- Select your name and check in or check out; the camera automatically detects and verifies your face.
- Server-side face-template comparison; biometric templates are never included in employee list responses.
- Fresh, high-accuracy browser geolocation checked on the server against the office boundary.
- One check-in and one check-out per employee per IST calendar day; duplicate scans and challenge replays rejected.
- Password-protected admin dashboard, date-based attendance, CSV export, employee activation, face deletion/reset and locker assignments.
- Face processing runs in the browser with bundled face-api.js models. Camera frames are not uploaded or stored. Templates are sensitive biometric data and are stored in the private D1 database.

## Architecture

- `public/`: generated attendance (`/` and `/attendance`), registration (`/register`) and admin (`/admin`) pages, bundled face-api.js and pretrained models.
- `src/page.html` and `scripts/build-pages.mjs`: shared page template and static page generator.
- `src/worker.js`: Cloudflare API, admin sessions, registration, matching and attendance.
- `src/core.js`: face comparison, geofence validation and IST date calculation.
- `migrations/`: Cloudflare D1 database schema and initial office configuration.
- `wrangler.jsonc`: Cloudflare Workers Static Assets + D1 configuration.

Office: **R3 Square Pvt Ltd**, coordinates **13.0942949, 77.5505004**, **500 m** radius, from the owner's Google Maps place link. Admin can change this. Location data must be no more than two minutes old and reported accuracy must be 150 m or better. Attendance stores distance to the office rather than raw location coordinates.

## First-time setup

1. Open **`/admin`** on the deployed website.
2. Enter the one-time setup code supplied privately by the deployment operator and choose an admin password of at least 12 characters.
3. Copy the employee registration code from **Office & registration**. Share it privately with the five employees.
4. Each employee opens the website in Safari or Chrome, opens **`/register`**, enters their name, a unique employee ID (e.g. `R3-01`) and the registration code, agrees to face-template use, then their face is captured automatically.
5. Employees use **Mark attendance**, select their name and choose check in or check out. Camera and location permissions are requested when employee pages open; browsers may require an explicit tap if they block automatic prompts. The camera used for the initial permission check is stopped immediately. Face models load only when a scan starts.
6. Assign locker numbers from the admin screen. Locker assignments do not operate physical locks.

The setup code stops working once the admin account is created. Admin sessions expire after 12 hours. Keep the admin password and registration code private. Employees can register again after the admin deletes their old registration; deletion also removes that employee's attendance history, with confirmation in the UI.

## Run locally

Requires Node.js 22+ and a Cloudflare account for deployment.

```sh
npm install
cp .dev.vars.example .dev.vars
npx wrangler d1 migrations apply r3-square-attendance --local
npm run dev
```

The local D1 database is isolated from production. Camera access works on localhost or HTTPS. Real phone testing needs an HTTPS deployment.

## Deploy to your Cloudflare account

This checkout is configured for the owner's existing D1 database. When deploying to a different account, create a new database and update `database_id` in `wrangler.jsonc` first.

```sh
npx wrangler login
npx wrangler d1 migrations apply r3-square-attendance --remote
npx wrangler secret put SETUP_TOKEN
npm run deploy
```

Use a random setup token; enter it at Wrangler's prompt. Never commit tokens, `.env`, `.dev.vars`, biometric data, or admin access instructions. Static assets, API and D1 are hosted on Cloudflare. There is no paid recognition API. Cloudflare plan quotas still apply.

## Verification

```sh
npm test
npm run check
```

Core tests cover face matching, invalid templates, office distance, inaccurate/stale/future coordinates, and IST date boundaries. Local API integration was exercised with synthetic templates: setup/login/session protection, registration, face mismatch, geofence, replay prevention, duplicate check-in/out, locker assignment and deletion. Synthetic vector tests establish API behavior, not real-world face recognition accuracy. Test enrollment and attendance on actual employee phones in the office before operational use.

## Practical limits

This is a cooperative-team attendance MVP, as requested. It performs face matching, not certified liveness detection. Browser-supplied biometric samples and coordinates can be manipulated by a determined user. It does not claim tamper-proof identity verification. Face-match distance is initially 0.5 and should be calibrated with actual employee scans; lighting, camera quality and glasses can affect results. No payroll decisions or automatic physical unlocking are implemented. Overnight shifts and multiple daily sessions are outside this first version.

## Third-party assets

Browser library: `face-api.js` 0.22.2 (MIT), license in `public/vendor/FACE-API-LICENSE`. Pretrained models are bundled from the upstream [face-api.js repository](https://github.com/justadudewhohacks/face-api.js/tree/master/weights). No third-party CDN is required at runtime.
