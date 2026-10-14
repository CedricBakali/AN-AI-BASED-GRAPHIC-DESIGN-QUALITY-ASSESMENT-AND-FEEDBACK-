# DesignScore — AI-Powered Graphic Design Quality Assessment Assistant

## 1. System Description

DesignScore is a Flask web application that automatically assesses the quality of finished graphic designs (posters, adverts, flyers, social posts).

A user uploads an image (`PNG, JPG, JPEG, WEBP, BMP, GIF`, max 10 MB). The system runs a **deterministic, rule-based computer-vision analysis** in `app/services/analysis.py` using OpenCV, NumPy, and Pillow — no trained model required at runtime — and returns:

- Overall score 0-100 + grade (`A - Excellent` to `E - Poor`)
- Six category scores 0-100 with plain-language summaries and improvement suggestions
- Dominant colour palette (hex codes via k-means)

Results are persisted per-user in SQLite (`app/models/designscore.db`) and viewable via dashboard, result page, and history.

Supporting `app/ai/` (`clip_analyzer.py`, `nima_analyzer.py`, `vlm_analyzer.py`, `model_manager.py`) and `app/cv/` (`color_analysis.py`, `contrast_analysis.py`, `layout_analysis.py`, `typography_analysis.py`, `whitespace_analysis.py`, `preprocessing.py`) modules provide experimental / modular extensions to the core rule-based pipeline.

### Key Features
- User auth: register / login / logout with hashed passwords, session management, `login_required` guard, `user` / `admin` roles, active-flag check.
- Upload + analyze: validates file presence, type, size; saves with `secure_filename` + UUID; handles 413 too-large.
- Six-principle assessment: Contrast, Color Harmony, Typography, Alignment, White Space, Readability (composite).
- Persistence: `users` and `analyses` tables, per-user isolation (`WHERE user_id = ?`), last-30 history.
- Dashboard: total designs, assessments completed, average score, recent assessments.
- Feedback: each category returns `summary` + `suggestions[]` rendered in templates.

## 2. Architecture

```
Browser (Jinja templates in app/templates/)
  -> Flask (app.py + Blueprints: auth, dashboard, analysis)
  -> Services (app/services/analysis.py orchestrator)
  -> CV helpers (OpenCV / NumPy / Pillow)
  -> Models (app/models/database.py + user.py, SQLite)
  -> static/uploads/ (stored images)
```

- Entry: `app.py` — creates `Flask(__name__, template_folder="app/templates")`, sets `MAX_CONTENT_LENGTH`, `secret_key`, inits DB, registers `dashboard_bp` and `auth_bp`, defines `/`, `/analyze`, `/result/<id>`, `/history`.
- Blueprints:
  - `app/routes/auth.py` — `/register`, `/login`, `/logout`, `login_required`.
  - `app/routes/dashboard.py` — `/dashboard` (user info + history + stats).
  - `app/routes/analysis.py` — `/analyze` (login-protected upload + `analyze_image()` + `save_analysis()` + `results/result.html`).
- Analysis orchestrator: `analyze_image(path)` in `app/services/analysis.py` — `load_image()` (downscale to max 1000px) → `dominant_colors(k=5)` → 6 scorers → average overall → grade.
- DB layer: `app/models/database.py` — `get_connection()`, `init_db()`, `save_analysis()`, `get_history()`, `get_analysis()`; `app/models/user.py` — `create_user()`, `get_user_by_email/id`, `hash/check_password` via Werkzeug.

## 3. Assessment Logic (0-100 each)

1. **Contrast** — WCAG 2.1 contrast ratio of top-2 dominant colours + grayscale spread. 4.5:1 (AA) ~70, 7:1 (AAA) ~90.
2. **Color Harmony** — colour-wheel check on hues with ≥5% weight: monochrome / analogous / complementary / triadic / mixed / busy. Penalizes over-saturation, rewards intentional schemes.
3. **White Space** — Canny edge density for quiet-space % (ideal 55-88%) + quadrant balance.
4. **Alignment** — column/row edge-profile concentration on few strong guides (grid tidiness).
5. **Typography** — MSER text-region height clustering into 2-4 size tiers = good hierarchy.
6. **Readability** — weighted blend: `0.5*Contrast + 0.25*WhiteSpace + 0.25*Typography`.

Overall = mean of 6 categories. Grade: ≥90 A, ≥80 B, ≥70 C, ≥60 D, else E.

## 4. Tech Stack

- Backend: Python, Flask, Werkzeug
- Vision: OpenCV (`cv2`), NumPy, Pillow
- DB: SQLite (`sqlite3`), JSON-serialized palette/categories
- Frontend: Jinja2 (`app/templates/base.html`, `index.html`, `auth/`, `dashboard/`, `analysis/`, `history/`), CSS in `static/`, root `templates/index.html` + `index.html`
- Tests: `tests/test_analysis.py`, `test_auth.py`, `test_scoring.py`, `test_upload.py`
- Docs: `UML Diagrams/*.puml` (use-case, class, sequence, activity, Level 0/1/2 DFD)

## 5. Project Structure

```
app.py                  # app factory/routes: /, /analyze, /result, /history
config.py               # config (currently empty placeholder)
app/routes/             # auth.py, dashboard.py, analysis.py, history.py
app/services/           # analysis.py (core), scoring.py, feedback.py (placeholders)
app/models/             # database.py, user.py, design.py, assessment.py, designscore.db
app/cv/                 # preprocessing, color, contrast, layout, typography, whitespace
app/ai/                 # clip, nima, vlm analyzers, model_manager
app/templates/ + app/static/  # UI; static/uploads/ for uploads
templates/index.html    # legacy template copy
tests/                  # analysis, auth, scoring, upload tests
research/sample_designs/ # sample inputs
instance/               # Flask instance folder
UML Diagrams/           # PlantUML sources + rendered PNG
```

## 6. Data Model (SQLite)

`users(id, username, email UNIQUE, password_hash, role[user|admin], is_active, created_at)`
`analyses(id, user_id FK->users CASCADE, filename, image_path, overall, grade, palette JSON, categories JSON, created_at)`

## 7. Setup & Run

```bash
pip install -r requirements.txt
python app.py
# http://127.0.0.1:5000 -> redirects / to auth.login
```

Set `DESIGNSCORE_SECRET` env var in production (defaults to `dev-secret-change-me`).

Upload flow: Dashboard → choose image → POST `/analyze` → redirect `/result/<id>`; `/history` lists last 30.

## 8. Limitations & Next Steps

- Rule-based only; typography uses MSER heuristic, may miss stylized text.
- Palette contrast uses top-2 k-means colours, not true text/background segmentation.
- Duplicate `/analyze` logic in `app.py` and `analysis` blueprint; `scoring.py`/`feedback.py`/`history.py`/`assessment.py`/`design.py` are stubs.
- Future: wire in `app/ai/` (CLIP/NIMA/VLM) scoring, admin dashboard, REST API, larger test coverage.
