# DesignScore — AI-Powered Graphic Design Quality Assessment Assistant

## 1. System Description

DesignScore is a Flask web application that automatically assesses the quality of finished graphic designs (posters, adverts, flyers, social posts).

A user uploads an image (`PNG, JPG, JPEG, WEBP, BMP, GIF`, max 10 MB). The system runs a **deterministic, rule-based computer-vision analysis** in `app/services/analysis.py` using OpenCV, NumPy, and Pillow — no trained model required at runtime — and returns:

- Overall score 0-100 + grade (`A - Excellent` to `E - Poor`)
- Eight category scores 0-100 with plain-language summaries
- Strengths, weaknesses and ordered recommendations (`app/services/feedback.py`)
- Dominant colour palette (hex codes via k-means)

Results are persisted per-user in SQLite (`app/models/designscore.db`) and viewable via dashboard, result page, and history.

Supporting `app/ai/` (`clip_analyzer.py`, `nima_analyzer.py`, `vlm_analyzer.py`, `model_manager.py`) and `app/cv/` (`color_analysis.py`, `contrast_analysis.py`, `layout_analysis.py`, `typography_analysis.py`, `whitespace_analysis.py`, `preprocessing.py`) modules provide experimental / modular extensions to the core rule-based pipeline.

### Key Features
- User auth: register / login / logout with hashed passwords, session management, `login_required` + `admin_required` guards, `user` / `admin` roles (register always creates `user`), active-flag check.
- Upload + analyze: validates file presence, type, size; saves with `secure_filename` + UUID; handles 413 too-large.
- Eight-principle assessment: Contrast, Color Harmony, Typography, Alignment, White Space, Readability (composite), Layout, Visual Hierarchy.
- Scoring (`app/services/scoring.py`): overall = mean of categories, fixed A–E bands — recomputable by hand.
- Feedback (`app/services/feedback.py`): score ≥ 75 → strength, < 60 → weakness, suggestions ordered weakest-first.
- Admin: separate `/admin/` dashboard (stats, recent activity), user list, per-user detail, enable/disable accounts; all role-guarded backend-side.
- Persistence: `users` and `analyses` tables, per-user isolation (`WHERE user_id = ?`), last-30 history.
- Dashboard: total designs, assessments completed, average score, recent assessments — all from the database.
- Tests: `pytest tests/` — 19 tests (analysis, auth, scoring/feedback, upload + isolation) on an isolated temp DB.

## 2. Architecture

```
Browser (Jinja templates in app/templates/)
  -> Flask (app.py + Blueprints: auth, dashboard, analysis)
  -> Services (app/services/analysis.py orchestrator)
  -> CV helpers (OpenCV / NumPy / Pillow)
  -> Models (app/models/database.py + user.py, SQLite)
  -> static/uploads/ (stored images)
```

- Entry: `app.py` — loads `config.Config`, inits DB, registers `auth_bp`, `dashboard_bp`, `admin_bp`; defines `/`, `/analyze`, `/result/<id>`, `/history`.
- Blueprints:
  - `app/routes/auth.py` — `/register`, `/login`, `/logout`, `login_required`, `admin_required`.
  - `app/routes/dashboard.py` — `/dashboard` (user info + history + stats from DB).
  - `app/routes/analysis.py` — no routes (single `/analyze` lives in `app.py`); keeps `allowed_file()` helper.
  - `app/routes/admin.py` — `/admin/` (stats + activity), `/admin/users`, `/admin/users/<id>`, toggle active flag.
- Analysis orchestrator: `analyze_image(path)` in `app/services/analysis.py` — `load_image()` (downscale to max 1000px) → `dominant_colors(k=5)` → 8 scorers → `compute_overall()` → `grade_for()`.
- DB layer: `app/models/database.py` — `get_connection()`, `init_db()`, `save_analysis()`, `get_history()`, `get_analysis()`; `app/models/user.py` — `create_user()`, `get_user_by_email/id`, `hash/check_password` via Werkzeug.

## 3. Assessment Logic (0-100 each)

1. **Contrast** — WCAG 2.1 contrast ratio of top-2 dominant colours + grayscale spread. 4.5:1 (AA) ~70, 7:1 (AAA) ~90.
2. **Color Harmony** — colour-wheel check on hues with ≥5% weight: monochrome / analogous / complementary / triadic / mixed / busy. Penalizes over-saturation, rewards intentional schemes.
3. **White Space** — Canny edge density for quiet-space % (ideal 55-88%) + quadrant balance.
4. **Alignment** — column/row edge-profile concentration on few strong guides (grid tidiness).
5. **Typography** — MSER text-region height clustering into 2-4 size tiers = good hierarchy.
6. **Readability** — weighted blend: `0.5*Contrast + 0.25*WhiteSpace + 0.25*Typography`.
7. **Layout** — 3x3 grid balance + share of detail near rule-of-thirds intersections.
8. **Visual Hierarchy** — focal dominance ratio (largest vs secondary masses) + headline-high position bonus.

Overall = mean of 8 categories (`compute_overall`). Grade: ≥90 A, ≥80 B, ≥70 C, ≥60 D, else E (`grade_for`).

## 4. Tech Stack

- Backend: Python, Flask, Werkzeug
- Vision: OpenCV (`cv2`), NumPy, Pillow
- DB: SQLite (`sqlite3`), JSON-serialized palette/categories
- Frontend: Jinja2 (`app/templates/base.html`, `index.html`, `auth/`, `dashboard/`, `analysis/`, `history/`), CSS in `static/`, root `templates/index.html` + `index.html`
- Tests: `tests/test_analysis.py`, `test_auth.py`, `test_scoring.py`, `test_upload.py`
- Docs: `UML Diagrams/*.puml` (use-case, class, sequence, activity, Level 0/1/2 DFD)

## 5. Project Structure

```
app.py                  # routes: /, /analyze, /result, /history (+ blueprint registration)
config.py               # central Config: secret, upload limits, paths
app/routes/             # auth.py, dashboard.py, analysis.py (helpers), admin.py, history.py (empty)
app/services/           # analysis.py (8 scorers), scoring.py (overall+grade), feedback.py (strengths/weaknesses)
app/models/             # database.py, user.py, design.py / assessment.py (empty placeholders), designscore.db
app/cv/                 # contrast_analysis.py (reference copy); color/layout/typography/whitespace/preprocessing empty
app/ai/                 # clip/nima/vlm/model_manager — empty stubs for future AI direction, not wired
app/templates/          # auth/, dashboard/, analysis/result.html, history/history.html, admin/
static/                 # css/, images/, uploads/
tests/                  # conftest.py + test_analysis/auth/scoring/upload.py (19 tests, isolated temp DB)
```

## 6. Data Model (SQLite)

`users(id, username, email UNIQUE, password_hash, role[user|admin], is_active, created_at)`
`analyses(id, user_id FK->users CASCADE, filename, image_path, overall, grade, palette JSON, categories JSON, created_at)`

## 7. Setup & Run

```bash
pip install -r requirements.txt
python app.py
# http://127.0.0.1:5000 -> redirects / to auth.login
python -m pytest tests/ -q
# 19 passed on an isolated temp DB (real designscore.db untouched)
```

Set `DESIGNSCORE_SECRET` env var in production (defaults to `dev-secret-change-me`).

Upload flow: Dashboard → choose image → POST `/analyze` → redirect `/result/<id>`; `/history` lists last 30.

## 8. Limitations & Next Steps

- Rule-based only (honestly so — no model trained); typography uses MSER heuristic, may miss stylized text.
- Palette contrast uses top-2 k-means colours, not true text/background segmentation.
- Future: wire in `app/ai/` (CLIP/NIMA/VLM) features behind the same 0–100 scorers, admin reports export, design refresh (deferred by owner until backend done).
