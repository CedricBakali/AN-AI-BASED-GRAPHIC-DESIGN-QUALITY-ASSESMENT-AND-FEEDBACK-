# DesignScore — AI-Powered Graphic Design Quality Assessment Assistant

## 1. System Description

DesignScore is a Flask web application that automatically assesses the quality of finished graphic designs (posters, adverts, flyers, social posts).

A user uploads an image (`PNG, JPG, JPEG, WEBP, BMP, GIF`, max 10 MB). The system runs a **deterministic, rule-based computer-vision analysis** in `app/services/analysis.py` using OpenCV, NumPy, and Pillow — no trained model required at runtime — and returns:

- Overall score 0-100 + grade (`A - Excellent` to `E - Poor`)
- Eight category scores 0-100 with plain-language summaries
- Strengths, weaknesses and ordered recommendations (`app/services/feedback.py`)
- Dominant colour palette (hex codes via k-means)

Results are persisted per-user in SQLite (`app/models/designscore.db`) and viewable via dashboard, result page, and history.

The **Design Quality Score above is never mixed with AI output.** Two further, clearly-separated opinions are shown next to it when available:

- **AI Visual Assessment** — zero-shot CLIP (`openai/clip-vit-base-patch32`) scores the image 0–100 against eight plain-language design statements (`app/ai/clip_analyzer.py`).
- **NIMA aesthetic score** — a VGG-16 trained on the AVA dataset reports aesthetic quality on its **native 0–10 scale** (`app/ai/nima_analyzer.py`).

Both are labelled and reported separately; neither is averaged with the other or with the deterministic score, and no combined weighted total exists anywhere in the code. Model weights (~670 MB total) are fetched on first run into gitignored folders and are never committed.

Supporting `app/ai/` (`clip_analyzer.py` and `nima_analyzer.py` — both live; `model_manager.py`, `model_loader.py`, `aesthetic_analyzer.py` — shared loading helpers; `vlm_analyzer.py` — an empty placeholder) and `app/cv/` (`preprocessing.py`, `feature_extraction.py` — shared image helpers) back the core rule-based pipeline.

### Key Features
- User auth: register / login / logout with hashed passwords, session management, `login_required` + `admin_required` guards, `user` / `admin` roles (register always creates `user`), active-flag check.
- Upload + analyze: validates file presence, type, size; saves with `secure_filename` + UUID; handles 413 too-large.
- Eight-principle assessment: Contrast, Color Harmony, Typography, Alignment, White Space, Readability (composite), Layout, Visual Hierarchy.
- Scoring (`app/services/scoring.py`): overall = mean of categories, fixed A–E bands — recomputable by hand.
- Feedback (`app/services/feedback.py`): score ≥ 75 → strength, < 60 → weakness, suggestions ordered weakest-first.
- Admin: separate `/admin/` dashboard (stats, recent activity), user list, per-user detail, enable/disable accounts; all role-guarded backend-side.
- Persistence: `users` and `analyses` tables, per-user isolation (`WHERE user_id = ?`), last-30 history.
- Dashboard: total designs, assessments completed, average score, recent assessments — all from the database.
- Reports: simple-English result page with the Design Quality Score and the AI Visual Assessment kept apart, a collapsible Technical AI Details block, a Full Report, and a Print / Save-as-PDF button (`@media print` styles in `result.css` / `report.css`).
- Security: hashed passwords, server-side role checks, session-bound CSRF tokens on every form, per-owner access control on `/static/uploads/`, random persisted secret key, debug mode off by default.
- Tests: `pytest tests/` — 128 tests (analysis, auth, scoring/feedback, upload + isolation, report wording, CLIP, NIMA real inference, security, upload privacy, CSRF) on an isolated temp DB.

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
- Frontend: Jinja2 (`app/templates/` — `auth/`, `dashboard/`, `analysis/`, `history/`, `admin/`, `profile/`, `partials/shell.html`), CSS in `static/css/`
- AI: PyTorch (CPU) + `timm` for NIMA, Hugging Face `transformers` for CLIP
- Tests: `tests/test_analysis.py`, `test_auth.py`, `test_scoring.py`, `test_upload.py`, `test_ai_model.py`, `test_clip_analyzer.py`, `test_nima.py`, `test_report.py`, `test_security.py`, `test_uploads_privacy.py`, `test_csrf.py`
- Docs: `UML Diagrams/*.puml` (use-case, class, sequence, activity, Level 0/1/2 DFD)

## 5. Project Structure

```
app.py                  # routes: /, /analyze, /result, /history (+ blueprint registration, upload privacy guard, CSRF check)
config.py               # central Config: secret resolution, debug flag, upload limits, paths
app/routes/             # auth.py, dashboard.py, analysis.py (helpers), admin.py, training.py (AI labelling UI)
app/services/           # analysis.py (8 scorers), scoring.py (overall+grade), feedback.py (strengths/weaknesses)
app/models/             # database.py, user.py, designscore.db
app/cv/                 # preprocessing.py, feature_extraction.py (shared image helpers)
app/ai/                 # clip_analyzer.py + nima_analyzer.py (live), model_manager/model_loader/aesthetic_analyzer (helpers), vlm_analyzer.py (empty)
app/templates/          # auth/, dashboard/, analysis/result.html + report.html, history/, admin/, profile/, partials/
static/                 # css/, images/, uploads/ (gitignored)
tests/                  # conftest.py + 11 test modules (128 tests, isolated temp DB)
```

## 6. Data Model (SQLite)

`users(id, username, email UNIQUE, password_hash, role[user|admin], is_active, created_at)`
`analyses(id, user_id FK->users CASCADE, filename, image_path, overall, grade, palette JSON, categories JSON, ai_score, ai_factors JSON, ai_model, nima_score, nima_model, created_at)`

`ai_score` is CLIP's 0–100 opinion, `nima_score` is NIMA's native 0–10 opinion; both are `NULL` when that model was unavailable for that run. Neither is ever folded into `overall`.

## 7. Setup & Run

```bash
pip install -r requirements.txt
python app.py
# http://127.0.0.1:5000 -> redirects / to auth.login
python -m pytest tests/ -q
# 128 passed on an isolated temp DB (real designscore.db untouched)
```

The secret key is resolved by `config.resolve_secret_key()`: the `DESIGNSCORE_SECRET` environment variable wins if set, otherwise a random key is generated once and persisted in `.designscore_secret` (gitignored). No hardcoded fallback secret exists. `FLASK_DEBUG=1` is required to enable the interactive debugger; it is off by default.

First run downloads the CLIP and NIMA weights into gitignored folders (`app/ai/weights/` and the Hugging Face cache); set `DESIGNSCORE_CLIP_WARMUP=0` to skip the CLIP warm-up (used by the test suites).

Validation: `python scripts/validate_clip.py` computes Spearman/Pearson correlation and a bootstrap confidence interval between CLIP's scores and a **user-supplied** human-ratings CSV (`validation/ratings.csv`). It ships with no dataset and no results, and exits with an explanatory error until you supply real ratings.

Upload flow: Dashboard → choose image → POST `/analyze` → redirect `/result/<id>`; `/history` lists last 30.

## 8. Limitations & Next Steps

- The 0–100 Design Quality Score is still rule-based — no model has been trained on human ratings for it; typography uses an MSER heuristic and may miss stylized text.
- Palette contrast uses top-2 k-means colours, not true text/background segmentation.
- CLIP and NIMA are wired in but **not yet validated against human ratings**; `scripts/validate_clip.py` is the scaffold for that, and until real ratings are supplied the AI scores are a second opinion, not ground truth.
- `app/ai/vlm_analyzer.py` remains an empty placeholder — a VLM is not implemented.
- Residual security note: `/logout` is still reachable as a GET link (forced-logout CSRF, low severity, deliberately left unchanged in this pass).
- Future: admin report export, design refresh (deferred by owner until backend done).
