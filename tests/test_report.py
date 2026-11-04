"""User-facing results experience: wording, full report, download, nav.

These tests guard the presentation layer only. They never change a score:
the design-quality score stays the mean of the category scores and the AI
score stays whatever clip_analyzer returned.
"""
from __future__ import annotations

import io


def _png_upload(client, name="design.png", colour=(200, 60, 60)):
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (300, 220), colour).save(buf, format="PNG")
    buf.seek(0)
    return client.post("/analyze", data={"image": (buf, name)},
                       content_type="multipart/form-data",
                       follow_redirects=False)


def _analysis_id(rv) -> int:
    assert rv.status_code == 302
    assert rv.location.startswith("/result/")
    return int(rv.location.rsplit("/", 1)[1])


def _fake_ai(score=44, available=True):
    return {
        "score": score,
        "model": "openai/clip-vit-base-patch32",
        "label": "CLIP (OpenAI CLIP)",
        "available": available,
        "explanation": "stub explanation",
        "impressions": [
            {"label": "Overall design quality", "positive_probability": 0.44},
            {"label": "Attractiveness", "positive_probability": 0.69},
            {"label": "Professionalism", "positive_probability": 0.18},
            {"label": "Attention-grabbing", "positive_probability": 0.37},
        ],
        "confidence": 0.44,
        "positive_probability": (score or 0) / 100.0,
    }


def _with_ai(monkeypatch, score=44, available=True):
    from app.ai import clip_analyzer
    monkeypatch.setattr(clip_analyzer, "analyze_with_ai",
                        lambda _path: _fake_ai(score=score, available=available))


# --------------------------------------------------------------- wording

def test_result_page_never_shows_raw_lean_language(client, logged_client,
                                                    monkeypatch):
    """'positive/negative lean' is technical evidence, not user wording."""
    _with_ai(monkeypatch, score=44)
    analysis_id = _analysis_id(_png_upload(logged_client))

    html = logged_client.get(f"/result/{analysis_id}").data.decode()
    assert "positive lean" not in html
    assert "negative lean" not in html
    assert "prompt-pair" not in html


def test_result_page_uses_simple_ai_words(client, logged_client, monkeypatch):
    _with_ai(monkeypatch, score=44)
    analysis_id = _analysis_id(_png_upload(logged_client, "simple.png",
                                           (60, 120, 200)))

    html = logged_client.get(f"/result/{analysis_id}").data.decode()
    assert "AI Aesthetic Opinion" in html
    assert "44/100" in html
    # 44 sits below the lowest band.
    assert "Needs improvement" in html
    # The two headings that explain the separation to a beginner.
    assert "Design Quality Score" in html
    assert "A separate AI opinion" in html


def test_result_page_hides_the_technical_detail(client, logged_client,
                                                monkeypatch):
    """Technical evidence stays available, but folded away by default."""
    _with_ai(monkeypatch, score=73)
    analysis_id = _analysis_id(_png_upload(logged_client, "hidden.png",
                                           (90, 120, 60)))

    html = logged_client.get(f"/result/{analysis_id}").data.decode()
    assert "<details class=\"tech\">" in html
    assert "<summary>Technical AI Details</summary>" in html
    # ...and it still records the model honestly when opened.
    assert "openai/clip-vit-base-patch32" in html
    assert "CLIP (OpenAI CLIP)" in html


def test_result_page_never_combines_the_two_scores(client, logged_client,
                                                   monkeypatch):
    """overall is the category mean; ai_score is separate. No blend."""
    _with_ai(monkeypatch, score=95)
    rv = _png_upload(logged_client, "separate.png", (30, 30, 200))
    analysis_id = _analysis_id(rv)

    from app.models import database
    record = database.get_analysis(analysis_id, _user_id(logged_client))
    cats = record["categories"]
    assert record["overall"] == round(
        sum(c["score"] for c in cats) / len(cats))
    assert record["ai_score"] == 95

    html = logged_client.get(f"/result/{analysis_id}").data.decode()
    assert f"{record['overall']}/100" in html
    assert "95/100" in html


def _user_id(client) -> int:
    from app.models import database
    with database.get_connection() as conn:
        row = conn.execute(
            "SELECT id FROM users WHERE email = ?", ("tester@test.com",)
        ).fetchone()
    assert row is not None
    return int(row["id"])


# ----------------------------------------------------------- full report

def test_result_page_links_to_the_full_report(client, logged_client,
                                              monkeypatch):
    _with_ai(monkeypatch, score=44)
    analysis_id = _analysis_id(_png_upload(logged_client, "link.png"))

    html = logged_client.get(f"/result/{analysis_id}").data.decode()
    assert "View Full Report" in html
    assert f'href="/report/{analysis_id}"' in html


def test_full_report_renders_every_section(client, logged_client,
                                           monkeypatch):
    _with_ai(monkeypatch, score=44)
    analysis_id = _analysis_id(_png_upload(logged_client, "full.png",
                                           (120, 80, 200)))

    rv = logged_client.get(f"/report/{analysis_id}")
    assert rv.status_code == 200
    html = rv.data.decode()

    for needle in (
        "Design Assessment Report",
        "Your Design",
        "Assessed on",
        "Design Quality Score",
        "AI Aesthetic Opinion",
        "Design Principle Scores",
        "Category scores for each principle, out of 100",
        "Strengths",
        "Areas to Improve",
        "Recommended Actions",
        "AI Assessment Summary",
        "Technical AI Details",
        "Download Report",
        # simple AI wording, no raw percentages
        "Overall visual impression",
        "Visual appeal",
        "Professional appearance",
        "Visual impact",
    ):
        assert needle in html, needle

    # No technical lean language, no unexplained jargon on the report.
    assert "positive lean" not in html
    assert "negative lean" not in html
    # Every principle is listed.
    for name in ("Color", "Contrast", "Alignment", "White Space",
                 "Typography", "Readability", "Layout", "Visual Hierarchy"):
        assert name in html, name


def test_full_report_download_is_self_contained(client, logged_client,
                                                monkeypatch):
    """The saved file carries its own CSS and its own preview image."""
    _with_ai(monkeypatch, score=44)
    analysis_id = _analysis_id(_png_upload(logged_client, "saved.png",
                                           (10, 140, 90)))

    rv = logged_client.get(f"/report/{analysis_id}/download")
    assert rv.status_code == 200
    assert "attachment" in rv.headers.get("Content-Disposition", "")
    assert f"DesignScore-Report-{analysis_id}.html" in \
        rv.headers.get("Content-Disposition", "")

    body = rv.data.decode()
    assert "<style>" in body                    # CSS is inlined
    assert "data:image/png;base64," in body     # preview is embedded
    assert "44/100" in body
    assert "Design Principle Scores" in body
    assert "Technical AI Details" in body


def test_report_is_private_to_its_owner(test_app, logged_client,
                                        monkeypatch):
    _with_ai(monkeypatch, score=44)
    analysis_id = _analysis_id(_png_upload(logged_client, "private.png",
                                           (70, 70, 70)))

    # Anonymous visitors are sent to login (a brand new client has no
    # session, unlike `client`, which logged_client shares).
    anon = test_app.test_client()
    assert anon.get(f"/report/{analysis_id}").status_code == 302
    assert anon.get(f"/report/{analysis_id}/download").status_code == 302

    # Another registered user gets a 404, not someone else's report.
    other = test_app.test_client()
    other.post("/register", data={
        "username": "nosy", "email": "nosy@test.com",
        "password": "password123",
    })
    other.post("/login", data={"email": "nosy@test.com",
                               "password": "password123"})
    assert other.get(f"/report/{analysis_id}").status_code == 404
    assert other.get(f"/report/{analysis_id}/download").status_code == 404


def test_report_of_a_missing_analysis_is_404(logged_client):
    assert logged_client.get("/report/999999").status_code == 404
    assert logged_client.get("/report/999999/download").status_code == 404


def test_report_still_works_when_the_ai_is_unavailable(client, logged_client,
                                                       monkeypatch):
    _with_ai(monkeypatch, score=None, available=False)
    analysis_id = _analysis_id(_png_upload(logged_client, "noai.png",
                                           (200, 200, 40)))

    rv = logged_client.get(f"/report/{analysis_id}")
    assert rv.status_code == 200
    html = rv.data.decode()
    assert "Design Principle Scores" in html
    assert "Not available" in html
    assert "Unavailable" in html          # recorded honestly, in the detail
    assert "There is no AI opinion for this design" in html


# ------------------------------------------------------------- navigation

def test_notification_icon_is_removed(logged_client):
    for url in ("/dashboard", "/history", "/profile"):
        html = logged_client.get(url).data.decode()
        assert "notification" not in html, url
        assert "M18 8a6 6 0" not in html, url  # the bell path data


def test_result_page_has_no_notification_icon(client, logged_client,
                                              monkeypatch):
    _with_ai(monkeypatch, score=44)
    analysis_id = _analysis_id(_png_upload(logged_client, "nav.png"))
    html = logged_client.get(f"/result/{analysis_id}").data.decode()
    assert "notification" not in html


# ---------------------------------------------------------- print / save

def _css(name: str) -> str:
    import os

    path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                        "static", "css", name)
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def _print_rules(css: str) -> str:
    """The contents of the @media print block, matched by brace counting."""
    start = css.find("@media print")
    assert start != -1, "stylesheet has no @media print block"
    open_brace = css.find("{", start)
    depth = 0
    for pos in range(open_brace, len(css)):
        if css[pos] == "{":
            depth += 1
        elif css[pos] == "}":
            depth -= 1
            if depth == 0:
                return css[open_brace:pos + 1]
    raise AssertionError("@media print block is never closed")


def test_full_report_prints_a_clean_document():
    """Print/Save as PDF on the report hides controls, keeps the evidence."""
    css = _css("report.css")
    assert "@page" in css, "no paper margins defined"

    rules = _print_rules(css)
    assert "display: none !important" in rules
    # navigation and buttons never go on paper...
    assert ".no-print" in rules
    assert ".doc-actions" in rules
    # ...and neither does the technical control panel
    assert ".tech" in rules
    # colour that carries meaning (grade pills, score bars) must survive
    assert "print-color-adjust" in rules


def test_results_page_now_has_print_rules():
    """Ctrl+P on the results page must not dump the whole app shell."""
    css = _css("result.css")
    rules = _print_rules(css)
    assert "@page" in css, "no paper margins defined"
    assert ".report-top nav" in rules      # Dashboard/History/Logout links
    assert ".cta" in rules                 # action buttons
    assert ".tech" in rules                # technical control panel
    assert "break-inside" in rules         # cards must not split pages


def test_results_page_offers_print_and_save(client, logged_client,
                                            monkeypatch):
    """Save/Print is reachable straight from the assessment results page."""
    _with_ai(monkeypatch, score=44)
    analysis_id = _analysis_id(_png_upload(logged_client, "printable.png",
                                           (110, 190, 130)))

    html = logged_client.get(f"/result/{analysis_id}").data.decode()
    assert "Print / Save as PDF" in html
    assert 'onclick="window.print()"' in html
    # it reuses the report's one print path - no second generator
    assert html.count("window.print()") == 1


def test_full_report_offers_print_and_download(client, logged_client,
                                               monkeypatch):
    _with_ai(monkeypatch, score=44)
    analysis_id = _analysis_id(_png_upload(logged_client, "printdoc.png",
                                           (60, 90, 170)))

    html = logged_client.get(f"/report/{analysis_id}").data.decode()
    assert "Print / Save as PDF" in html
    assert 'onclick="window.print()"' in html
    assert f'href="/report/{analysis_id}/download"' in html


def test_downloaded_report_never_shows_lean_language(client, logged_client,
                                                     monkeypatch):
    """The saved/printable file is user-facing wording too."""
    _with_ai(monkeypatch, score=44)
    analysis_id = _analysis_id(_png_upload(logged_client, "leanfree.png",
                                           (80, 160, 90)))

    body = logged_client.get(f"/report/{analysis_id}/download").data.decode()
    for banned in ("positive lean", "negative lean", "prompt-pair"):
        assert banned not in body, banned
    # The honest technical evidence is still recorded in the detail block.
    assert "openai/clip-vit-base-patch32" in body
    assert "Technical AI Details" in body
