"""NIMA: the second, independent AI opinion.

Two kinds of test here:

* Contract tests that always run - fail-soft behaviour, storage, wording,
  and the architectural guarantee that NIMA is never merged into another
  score.
* A real-inference test that runs genuine forward passes through the
  released AVA checkpoint when it has been downloaded. It SKIPS rather
  than fakes when the weights are absent, so a number is never invented.
"""
from __future__ import annotations

import io
import os
import pathlib
import re

import pytest

from app.ai import nima_analyzer

ROOT = pathlib.Path(__file__).resolve().parent.parent


# ------------------------------------------------------------- helpers

def _real_image(path, size=(320, 240)):
    """A genuine, non-trivial raster - not a stub, not a constant."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", size, (30, 40, 70))
    draw = ImageDraw.Draw(img)
    for x in range(0, size[0], 24):
        draw.rectangle([x, 0, x + 12, size[1]], fill=(200, 120 + x % 90, 60))
    draw.ellipse([40, 40, 200, 180], fill=(240, 240, 240))
    draw.rectangle([210, 60, 300, 200], fill=(20, 150, 140))
    img.save(path, format="PNG")
    return str(path)


def _upload(client, name="nima.png", colour=(200, 60, 60)):
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


def _patch_nima(monkeypatch, **overrides):
    """Stand in for NIMA without ever fabricating a stored score silently."""
    payload = {
        "available": True,
        "score": 6.4,
        "model": nima_analyzer.MODEL_ID,
        "label": nima_analyzer.MODEL_LABEL,
        "reason": "ok",
        "seconds": 0.5,
        "top_rating": 6,
    }
    payload.update(overrides)
    monkeypatch.setattr(nima_analyzer, "analyze_with_nima", lambda _p: payload)


# ------------------------------------------------- contract / fail-soft

def test_nima_returns_the_documented_shape(tmp_path):
    out = nima_analyzer.analyze_with_nima(_real_image(tmp_path / "a.png"))
    for key in ("available", "score", "model", "label", "reason",
                "seconds", "top_rating"):
        assert key in out, key
    assert out["model"] == nima_analyzer.MODEL_ID
    assert out["label"] == "NIMA (Neural Image Assessment)"


def test_nima_is_fail_soft_when_the_model_cannot_load(tmp_path, monkeypatch):
    """A missing model must report, never raise - a report must still save."""
    monkeypatch.setattr(nima_analyzer, "load_nima", lambda: None)
    out = nima_analyzer.analyze_with_nima(_real_image(tmp_path / "b.png"))
    assert out["available"] is False
    assert out["score"] is None
    assert out["reason"], "an unavailable model must say why"


def test_nima_reports_a_missing_file_instead_of_raising(tmp_path, monkeypatch):
    monkeypatch.setattr(nima_analyzer, "load_nima", lambda: None)
    out = nima_analyzer.analyze_with_nima(str(tmp_path / "nope.png"))
    assert out["available"] is False
    assert out["score"] is None


def test_nima_status_is_machine_readable():
    info = nima_analyzer.status()
    assert info["scale"] == "0-10"
    assert info["device"] == "cpu"
    assert set(info) >= {"available", "model", "label", "reason",
                         "weights_ready", "offline"}


def test_nima_weights_can_never_be_committed_to_git():
    """Model weights live in a gitignored folder by design."""
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "app/ai/weights/" in ignore
    assert nima_analyzer.WEIGHTS_DIR.replace("\\", "/").endswith(
        "app/ai/weights")


def test_an_analysis_saves_even_when_nima_raises(client, logged_client,
                                                 monkeypatch):
    """Hard fail-soft: an exception must not stop the assessment."""
    def _boom(_path):
        raise RuntimeError("simulated NIMA crash")

    monkeypatch.setattr(nima_analyzer, "analyze_with_nima", _boom)
    analysis_id = _analysis_id(_upload(logged_client))

    from app.models import database
    record = database.get_analysis(analysis_id, _user_id(logged_client))
    assert record["nima_score"] is None
    assert record["nima_model"] is None

    html = logged_client.get(f"/result/{analysis_id}").data.decode()
    assert "Second AI Opinion" in html
    assert "NIMA could not look at this design" in html


def _user_id(client) -> int:
    from app.models import database
    with database.get_connection() as conn:
        row = conn.execute(
            "SELECT id FROM users WHERE email = ?", ("tester@test.com",)
        ).fetchone()
    assert row is not None
    return int(row["id"])


# ------------------------------------------- separation from other scores

def test_nima_is_stored_in_its_own_columns(client, logged_client,
                                            monkeypatch):
    """nima_score never lands in ai_score, and vice versa."""
    _patch_nima(monkeypatch, score=7.25)
    analysis_id = _analysis_id(_upload(logged_client, "sep.png",
                                       (10, 140, 200)))

    from app.models import database
    record = database.get_analysis(analysis_id, _user_id(logged_client))
    assert record["nima_score"] == pytest.approx(7.25)
    # CLIP's score is a different quantity on a different scale and must
    # not have absorbed NIMA's number.
    if record["ai_score"] is not None:
        assert record["nima_score"] != record["ai_score"]


def test_nima_is_never_rescaled_to_100(client, logged_client, monkeypatch):
    """0-10 stays 0-10: a /100 rendering would invite merging."""
    _patch_nima(monkeypatch, score=6.4)
    analysis_id = _analysis_id(_upload(logged_client, "scale.png",
                                       (60, 90, 160)))

    html = logged_client.get(f"/result/{analysis_id}").data.decode()
    assert "Second AI Opinion" in html
    assert "6.4" in html
    assert "/10" in html
    assert "6.4/100" not in html
    assert "6.4/ 100" not in html


def test_nima_block_states_it_is_not_combined(client, logged_client,
                                              monkeypatch):
    _patch_nima(monkeypatch, score=6.4)
    analysis_id = _analysis_id(_upload(logged_client, "combine.png",
                                       (90, 40, 120)))

    html = logged_client.get(f"/result/{analysis_id}").data.decode()
    assert "not added to, averaged with" in html
    # and the two original scores are still explained as separate
    assert "A separate AI opinion" in html
    assert "Design Quality Score" in html


def test_nima_wording_never_leaks_technical_language(client, logged_client,
                                                     monkeypatch):
    _patch_nima(monkeypatch, score=6.4)
    analysis_id = _analysis_id(_upload(logged_client, "wordy.png",
                                       (30, 160, 110)))

    for url in (f"/result/{analysis_id}", f"/report/{analysis_id}",
                f"/report/{analysis_id}/download"):
        body = logged_client.get(url).data.decode()
        for banned in ("positive lean", "negative lean", "prompt-pair"):
            assert banned not in body, f"{banned} in {url}"


def test_report_and_download_show_the_nima_block(client, logged_client,
                                                 monkeypatch):
    _patch_nima(monkeypatch, score=6.4)
    analysis_id = _analysis_id(_upload(logged_client, "doc.png",
                                       (140, 90, 30)))

    page = logged_client.get(f"/report/{analysis_id}").data.decode()
    assert "Second AI Opinion" in page
    assert "6.4" in page
    assert "/10" in page

    saved = logged_client.get(f"/report/{analysis_id}/download").data.decode()
    assert "Second AI Opinion" in saved
    assert "6.4" in saved


def test_nima_unavailable_is_reported_not_hidden(client, logged_client,
                                                 monkeypatch):
    """When NIMA simply did not run, the page says so plainly."""
    _patch_nima(monkeypatch, available=False, score=None, reason="weights off")

    analysis_id = _analysis_id(_upload(logged_client, "unavail.png",
                                       (120, 60, 160)))

    from app.models import database
    record = database.get_analysis(analysis_id, _user_id(logged_client))
    assert record["nima_score"] is None

    html = logged_client.get(f"/result/{analysis_id}").data.decode()
    assert "Second AI Opinion" in html
    assert "NIMA could not look at this design" in html
    # Nothing is shown as a score when no score exists - the "/10" marker
    # belongs to the nima-den span only, so test that element directly
    # (a naive "/10" check would false-positive on "54/100").
    assert 'class="nima-den"' not in html
    assert 'class="nima-num"' not in html


# ------------------------------------------------------ REAL INFERENCE

@pytest.mark.skipif(
    not nima_analyzer.weights_ready(),
    reason="released NIMA checkpoint has not been downloaded",
)
def test_nima_real_inference_produces_a_real_score(tmp_path):
    """Genuine forward pass: released weights, real pixels, no stub.

    This is the test that stops a placeholder ever standing in for NIMA.
    """
    from app.ai import model_loader  # noqa: F401 - imports share weights dir

    assert nima_analyzer.weights_ready(), "checkpoint must be on disk"
    assert os.path.getsize(nima_analyzer.weights_path()) > 50_000_000, \
        "the released checkpoint is ~59 MB; a placeholder would be tiny"

    path = _real_image(tmp_path / "real.png", size=(640, 480))
    out = nima_analyzer.analyze_with_nima(path)

    assert out["available"] is True, out.get("reason")
    assert isinstance(out["score"], float)
    assert 0.0 <= out["score"] <= 10.0, "NIMA reports its native 0-10 scale"
    assert out["score"] < 100, "NIMA must never be rescaled to 100"
    assert out["top_rating"] in range(1, 11)
    assert out["seconds"] is not None and out["seconds"] > 0
    assert out["model"] == nima_analyzer.MODEL_ID


@pytest.mark.skipif(
    not nima_analyzer.weights_ready(),
    reason="released NIMA checkpoint has not been downloaded",
)
def test_nima_real_inference_is_deterministic(tmp_path):
    """The same pixels must give the same answer - no sampling involved."""
    path = _real_image(tmp_path / "twice.png", size=(400, 300))
    first = nima_analyzer.analyze_with_nima(path)
    second = nima_analyzer.analyze_with_nima(path)
    assert first["available"] and second["available"]
    assert first["score"] == second["score"]
    assert first["top_rating"] == second["top_rating"]


# -------------------------------------- the designer-facing message first
#
# The block is written for a designer, not for someone who wants to read
# about NIMA: it leads with what NIMA makes of THIS image plus one next
# step, and pushes the "what is NIMA" explanation into small print.

BAND_SAMPLES = (
    (8.2, "Strong"),
    (6.4, "Good"),
    (5.0, "Average"),
    (3.4, "Below average"),
    (1.9, "Weak"),
)


@pytest.mark.parametrize("score,band", BAND_SAMPLES)
def test_nima_block_leads_with_a_verdict_and_a_next_step(
        client, logged_client, monkeypatch, score, band):
    _patch_nima(monkeypatch, score=score)
    analysis_id = _analysis_id(_upload(logged_client, f"band{score}.png",
                                       (30 + int(score), 120, 90)))
    html = logged_client.get(f"/result/{analysis_id}").data.decode()

    assert f'class="band nima-band">{band}<' in html

    verdict = re.search(r'class="nima-feedback">([^<]+)</p>', html)
    assert verdict, "the block must lead with a message about the design"
    assert "design" in verdict.group(1).lower()
    assert len(verdict.group(1).strip()) > 40

    advice = re.search(
        r'class="nima-advice">\s*<span class="nima-next">Next step</span>'
        r"\s*([^<]+)", html)
    assert advice, "every band needs a concrete next step"
    assert len(advice.group(1).strip()) > 40

    fine = re.search(r'class="nima-fine">([^<]+)</p>', html)
    assert fine, "the explanation is still shown, as small print"
    assert "different network" in fine.group(1)


def test_nima_explanation_sits_below_the_message(client, logged_client,
                                                 monkeypatch):
    """What NIMA is must never appear above what NIMA says about the art."""
    _patch_nima(monkeypatch, score=6.4)
    analysis_id = _analysis_id(_upload(logged_client, "order.png",
                                       (70, 60, 150)))
    html = logged_client.get(f"/result/{analysis_id}").data.decode()

    message_at = html.index('class="nima-feedback"')
    fine_at = html.index('class="nima-fine"')
    explanation_at = html.index("different network")
    assert message_at < fine_at < explanation_at


def test_report_and_download_lead_with_the_message_too(client, logged_client,
                                                       monkeypatch):
    """The printable report gets the same hierarchy as the result page."""
    _patch_nima(monkeypatch, score=6.4)
    analysis_id = _analysis_id(_upload(logged_client, "docmsg.png",
                                       (150, 110, 40)))

    for url in (f"/report/{analysis_id}", f"/report/{analysis_id}/download"):
        body = logged_client.get(url).data.decode()
        verdict = re.search(r'class="nima-feedback">([^<]+)</p>', body)
        assert verdict, url
        assert "design" in verdict.group(1).lower()
        assert 'class="nima-next">Next step<' in body
        assert body.index('class="nima-feedback"') < \
            body.index("different network")
        assert 'class="nima-fine"' in body
