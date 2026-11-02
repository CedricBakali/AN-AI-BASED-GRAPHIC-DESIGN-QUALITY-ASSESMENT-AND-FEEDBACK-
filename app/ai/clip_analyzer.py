"""AI-assisted aesthetic assessment - zero-shot CLIP inference.

Implements the prompt-pairing idea of CLIP-IQA
(Wang, Chan & Loy, "Exploring CLIP for Assessing the Look and Feel of
Images", AAAI 2023) using the **pre-trained** OpenAI CLIP model:

    image embedding  vs  two opposing ("antonym") text prompts
                        -> cosine similarity
                        -> softmax over the pair
                        -> probability  -> 0-100 score

Key facts about what this module does and does NOT do
-----------------------------------------------------
* It performs **inference only**. Nothing is trained, fine-tuned or fitted
  on this project's data.
* The score comes from **one** antonym pair, so no weighting scheme is
  invented and no blending with the rule-based score occurs.
* CLIP produces a **general visual / aesthetic impression**. It does NOT
  measure typography, colour, contrast, alignment, layout, white space or
  visual hierarchy. Those design-principle scores come exclusively from the
  deterministic computer-vision analysis in ``app/services/analysis.py``.
* If anything fails, the function returns ``available=False`` with a reason
  instead of raising, so the rule-based report is never blocked.
"""
from __future__ import annotations

import time
from typing import Any

from app.ai import model_loader

MODEL_ID = model_loader.MODEL_ID
MODEL_LABEL = model_loader.MODEL_LABEL

# ------------------------------------------------------------------
# Prompts
# ------------------------------------------------------------------
# The single antonym pair that produces the SCORE. CLIP-IQA scores an image
# by asking which of two opposing descriptions it aligns with more closely.
SCORE_POSITIVE = "a high quality, well designed graphic poster"
SCORE_NEGATIVE = "a low quality, badly designed graphic poster"

# Additional antonym pairs. These are reported as general impressions in the
# explanation ONLY - they never contribute to the score, and they are not
# presented as measurements of any individual design principle.
IMPRESSION_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("Attractiveness",
     "an attractive graphic design", "an unattractive graphic design"),
    ("Professionalism",
     "a professional modern design", "an amateur outdated design"),
    ("Attention-grabbing",
     "a design that grabs attention", "a design that goes unnoticed"),
)


def _unavailable(reason: str, **extra: Any) -> dict:
    """Structured 'no AI today' answer - always safe to store and render."""
    out: dict[str, Any] = {
        "score": None,
        "model": MODEL_ID,
        "label": MODEL_LABEL,
        "available": False,
        "explanation": reason,
        "impressions": [],
        "confidence": None,
        "inference_seconds": None,
    }
    out.update(extra)
    return out


def _sigmoid(x: float) -> float:
    """Numerically stable logistic function."""
    import math
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _softmax_pair(a: float, b: float) -> float:
    """Probability of the FIRST of two values under a 2-way softmax."""
    return _sigmoid(a - b)


def analyze_with_ai(image_path: str) -> dict:
    """Run zero-shot CLIP inference on one uploaded design.

    Returns::

        {
            "score":        int 0-100, or None when unavailable,
            "model":        str  model identifier,
            "available":    bool,
            "explanation":  str  plain-English description,
            "label":        str  display name,
            "impressions":  list extra prompt-pair readings (not scored),
            "confidence":   float | None  0.5-1.0, how decisive the result was,
            "inference_seconds": float | None,
        }

    Never raises and never invents a score: if inference cannot run,
    ``available`` is False and ``score`` is None.
    """
    bundle = model_loader.load_clip()
    if bundle is None:
        return _unavailable(
            "AI assessment is unavailable: "
            + model_loader.status().get("reason", "the model could not be loaded.")
        )

    model, processor = bundle

    try:
        from PIL import Image
        import torch
    except Exception as exc:  # noqa: BLE001
        return _unavailable(
            f"AI assessment is unavailable ({exc.__class__.__name__}).")

    started = time.perf_counter()
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        return _unavailable(
            f"AI assessment could not read the image ({exc.__class__.__name__}).")

    # Build one batch: the image plus every prompt, encoded together.
    prompts: list[str] = [SCORE_POSITIVE, SCORE_NEGATIVE]
    for _label, pos, neg in IMPRESSION_PAIRS:
        prompts.extend((pos, neg))

    try:
        inputs = processor(
            text=prompts,
            images=image,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        with torch.inference_mode():
            image_features = model.get_image_features(
                pixel_values=inputs["pixel_values"])
            text_features = model.get_text_features(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
            # Cosine similarity, then the model's own learned temperature
            # (logit_scale) - a parameter of the pre-trained model, not a
            # constant chosen by us.
            scale = model.logit_scale.exp()

            image_features = image_features / image_features.norm(
                dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(
                dim=-1, keepdim=True)

            sims = (text_features @ image_features.T).squeeze(-1)
            logits = scale * sims

    except Exception as exc:  # noqa: BLE001 - degrade to rule-based only
        return _unavailable(
            "AI assessment could not run inference "
            f"({exc.__class__.__name__}: {exc})")

    values = [float(v) for v in logits.detach().cpu().flatten()]

    # --- The score: one antonym pair, softmax -> 0..100 -----------------
    p_score = _softmax_pair(values[0], values[1])
    score = int(round(p_score * 100))

    # --- Extra impressions (never added to the score) -------------------
    impressions = []
    idx = 2
    for label, pos, neg in IMPRESSION_PAIRS:
        p_pos = _softmax_pair(values[idx], values[idx + 1])
        idx += 2
        impressions.append({
            "label": label,
            "positive": pos,
            "negative": neg,
            "positive_probability": round(p_pos, 3),
            "leaning": "positive" if p_pos >= 0.5 else "negative",
        })

    confidence = round(max(p_score, 1.0 - p_score), 3)
    seconds = round(time.perf_counter() - started, 3)

    explanation = (
        f"The pre-trained CLIP model compared this design against two "
        f"opposing descriptions - \"{SCORE_POSITIVE}\" versus "
        f"\"{SCORE_NEGATIVE}\" - and aligned more closely with the first "
        f"({round(p_score * 100)}% versus {round((1 - p_score) * 100)}%). "
        f"This is a zero-shot AI-assisted aesthetic assessment: the model "
        f"was pre-trained by OpenAI and nothing was trained or fine-tuned "
        f"on this project's data. It gives a general visual impression "
        f"only - it does not measure typography, alignment, white space or "
        f"any other individual design principle. The design-principle "
        f"scores on this report are produced separately by rule-based "
        f"computer-vision analysis."
    )

    return {
        "score": score,
        "model": MODEL_ID,
        "label": MODEL_LABEL,
        "available": True,
        "explanation": explanation,
        "impressions": impressions,
        "confidence": confidence,
        "inference_seconds": seconds,
        "positive_probability": round(p_score, 4),
    }


def ai_status() -> dict:
    """Model availability / identity, for the report and admin monitoring."""
    st = model_loader.status()
    return {
        "available": st.get("available", False),
        "model": st.get("model", MODEL_ID),
        "label": st.get("label", MODEL_LABEL),
        "reason": st.get("reason", ""),
        "load_seconds": st.get("load_seconds"),
        "device": st.get("device", "cpu"),
        "method": "zero-shot CLIP-IQA style prompt pairing (inference only)",
    }
