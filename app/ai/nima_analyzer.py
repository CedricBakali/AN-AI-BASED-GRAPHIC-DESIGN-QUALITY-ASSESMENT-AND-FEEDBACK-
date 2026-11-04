"""NIMA (Neural Image Assessment) - a SECOND, independent AI opinion.

What this is
------------
NIMA (Talebi & Milanfar, CVPR 2018) predicts a probability distribution over
ratings 1..10 for an image and reports the *expected* rating (mean opinion
score). The weights used here are the released AVA checkpoint
``NIMA_VGG16_ava-dc4e8265.pth`` published by the IQA-PyTorch project.

What this is deliberately NOT
-----------------------------
* It does not touch the Design Quality Score. That score stays the mean of
  the eight computer-vision principles in ``app/services/analysis.py``.
* It is never averaged, blended or weighted together with CLIP's opinion,
  in the database, in Python, or in any template.
* It reports its native **0-10** scale on purpose. Rescaling it to 0-100
  would invite exactly the comparison we refuse to make, and would let it
  be silently folded into a score that means something else.

Why not pyiqa
-------------
`pyiqa` is the usual way to get NIMA, but it requires ``transformers>=5.0``
(this project pins 4.57.6 for CLIP) and pulls in bitsandbytes, tensorboard,
datasets, facexlib, pre-commit, ruff and yapf. Installing it would risk the
one model that is actually in production. NIMA only needs ``timm`` for its
backbone, so this module builds the network itself from the released
checkpoint.

Runtime shape (matches ``model_loader``)
----------------------------------------
Imported lazily, weights fetched once into the gitignored ``weights/``
directory, model built once per process, CPU only, and **every failure is
reported as ``available=False`` with a plain-English reason** - never raised
- so a missing dependency or dead network can never stop the report.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any

# ------------------------------------------------------------------
# Identity and weights
# ------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))

#: Share CLIP's gitignored weight directory (see .gitignore).
WEIGHTS_DIR = os.path.join(HERE, "weights")

WEIGHT_FILENAME = "NIMA_VGG16_ava-dc4e8265.pth"

#: Stable, human-readable id recorded on the report so a later model swap
#: cannot silently rewrite what an old report claims produced the number.
MODEL_ID = "chaofengc/IQA-PyTorch-Weights: NIMA_VGG16_ava (AVA)"

MODEL_LABEL = "NIMA (Neural Image Assessment)"

#: Two mirrors, same file, same checksum - so one hosting outage does not
#: take the feature down. HuggingFace first, GitHub releases as fallback.
BASE_URLS = (
    "https://huggingface.co/chaofengc/IQA-PyTorch-Weights/resolve/main/",
    "https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/",
)

#: NIMA predicts one probability per rating 1..10.
NUM_CLASSES = 10

#: VGG16 backbone at the input size timm's vgg16 config declares.
BASE_MODEL_NAME = "vgg16"
INPUT_SIZE = 224

#: Two threads, same budget as CLIP, so inference stays responsive on CPU.
NUM_THREADS = 2

RETRY_COOLDOWN_SECONDS = 60.0

_CHUNK = 1 << 20  # 1 MiB

# ------------------------------------------------------------------
# Process-wide cache
# ------------------------------------------------------------------

_lock = threading.Lock()
_model: Any = None

_available = False
_reason = "NIMA has not been loaded yet."
_last_attempt = 0.0
_load_seconds: float | None = None
_permanent = False      # dependency problem -> never worth retrying


def weights_path() -> str:
    return os.path.join(WEIGHTS_DIR, WEIGHT_FILENAME)


def weights_ready() -> bool:
    """True when the released checkpoint is on disk."""
    return os.path.isfile(weights_path())


def _offline_requested() -> bool:
    """True when the caller has asked never to touch the network.

    ``DESIGNSCORE_CLIP_OFFLINE`` is this project's existing "no network for
    model downloads" switch, so NIMA honours it too; ``DESIGNSCORE_NIMA_OFFLINE``
    can also be used to freeze NIMA alone.
    """
    off = {"1", "true", "yes", "on"}
    return (
        os.environ.get("DESIGNSCORE_CLIP_OFFLINE", "").strip().lower() in off
        or os.environ.get("DESIGNSCORE_NIMA_OFFLINE", "").strip().lower() in off
    )


# ------------------------------------------------------------------
# Download
# ------------------------------------------------------------------

def _download_weights() -> str | None:
    """Fetch the checkpoint once. Returns None on success, else a message."""
    dest = weights_path()
    if os.path.isfile(dest):
        return None
    if _offline_requested():
        return ("Offline mode is on and the NIMA weights have not been "
                "downloaded yet.")

    try:
        import requests
    except Exception as exc:  # noqa: BLE001
        return f"requests is not available ({exc.__class__.__name__})."

    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    part = dest + ".part"
    errors: list[str] = []

    for base in BASE_URLS:
        try:
            with requests.get(base + WEIGHT_FILENAME, stream=True,
                              timeout=(15, 180)) as resp:
                resp.raise_for_status()
                written = 0
                with open(part, "wb") as fh:
                    for chunk in resp.iter_content(chunk_size=_CHUNK):
                        if chunk:
                            fh.write(chunk)
                            written += len(chunk)
            os.replace(part, dest)
            print(f"    nima weights: {WEIGHT_FILENAME} ({written / 1e6:.1f} MB)")
            return None
        except Exception as exc:  # noqa: BLE001 - try the next mirror
            errors.append(f"{base.split('/')[2]}: {exc.__class__.__name__}")

    try:
        if os.path.exists(part):
            os.remove(part)
    except OSError:
        pass
    return f"could not download {WEIGHT_FILENAME} ({'; '.join(errors)})"


# ------------------------------------------------------------------
# Model construction
# ------------------------------------------------------------------

def _build_model():
    """Import torch/timm, build the network, load the released weights.

    Returns ``(model, load_seconds)`` or ``(None, message)``. Never raises.
    """
    started = time.perf_counter()

    try:
        import torch
        import torch.nn as nn
        import torchvision.transforms as T
        import timm
    except Exception as exc:  # noqa: BLE001
        return None, (f"timm/torchvision is not available "
                      f"({exc.__class__.__name__}).")

    class _NIMA(nn.Module):
        """Network only: vgg16 features -> pool -> 10-way softmax."""

        def __init__(self) -> None:
            super().__init__()
            # pretrained=False on purpose: the released NIMA checkpoint
            # already contains the fine-tuned backbone, so the separate
            # ImageNet download is both redundant and a demo-day hazard.
            self.base_model = timm.create_model(
                BASE_MODEL_NAME, pretrained=False, features_only=True)
            self.global_pool = nn.AdaptiveAvgPool2d(1)
            in_ch = self.base_model.feature_info.channels()[-1]
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(p=0.0),
                nn.Linear(in_features=in_ch, out_features=NUM_CLASSES),
                nn.Softmax(dim=-1),
            )
            mean = self.base_model.pretrained_cfg["mean"]
            std = self.base_model.pretrained_cfg["std"]
            self.default_mean = torch.Tensor(mean).view(1, 3, 1, 1)
            self.default_std = torch.Tensor(std).view(1, 3, 1, 1)

        def preprocess(self, x):
            if not self.training:
                x = T.functional.resize(x, INPUT_SIZE)
                x = T.functional.center_crop(x, INPUT_SIZE)
            return (x - self.default_mean.to(x)) / self.default_std.to(x)

        def forward(self, x):
            x = self.preprocess(x)
            x = self.base_model(x)[-1]
            x = self.global_pool(x)
            dist = self.classifier(x)
            # Expected rating over classes 1..10 (pyiqa's dist_to_mos).
            mos = dist * torch.arange(1, NUM_CLASSES + 1).to(dist)
            return mos.sum(dim=-1), dist

    model = _NIMA()

    path = weights_path()
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "params" in state:
            state = state["params"]
        state = {
            (k[7:] if k.startswith("module.") else k): v
            for k, v in state.items()
        }
        model.load_state_dict(state, strict=True)
    except Exception as exc:  # noqa: BLE001
        return None, (f"the NIMA checkpoint could not be loaded "
                      f"({exc.__class__.__name__}: {exc}).")

    try:
        torch.set_num_threads(NUM_THREADS)
    except Exception:  # noqa: BLE001 - non-fatal
        pass

    model.to("cpu")
    model.eval()
    return model, round(time.perf_counter() - started, 2)


def _attempt_load() -> None:
    """Try to load NIMA. Populates module state. Never raises."""
    global _model, _available, _reason, _last_attempt, _permanent, _load_seconds

    _last_attempt = time.time()

    problem = _download_weights()
    if problem:
        _available = False
        _permanent = False
        _reason = f"NIMA weights are unavailable: {problem}"
        return

    model, info = _build_model()
    if model is None:
        # ``info`` carries the failure message in this branch.
        _available = False
        _model = None
        _permanent = "not available" in str(info)
        _reason = str(info)
        return

    _model = model
    _load_seconds = float(info)
    _available = True
    _permanent = False
    _reason = "NIMA model loaded."


def load_nima() -> Any:
    """Return the cached NIMA model, or None. Safe to call per request."""
    with _lock:
        if _available and _model is not None:
            return _model

        if _permanent:
            return None
        if _last_attempt and (time.time() - _last_attempt) < RETRY_COOLDOWN_SECONDS:
            return None

        _attempt_load()
        return _model if _available else None


def status() -> dict:
    """Machine-readable state, for the report and the admin panel."""
    with _lock:
        return {
            "available": bool(_available and _model is not None),
            "model": MODEL_ID,
            "label": MODEL_LABEL,
            "reason": _reason,
            "scale": "0-10",
            "load_seconds": (_load_seconds if _load_seconds is not None else None),
            "device": "cpu",
            "weights_dir": WEIGHTS_DIR,
            "weights_ready": weights_ready(),
            "offline": _offline_requested(),
        }


# ------------------------------------------------------------------
# Inference
# ------------------------------------------------------------------

def analyze_with_nima(path: str) -> dict:
    """Run real NIMA inference on one image.

    Returns a dict that is always present and never raises::

        {"available": bool, "score": float 0-10 | None, "model": str,
         "label": str, "reason": str, "seconds": float | None,
         "top_rating": int | None}

    ``score`` is on NIMA's native 0-10 scale and is stored and shown as
    such. It is not comparable with the 0-100 Design Quality Score or the
    0-100 CLIP opinion, and must never be combined with either.
    """
    out: dict = {
        "available": False,
        "score": None,
        "model": MODEL_ID,
        "label": MODEL_LABEL,
        "reason": "",
        "seconds": None,
        "top_rating": None,
    }

    try:
        import torch
        import numpy as np
        from PIL import Image
    except Exception as exc:  # noqa: BLE001
        out["reason"] = f"NIMA dependencies are missing ({exc.__class__.__name__})."
        return out

    model = load_nima()
    if model is None:
        with _lock:
            out["reason"] = _reason
        return out

    try:
        image = Image.open(path).convert("RGB")
        array = np.asarray(image).astype(np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    except Exception as exc:  # noqa: BLE001
        out["reason"] = (f"NIMA could not open that image "
                         f"({exc.__class__.__name__}).")
        return out

    try:
        started = time.perf_counter()
        with torch.no_grad():
            mos, dist = model(tensor)
        elapsed = time.perf_counter() - started
    except Exception as exc:  # noqa: BLE001 - degrade, never crash a report
        out["reason"] = (f"NIMA could not look at that design "
                         f"({exc.__class__.__name__}: {exc}).")
        return out

    try:
        score = float(mos.reshape(-1)[0].item())
        top_rating = int(torch.argmax(dist, dim=-1).reshape(-1)[0].item()) + 1
    except Exception as exc:  # noqa: BLE001
        out["reason"] = f"NIMA returned an unreadable result ({exc.__class__.__name__})."
        return out

    out.update({
        "available": True,
        "score": round(max(0.0, min(10.0, score)), 2),
        "reason": "ok",
        "seconds": round(elapsed, 2),
        "top_rating": top_rating,
    })
    return out


def reset() -> None:
    """Drop the cached model and state. For tests only."""
    global _model, _available, _reason, _last_attempt, _load_seconds, _permanent
    with _lock:
        _model = None
        _available = False
        _reason = "NIMA has not been loaded yet."
        _last_attempt = 0.0
        _load_seconds = None
        _permanent = False
