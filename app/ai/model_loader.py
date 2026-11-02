"""Pre-trained CLIP model loader (AI layer).

Responsibilities
----------------
* Import torch / transformers lazily, **only when first needed**.
* Download the pre-trained CLIP weights **once** into ``app/ai/weights/``
  (gitignored) and then always load them from disk, so the app works
  offline afterwards - including during a live review demo.
* Load the model **exactly once per process** and cache it.
* Keep the model on CPU and limit PyTorch to NUM_THREADS threads, so the
  server stays responsive on a low-power machine.
* **Fail soft.** If the dependencies or the weights cannot be loaded, this
  module never raises: it reports ``available=False`` with a plain-English
  reason, and the rule-based computer-vision report continues to work.

Why the weights are downloaded flat (not via ``huggingface_hub``'s cache)
--------------------------------------------------------------------------
The standard cache layout adds ``models--org--name/snapshots/<40-char sha>/``
to the path - about 90 characters. Inside this project directory that pushes
file paths past the Windows 260-character limit, so the download aborts with
``FileNotFoundError``. Writing the files straight into ``app/ai/weights/``
keeps every path well under the limit.

Nothing at module import time imports torch, so ``import app`` still succeeds
even if the AI packages are missing.

This loader performs INFERENCE ONLY. It never trains anything.
"""
from __future__ import annotations

import os
import threading
import time
from typing import Any

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
#: Official OpenAI CLIP weights (zero-shot; no local training involved).
MODEL_ID = "openai/clip-vit-base-patch32"

#: How the model is named on the report / in the database.
MODEL_LABEL = "CLIP (OpenAI CLIP)"

#: Weights live here. Downloaded once, never committed (see .gitignore).
HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(HERE, "weights")

#: Files ``from_pretrained`` needs for this repository.
WEIGHT_FILES = (
    "config.json",
    "preprocessor_config.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "vocab.json",
    "merges.txt",
    "pytorch_model.bin",
)

#: Two threads keeps inference usable without starving the web server.
NUM_THREADS = 2

#: Seconds to wait before retrying a failed download, so a broken network
#: cannot add a timeout to every single upload.
RETRY_COOLDOWN_SECONDS = 60.0

_CHUNK = 1 << 20  # 1 MiB

# ------------------------------------------------------------------
# Process-wide cache
# ------------------------------------------------------------------
_lock = threading.Lock()
_model: Any = None
_processor: Any = None

_available = False
_reason = "Model has not been loaded yet."
_last_attempt = 0.0
_load_seconds: float | None = None
_permanent = False       # dependency problem -> never worth retrying
_downloaded_bytes = 0


def _offline_requested() -> bool:
    """True when the caller has asked never to touch the network."""
    return os.environ.get("DESIGNSCORE_CLIP_OFFLINE", "").strip().lower() in {
        "1", "true", "yes", "on",
    }


def weights_ready() -> bool:
    """True when every required weight file is on disk."""
    return all(
        os.path.isfile(os.path.join(WEIGHTS_DIR, name))
        for name in WEIGHT_FILES
    )


def missing_files() -> list[str]:
    """Required files that are not on disk yet."""
    return [
        name for name in WEIGHT_FILES
        if not os.path.isfile(os.path.join(WEIGHTS_DIR, name))
    ]


def _download_weights() -> str | None:
    """Fetch any missing files. Returns None on success, else a message."""
    global _downloaded_bytes

    pending = missing_files()
    if not pending:
        return None
    if _offline_requested():
        return ("Offline mode is on and the pre-trained CLIP weights have "
                "not been downloaded yet.")

    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    base = f"https://huggingface.co/{MODEL_ID}/resolve/main/"

    try:
        import requests
    except Exception as exc:  # noqa: BLE001
        return f"requests is not available ({exc.__class__.__name__})."

    for name in pending:
        dest = os.path.join(WEIGHTS_DIR, name)
        part = dest + ".part"
        try:
            with requests.get(base + name, stream=True, timeout=(15, 120)) as resp:
                resp.raise_for_status()
                written = 0
                with open(part, "wb") as fh:
                    for chunk in resp.iter_content(chunk_size=_CHUNK):
                        if chunk:
                            fh.write(chunk)
                            written += len(chunk)
            os.replace(part, dest)
            _downloaded_bytes += written
            print(f"    weights: {name} ({written / 1e6:.1f} MB)")
        except Exception as exc:  # noqa: BLE001 - report, never crash
            try:
                if os.path.exists(part):
                    os.remove(part)
            except OSError:
                pass
            return f"could not download {name} ({exc.__class__.__name__}: {exc})"
    return None


def _attempt_load() -> None:
    """Try to import the libraries, fetch weights, load the model.

    Populates the module state. Never raises.
    """
    global _model, _processor, _available, _reason
    global _last_attempt, _load_seconds, _permanent

    _last_attempt = time.time()

    # --- 1. Dependencies ------------------------------------------------
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        _available = False
        _permanent = True
        _reason = f"PyTorch is not available ({exc.__class__.__name__})."
        return

    try:
        from transformers import CLIPModel, CLIPProcessor
    except Exception as exc:  # noqa: BLE001
        _available = False
        _permanent = True
        _reason = f"transformers is not available ({exc.__class__.__name__})."
        return

    # --- 2. Weights -----------------------------------------------------
    problem = _download_weights()
    if problem:
        _available = False
        _permanent = False
        _reason = f"Pre-trained CLIP weights are unavailable: {problem}"
        return

    # --- 3. Load from the local directory (offline-safe) ----------------
    started = time.perf_counter()
    try:
        processor = CLIPProcessor.from_pretrained(
            WEIGHTS_DIR, local_files_only=True)
        model = CLIPModel.from_pretrained(
            WEIGHTS_DIR, local_files_only=True)
    except Exception as exc:  # noqa: BLE001
        _available = False
        _permanent = False
        _model = _processor = None
        _reason = (f"Pre-trained weights could not be loaded "
                   f"({exc.__class__.__name__}: {exc}).")
        return

    # --- 4. Configure for CPU inference ---------------------------------
    try:
        torch.set_num_threads(NUM_THREADS)
    except Exception:  # noqa: BLE001 - non-fatal
        pass

    try:
        model.to("cpu")
    except Exception:  # noqa: BLE001 - non-fatal
        pass
    model.eval()

    _model = model
    _processor = processor
    _load_seconds = time.perf_counter() - started
    _available = True
    _permanent = False
    _reason = "Pre-trained CLIP model loaded."


def load_clip() -> tuple[Any, Any] | None:
    """Return ``(model, processor)`` cached for this process, or ``None``.

    Safe to call on every request: after the first successful load the model
    is read straight from the cache and never reloaded per request.
    """
    with _lock:
        if _available and _model is not None:
            return _model, _processor

        # Never retry a permanent dependency problem; throttle transient
        # failures so a dead network cannot slow every upload.
        if _permanent:
            return None
        if _last_attempt and (time.time() - _last_attempt) < RETRY_COOLDOWN_SECONDS:
            return None

        _attempt_load()
        if not _available:
            return None
        return _model, _processor


def status() -> dict:
    """Machine-readable model state, for the report and the admin panel."""
    with _lock:
        return {
            "available": bool(_available and _model is not None),
            "model": MODEL_ID,
            "label": MODEL_LABEL,
            "reason": _reason,
            "load_seconds": (round(_load_seconds, 2)
                             if _load_seconds is not None else None),
            "threads": NUM_THREADS,
            "device": "cpu",
            "weights_dir": WEIGHTS_DIR,
            "weights_ready": weights_ready(),
            "missing": missing_files(),
            "offline": _offline_requested(),
        }


def warmup() -> dict:
    """Load the model now (tooling / tests / optional startup), and report."""
    load_clip()
    return status()


_warmup_started = False


def start_background_warmup() -> bool:
    """Kick off the one-off model load on a daemon thread.

    Loading costs roughly 30 s on this machine (almost all of it the cold
    ``import torch``), so doing it on the first upload would stall that
    user. Starting it once in the background means the model is ready
    before anyone uploads - while never blocking application start-up and
    never reloading per request.

    Returns True when a warm-up thread was started.
    """
    global _warmup_started
    with _lock:
        if _warmup_started or (_available and _model is not None):
            return False
        _warmup_started = True

    def _worker() -> None:
        try:
            load_clip()
        except Exception:  # noqa: BLE001 - warm-up must never crash the app
            pass

    threading.Thread(target=_worker, daemon=True,
                     name="clip-model-warmup").start()
    return True


def reset() -> None:
    """Drop the cached model and state. For tests only."""
    global _model, _processor, _available, _reason
    global _last_attempt, _load_seconds, _permanent, _downloaded_bytes
    with _lock:
        _model = _processor = None
        _available = False
        _reason = "Model has not been loaded yet."
        _last_attempt = 0.0
        _load_seconds = None
        _permanent = False
        _downloaded_bytes = 0
