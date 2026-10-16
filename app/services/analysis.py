"""
DesignScore - rule-based graphic design quality analyzer.

Each design principle is scored 0-100 using well-established, deterministic
rules (WCAG contrast ratios, colour-wheel harmony, alignment grids, white-space
balance) rather than a trained model. Every score comes with plain-language
suggestions for improvement.
"""
from __future__ import annotations

import colorsys
import math
from dataclasses import dataclass, field
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image

from app.services.scoring import compute_overall, grade_for


# --------------------------------------------------------------------------- #
# Data containers
# --------------------------------------------------------------------------- #
@dataclass
class CategoryResult:
    name: str
    score: int                       # 0-100
    summary: str
    suggestions: List[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "score": self.score,
            "summary": self.summary,
            "suggestions": self.suggestions,
        }


@dataclass
class AnalysisResult:
    overall: int
    grade: str
    categories: List[CategoryResult]
    palette: List[str]               # hex strings of dominant colours

    def as_dict(self) -> dict:
        return {
            "overall": self.overall,
            "grade": self.grade,
            "categories": [c.as_dict() for c in self.categories],
            "palette": self.palette,
        }


# --------------------------------------------------------------------------- #
# Low-level helpers
# --------------------------------------------------------------------------- #
MAX_SIDE = 1000  # downscale large uploads for speed / stable metrics


def load_image(path: str) -> np.ndarray:
    """Load an image as an RGB uint8 numpy array, downscaled if very large."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = min(1.0, MAX_SIDE / max(w, h))
    if scale < 1.0:
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))))
    return np.asarray(img, dtype=np.uint8)


def _rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    return "#{:02X}{:02X}{:02X}".format(int(rgb[0]), int(rgb[1]), int(rgb[2]))


def _relative_luminance(rgb: np.ndarray) -> float:
    """WCAG 2.1 relative luminance for an sRGB colour (0-255 channels)."""
    srgb = rgb.astype(np.float64) / 255.0
    linear = np.where(srgb <= 0.03928, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)
    return float(0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2])


def _contrast_ratio(c1: np.ndarray, c2: np.ndarray) -> float:
    """WCAG contrast ratio between two colours (1.0 - 21.0)."""
    l1 = _relative_luminance(c1)
    l2 = _relative_luminance(c2)
    lighter, darker = max(l1, l2), min(l1, l2)
    return (lighter + 0.05) / (darker + 0.05)


def dominant_colors(img: np.ndarray, k: int = 5) -> List[Tuple[Tuple[int, int, int], float]]:
    """Return up to k dominant colours as (rgb, fraction) using k-means."""
    pixels = img.reshape(-1, 3).astype(np.float32)
    # sample for speed
    if pixels.shape[0] > 40000:
        idx = np.random.default_rng(42).choice(pixels.shape[0], 40000, replace=False)
        pixels = pixels[idx]
    k = int(min(k, max(1, len(np.unique(pixels, axis=0)))))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(
        pixels, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS
    )
    labels = labels.flatten()
    counts = np.bincount(labels, minlength=k).astype(np.float64)
    fracs = counts / counts.sum()
    order = np.argsort(fracs)[::-1]
    out = []
    for i in order:
        rgb = tuple(int(round(v)) for v in centers[i])
        out.append((rgb, float(fracs[i])))
    return out


# --------------------------------------------------------------------------- #
# Principle 1 - Contrast (WCAG)
# --------------------------------------------------------------------------- #
def score_contrast(img: np.ndarray, palette) -> CategoryResult:
    """Score text/background legibility using the WCAG contrast ratio between
    the two most prominent colours plus overall tonal spread."""
    suggestions: List[str] = []
    if len(palette) >= 2:
        c1 = np.array(palette[0][0])
        c2 = np.array(palette[1][0])
        ratio = _contrast_ratio(c1, c2)
    else:
        ratio = 1.0

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float64)
    spread = (np.percentile(gray, 95) - np.percentile(gray, 5)) / 255.0

    # Map the 1-21 ratio onto 0-100. 4.5:1 (WCAG AA) ~ 70, 7:1 (AAA) ~ 90.
    ratio_score = min(100.0, max(0.0, (math.log(ratio) / math.log(21)) * 115))
    score = int(round(0.75 * ratio_score + 0.25 * spread * 100))
    score = max(0, min(100, score))

    summary = f"Dominant colour contrast ratio is {ratio:.1f}:1."
    if ratio < 3.0:
        suggestions.append(
            "Contrast is very low. Darken your text or lighten the background so "
            "the two main colours reach at least a 4.5:1 ratio (WCAG AA)."
        )
    elif ratio < 4.5:
        suggestions.append(
            "Contrast is borderline. Push the text and background further apart "
            "in brightness to pass the 4.5:1 WCAG AA standard for body text."
        )
    elif ratio < 7.0:
        suggestions.append(
            "Contrast passes AA. For small text or accessibility-critical work, "
            "aim for 7:1 (AAA)."
        )
    else:
        suggestions.append("Excellent contrast - text should be easy to read.")
    if spread < 0.4:
        suggestions.append(
            "The image is tonally flat. Add brighter highlights or darker "
            "shadows to create visual hierarchy."
        )
    return CategoryResult("Contrast", score, summary, suggestions)


# --------------------------------------------------------------------------- #
# Principle 2 - Colour harmony (colour wheel)
# --------------------------------------------------------------------------- #
def _hue_degrees(rgb) -> float:
    r, g, b = [v / 255.0 for v in rgb]
    h, _, _ = colorsys.rgb_to_hsv(r, g, b)
    return h * 360.0


def score_color_harmony(img: np.ndarray, palette) -> CategoryResult:
    """Reward palettes that match a recognised colour-wheel relationship
    (monochrome, analogous, complementary, triadic) and penalise clashing
    or overly busy palettes."""
    suggestions: List[str] = []
    sig = [p for p in palette if p[1] >= 0.05] or palette[:1]
    hues = [_hue_degrees(p[0]) for p in sig]
    sats = []
    for p in sig:
        r, g, b = [v / 255.0 for v in p[0]]
        _, s, _ = colorsys.rgb_to_hsv(r, g, b)
        sats.append(s)
    avg_sat = float(np.mean(sats)) if sats else 0.0

    def hue_dist(a, b):
        d = abs(a - b) % 360
        return min(d, 360 - d)

    scheme = "neutral"
    score = 70
    if len(hues) == 1 or max(hues) - min(hues) < 1e-6:
        scheme = "monochrome"
        score = 82
    else:
        dists = [hue_dist(hues[i], hues[j])
                 for i in range(len(hues)) for j in range(i + 1, len(hues))]
        max_d = max(dists)
        mean_d = float(np.mean(dists))
        if max_d <= 40:
            scheme, score = "analogous", 88
        elif any(150 <= d <= 210 for d in dists) and len(hues) <= 3:
            scheme, score = "complementary", 85
        elif 95 <= mean_d <= 135:
            scheme, score = "triadic", 83
        elif len(hues) >= 5 and mean_d > 60:
            scheme, score = "busy / unstructured", 58
        else:
            scheme, score = "mixed", 70

    if avg_sat > 0.85 and scheme not in ("monochrome",):
        score -= 10
        suggestions.append(
            "Colours are highly saturated. Mute one or two of them so they do "
            "not compete for attention."
        )
    if avg_sat < 0.08:
        suggestions.append(
            "The palette is almost greyscale. A single accent colour would add "
            "focus without hurting harmony."
        )

    score = max(0, min(100, int(round(score))))
    summary = f"Palette reads as a {scheme} scheme."
    if scheme in ("busy / unstructured", "mixed"):
        suggestions.append(
            "Reduce to 2-3 core colours chosen from a colour-wheel relationship "
            "(analogous, complementary, or triadic) for a more cohesive look."
        )
    elif not suggestions:
        suggestions.append("Colour relationships look balanced and intentional.")
    return CategoryResult("Color Harmony", score, summary, suggestions)


# --------------------------------------------------------------------------- #
# Principle 3 - White space / balance
# --------------------------------------------------------------------------- #
def score_white_space(img: np.ndarray) -> CategoryResult:
    """Estimate how much of the canvas is 'quiet' (low-detail) space and how
    evenly the busy content is distributed across the four quadrants."""
    suggestions: List[str] = []
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 80, 180)
    busy_fraction = float((edges > 0).mean())
    quiet = 1.0 - busy_fraction

    # Ideal breathing room: roughly 55-85% quiet space.
    if 0.55 <= quiet <= 0.88:
        space_score = 90
    elif quiet > 0.88:
        space_score = 70  # possibly too empty
    else:
        space_score = max(30, int(100 * quiet))

    # Balance: compare content density across quadrants.
    h, w = edges.shape
    q = [
        edges[: h // 2, : w // 2].mean(),
        edges[: h // 2, w // 2:].mean(),
        edges[h // 2:, : w // 2].mean(),
        edges[h // 2:, w // 2:].mean(),
    ]
    q = np.array(q) + 1e-6
    balance = 1.0 - (q.std() / q.mean()) / 2.0
    balance = max(0.0, min(1.0, balance))
    score = int(round(0.6 * space_score + 0.4 * balance * 100))
    score = max(0, min(100, score))

    summary = (f"About {quiet*100:.0f}% of the canvas is open space; "
               f"content balance {balance*100:.0f}%.")
    if quiet < 0.5:
        suggestions.append(
            "The layout feels crowded. Remove secondary elements or increase "
            "margins and spacing so the design can breathe."
        )
    if quiet > 0.9:
        suggestions.append(
            "There may be too much empty space. Enlarge the focal element or "
            "tighten the layout so it does not feel sparse."
        )
    if balance < 0.6:
        suggestions.append(
            "Content is unevenly distributed. Shift elements to balance visual "
            "weight across the composition."
        )
    if not suggestions:
        suggestions.append("Good balance of content and white space.")
    return CategoryResult("White Space", score, summary, suggestions)


# --------------------------------------------------------------------------- #
# Principle 4 - Alignment
# --------------------------------------------------------------------------- #
def score_alignment(img: np.ndarray) -> CategoryResult:
    """Reward content whose edges line up to a small number of vertical and
    horizontal guides (a sign of a tidy grid) rather than scattered freely."""
    suggestions: List[str] = []
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 80, 180)
    h, w = edges.shape

    col_profile = edges.sum(axis=0).astype(np.float64)
    row_profile = edges.sum(axis=1).astype(np.float64)

    def guide_strength(profile, length):
        if profile.max() <= 0:
            return 0.0
        norm = profile / profile.max()
        strong = norm > 0.5           # prominent guide lines
        # Fraction of structure concentrated on few strong guides is good.
        return float(norm[strong].sum() / (norm.sum() + 1e-6))

    v = guide_strength(col_profile, w)
    hz = guide_strength(row_profile, h)
    score = int(round((v + hz) / 2.0 * 100))
    score = max(20, min(100, score + 20))  # offset: diffuse edges still ok

    summary = "Edges align to clear vertical and horizontal guides."
    if score < 55:
        summary = "Elements do not share common alignment guides."
        suggestions.append(
            "Align text and graphics to a shared grid or common edges "
            "(left-align a text block, match element margins) for a tidier look."
        )
    elif score < 75:
        suggestions.append(
            "Alignment is mostly fine. Snap any stray elements to the same "
            "margins or baseline to sharpen the layout."
        )
    else:
        suggestions.append("Elements are well aligned to a consistent grid.")
    return CategoryResult("Alignment", score, summary, suggestions)


# --------------------------------------------------------------------------- #
# Principle 5 - Typography (text region size / count heuristic)
# --------------------------------------------------------------------------- #
def _text_regions(gray: np.ndarray):
    """Rough text-like region detection using MSER; returns list of (w, h)."""
    try:
        mser = cv2.MSER_create()
        regions, _ = mser.detectRegions(gray)
    except Exception:
        return []
    boxes = []
    for r in regions:
        x, y, bw, bh = cv2.boundingRect(r.reshape(-1, 1, 2))
        ar = bw / (bh + 1e-6)
        if 0.1 < ar < 15 and 5 < bh < gray.shape[0] * 0.5:
            boxes.append((bw, bh))
    return boxes


def score_typography(img: np.ndarray) -> CategoryResult:
    """Reward a clear typographic hierarchy: a small number of distinct text
    sizes rather than one flat size or dozens of competing sizes."""
    suggestions: List[str] = []
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    boxes = _text_regions(gray)
    heights = np.array([b[1] for b in boxes], dtype=np.float64)

    if len(heights) < 4:
        summary = "Little detectable text; typography not strongly assessed."
        suggestions.append(
            "If this design contains text, ensure headings are clearly larger "
            "than body text to establish hierarchy."
        )
        return CategoryResult("Typography", 70, summary, suggestions)

    # Cluster heights into size tiers.
    tiers = np.unique(np.round(heights / (heights.max() / 6 + 1e-6)))
    n_tiers = len(tiers)
    if 2 <= n_tiers <= 4:
        score = 88
        summary = f"Clear typographic hierarchy ({n_tiers} size tiers detected)."
        suggestions.append("Type hierarchy looks well structured.")
    elif n_tiers == 1:
        score = 62
        summary = "Text appears to use a single size."
        suggestions.append(
            "Introduce a larger heading and smaller supporting text to guide "
            "the reader's eye."
        )
    else:
        score = 60
        summary = f"Many competing text sizes ({n_tiers} tiers)."
        suggestions.append(
            "Limit the design to 2-3 font sizes so the hierarchy stays clear."
        )
    return CategoryResult("Typography", score, summary, suggestions)


# --------------------------------------------------------------------------- #
# Principle 6 - Overall readability (composite)
# --------------------------------------------------------------------------- #
def score_readability(contrast: CategoryResult, white_space: CategoryResult,
                      typography: CategoryResult) -> CategoryResult:
    """Readability is a weighted blend of contrast, breathing room and type
    hierarchy - the factors that most affect whether a viewer can take in the
    message quickly."""
    score = int(round(0.5 * contrast.score
                      + 0.25 * white_space.score
                      + 0.25 * typography.score))
    score = max(0, min(100, score))
    suggestions: List[str] = []
    if score >= 80:
        summary = "The design should read clearly at a glance."
        suggestions.append("Readability is strong overall.")
    elif score >= 60:
        summary = "Readability is acceptable but could be sharper."
        suggestions.append(
            "Improving contrast or spacing (see those sections) will make the "
            "message easier to absorb."
        )
    else:
        summary = "The design may be hard to read quickly."
        suggestions.append(
            "Focus first on raising contrast and giving the main message more "
            "space - these have the biggest impact on readability."
        )
    return CategoryResult("Readability", score, summary, suggestions)


# --------------------------------------------------------------------------- #
# Principle 7 - Layout (compositional structure)
# --------------------------------------------------------------------------- #
def score_layout(img: np.ndarray) -> CategoryResult:
    """Reward a structured composition: content spread across the 3x3 grid
    (rule-of-thirds balance) with focal energy near the thirds intersections
    rather than dumped in one corner or dead centre."""
    suggestions: List[str] = []
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 80, 180).astype(np.float64)
    h, w = edges.shape
    if edges.sum() <= 0:
        return CategoryResult(
            "Layout", 70,
            "Almost no detectable structure; layout not strongly assessed.",
            ["Add a clear focal element so the composition has structure."],
        )

    # 3x3 grid balance.
    cells = []
    for r in range(3):
        for c_ in range(3):
            cells.append(edges[h * r // 3: h * (r + 1) // 3,
                               w * c_ // 3: w * (c_ + 1) // 3].mean())
    cells = np.array(cells) + 1e-6
    balance = 1.0 - min(1.0, (cells.std() / cells.mean()) / 2.0)

    # Focal energy near the four thirds intersections.
    yy, xx = np.nonzero(edges)
    total = len(xx)
    pts = np.stack([xx / w, yy / h], axis=1)
    thirds = np.array([[1 / 3, 1 / 3], [2 / 3, 1 / 3],
                       [1 / 3, 2 / 3], [2 / 3, 2 / 3]])
    dists = np.linalg.norm(pts[:, None, :] - thirds[None, :, :], axis=2).min(axis=1)
    near_thirds = float((dists < 0.18).sum() / total)

    score = int(round(100 * (0.55 * balance + 0.45 * min(1.0, near_thirds * 2.2))))
    score = max(0, min(100, score))

    summary = (f"Grid balance {balance*100:.0f}%; "
               f"{near_thirds*100:.0f}% of detail near thirds intersections.")
    if balance < 0.5:
        suggestions.append(
            "Content is bunched in one area. Spread elements across the "
            "canvas so no third of the design sits empty while another overflows."
        )
    if near_thirds < 0.15:
        suggestions.append(
            "Try placing the main subject near a rule-of-thirds intersection "
            "instead of dead centre for a more dynamic composition."
        )
    if not suggestions:
        suggestions.append("Composition is well structured across the grid.")
    return CategoryResult("Layout", score, summary, suggestions)


# --------------------------------------------------------------------------- #
# Principle 8 - Visual hierarchy (focal dominance)
# --------------------------------------------------------------------------- #
def score_visual_hierarchy(img: np.ndarray) -> CategoryResult:
    """Reward one clear focal point: the largest visual mass should clearly
    dominate the secondary elements and sit where the eye lands first
    (upper part of the design for posters and flyers)."""
    suggestions: List[str] = []
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 80, 180)
    dilated = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    areas = sorted([cv2.contourArea(c) for c in contours if cv2.contourArea(c) > 50],
                   reverse=True)
    if len(areas) < 3:
        return CategoryResult(
            "Visual Hierarchy", 70,
            "Too few distinct elements to judge hierarchy.",
            ["Give the design one dominant headline or image so the eye "
             "knows where to start."],
        )

    top = np.array(areas[:8], dtype=np.float64)
    dominance = float(top[0] / (np.median(top[1:]) + 1e-6))
    if dominance >= 4.0:
        dom_score, dom_note = 88, "one element clearly dominates"
    elif dominance >= 2.0:
        dom_score, dom_note = 72, "focal element is moderately dominant"
    else:
        dom_score, dom_note = 55, "no element clearly dominates"

    biggest = max((c for c in contours if cv2.contourArea(c) > 50),
                  key=cv2.contourArea)
    _, cy, _, ch = cv2.boundingRect(biggest)
    centre_y = (cy + ch / 2) / img.shape[0]
    position_bonus = 6 if centre_y < 0.6 else 0

    score = max(0, min(100, int(round(dom_score + position_bonus))))
    summary = (f"Focal dominance ratio {dominance:.1f}:1 ({dom_note}); "
               f"main element sits at {centre_y*100:.0f}% height.")
    if dominance < 2.0:
        suggestions.append(
            "Everything competes at the same size. Enlarge the single most "
            "important message or image so it is 2-4x bigger than the rest."
        )
    if centre_y >= 0.6:
        suggestions.append(
            "The biggest element sits low on the canvas. Move the headline "
            "or hero image higher so viewers meet it first."
        )
    if not suggestions:
        suggestions.append("Clear focal point guides the eye well.")
    return CategoryResult("Visual Hierarchy", score, summary, suggestions)


# --------------------------------------------------------------------------- #
# Grade mapping lives in app/services/scoring.py (Module 5).
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def analyze_image(path: str) -> AnalysisResult:
    """Run every principle on the image at `path` and return an AnalysisResult."""
    img = load_image(path)
    palette = dominant_colors(img, k=5)

    contrast = score_contrast(img, palette)
    harmony = score_color_harmony(img, palette)
    white_space = score_white_space(img)
    alignment = score_alignment(img)
    typography = score_typography(img)
    readability = score_readability(contrast, white_space, typography)
    layout = score_layout(img)
    hierarchy = score_visual_hierarchy(img)

    categories = [contrast, harmony, typography, alignment, white_space,
                  readability, layout, hierarchy]
    overall = compute_overall([c.score for c in categories])
    palette_hex = [_rgb_to_hex(rgb) for rgb, frac in palette if frac >= 0.03]

    return AnalysisResult(
        overall=overall,
        grade=grade_for(overall),
        categories=categories,
        palette=palette_hex,
    )
