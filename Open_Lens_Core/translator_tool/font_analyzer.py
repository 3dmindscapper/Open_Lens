"""
Font analysis module.

Provides analysis functions for:
  - **Colour extraction** via k-means clustering (replaces percentile heuristic)
  - **Stroke width** measurement via distance transform (replaces ink-ratio heuristic)
  - **Alignment detection** from word positions (left / centre / right)
  - **Line spacing** measurement from original multi-line blocks
  - **Font size calibration** using rendered glyph height comparison

All functions operate on numpy arrays cropped from the original image, so
they are independent of the OCR engine used.
"""

from __future__ import annotations

import logging
from typing import Dict, Any, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageFont

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# K-means colour extraction
# ---------------------------------------------------------------------------

def extract_text_color(
    original_image: Image.Image,
    block: Dict[str, Any],
    n_clusters: int = 3,
) -> Tuple[int, int, int]:
    """Extract the dominant text (ink) colour from a block region.

    Uses k-means clustering to separate background from ink pixels,
    then selects the darkest non-background cluster.

    Falls back to (0, 0, 0) if the region is too small or featureless.
    """
    x, y, x2, y2 = block["x"], block["y"], block["x2"], block["y2"]
    region = np.array(original_image.crop((x, y, x2, y2)), dtype=np.float32)

    if region.size == 0 or region.shape[0] < 2 or region.shape[1] < 2:
        return (0, 0, 0)

    pixels = region.reshape(-1, 3)
    if len(pixels) < n_clusters:
        return (0, 0, 0)

    # k-means with OpenCV (fast, no sklearn dependency)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(
        pixels, n_clusters, None, criteria, 3, cv2.KMEANS_PP_CENTERS,
    )

    # Find the background cluster (brightest, most populous)
    # and the text cluster (darkest that is NOT background)
    centers = centers.astype(np.float64)
    counts = np.bincount(labels.flatten(), minlength=n_clusters)
    brightness = 0.299 * centers[:, 0] + 0.587 * centers[:, 1] + 0.114 * centers[:, 2]

    # Background = cluster with highest brightness among those with significant count
    min_count = len(pixels) * 0.05  # at least 5% of pixels
    bg_idx = int(np.argmax(brightness))

    # Text = darkest cluster that has meaningful pixel count
    text_idx = None
    for idx in np.argsort(brightness):
        if idx != bg_idx and counts[idx] >= min_count:
            text_idx = idx
            break

    if text_idx is None:
        # Only one meaningful cluster: use the darkest overall
        text_idx = int(np.argmin(brightness))

    colour = tuple(int(c) for c in centers[text_idx][:3])
    return colour  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Stroke width analysis (font weight detection)
# ---------------------------------------------------------------------------

def detect_font_weight(
    original_image: Image.Image,
    block: Dict[str, Any],
) -> str:
    """Detect whether text in *block* is bold, italic, or regular.

    Uses distance-transform-based stroke width measurement, which is
    more robust than the ink-ratio heuristic.

    Returns:
        ``"bold"``, ``"italic"``, or ``"regular"``.
    """
    x, y, x2, y2 = block["x"], block["y"], block["x2"], block["y2"]
    region = np.array(original_image.crop((x, y, x2, y2)))
    if region.size == 0 or region.shape[0] < 4 or region.shape[1] < 4:
        return "regular"

    gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)

    # Binarise: Otsu's method adapts to any background
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    ink_pixels = np.count_nonzero(binary)
    if ink_pixels < 10:
        return "regular"

    # Distance transform — each ink pixel gets its distance to the nearest
    # background pixel; the maximum along the skeleton ≈ half stroke width.
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    ink_distances = dist[binary > 0]
    if len(ink_distances) == 0:
        return "regular"

    mean_stroke = float(np.mean(ink_distances))
    median_stroke = float(np.median(ink_distances))

    # Normalise by block height to make the thresholds resolution-independent
    block_h = max(y2 - y, 1)
    stroke_ratio = mean_stroke / block_h

    # Bold: relatively thick strokes
    if stroke_ratio > 0.06:
        return "bold"

    # Italic detection: check slant via vertical projection profile
    if _detect_slant(binary):
        return "italic"

    # ALL-CAPS with moderate stroke → bold header
    text = block.get("text", "")
    if text == text.upper() and len(text) > 3 and stroke_ratio > 0.04:
        return "bold"

    return "regular"


def _detect_slant(binary: np.ndarray, threshold_deg: float = 8.0) -> bool:
    """Detect italic slant from a binarised text image.

    Uses the Hough transform on edges to find dominant near-vertical
    line angles.  If the dominant angle deviates from vertical by more
    than *threshold_deg*, the text is likely italic.
    """
    if binary.shape[0] < 10 or binary.shape[1] < 10:
        return False

    edges = cv2.Canny(binary, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=20,
                            minLineLength=binary.shape[0] // 4, maxLineGap=3)
    if lines is None or len(lines) == 0:
        return False

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx = x2 - x1
        dy = y2 - y1
        if abs(dy) < abs(dx):
            continue  # skip near-horizontal lines
        angle = np.degrees(np.arctan2(dx, dy))
        angles.append(angle)

    if not angles:
        return False

    median_angle = float(np.median(angles))
    return abs(median_angle) > threshold_deg


# ---------------------------------------------------------------------------
# Text alignment detection
# ---------------------------------------------------------------------------

def detect_alignment(block: Dict[str, Any], img_width: int) -> str:
    """Detect horizontal text alignment within a block.

    Analyses word positions across lines to determine left, centre, or right
    alignment.

    Returns:
        ``"left"``, ``"center"``, or ``"right"``.
    """
    word_boxes = block.get("word_boxes", [])
    if not word_boxes:
        return "left"

    block_x, block_x2 = block["x"], block["x2"]
    block_w = block_x2 - block_x
    if block_w <= 0:
        return "left"

    # Group words into lines
    lines = _group_words_into_lines(word_boxes)
    if len(lines) < 2:
        return "left"  # single line — can't determine alignment

    # For each line, compute left margin and right margin within the block bbox
    left_margins = []
    right_margins = []
    for line in lines:
        line_x1 = min(w["x"] for w in line)
        line_x2 = max(w["x2"] for w in line)
        left_margins.append(line_x1 - block_x)
        right_margins.append(block_x2 - line_x2)

    left_var = float(np.std(left_margins)) if len(left_margins) > 1 else 0.0
    right_var = float(np.std(right_margins)) if len(right_margins) > 1 else 0.0
    center_offsets = [(lm + rm) / 2 for lm, rm in zip(left_margins, right_margins)]
    center_var = float(np.std(center_offsets)) if len(center_offsets) > 1 else 0.0

    tolerance = block_w * 0.05  # 5% of block width

    if left_var <= tolerance and right_var > tolerance:
        return "left"
    if right_var <= tolerance and left_var > tolerance:
        return "right"
    if center_var <= tolerance:
        return "center"

    return "left"


def _group_words_into_lines(word_boxes: List[Dict]) -> List[List[Dict]]:
    """Group word boxes into lines by y-coordinate proximity."""
    if not word_boxes:
        return []

    heights = [wb["h"] for wb in word_boxes if wb.get("h", 0) > 0]
    if not heights:
        return [word_boxes]
    median_h = sorted(heights)[len(heights) // 2]
    y_tol = max(median_h * 0.5, 5)

    sorted_words = sorted(word_boxes, key=lambda w: (w["y"], w["x"]))
    lines: List[List[Dict]] = []
    for wb in sorted_words:
        placed = False
        for line in lines:
            if abs(wb["y"] - line[0]["y"]) <= y_tol:
                line.append(wb)
                placed = True
                break
        if not placed:
            lines.append([wb])

    return lines


# ---------------------------------------------------------------------------
# Line spacing measurement
# ---------------------------------------------------------------------------

def measure_line_spacing(block: Dict[str, Any]) -> Optional[float]:
    """Measure the average line spacing in a multi-line block.

    Returns the measured pixel distance between line baselines, or ``None``
    if the block has fewer than 2 lines.
    """
    word_boxes = block.get("word_boxes", [])
    if not word_boxes:
        return None

    lines = _group_words_into_lines(word_boxes)
    if len(lines) < 2:
        return None

    # Use the top of each line as a proxy for baseline
    line_tops = sorted(min(w["y"] for w in line) for line in lines)
    gaps = [line_tops[i + 1] - line_tops[i] for i in range(len(line_tops) - 1)]
    if not gaps:
        return None

    return float(np.median(gaps))


# ---------------------------------------------------------------------------
# Font size calibration
# ---------------------------------------------------------------------------

def calibrate_font_size(
    block: Dict[str, Any],
    font_path: Optional[str] = None,
) -> Optional[int]:
    """Estimate the font size that best matches the original text height.

    Uses word-level bounding box heights and compensates for the vertical
    padding that OCR engines add around glyph bounding boxes by actually
    rendering a reference string and comparing heights.
    """
    word_boxes = block.get("word_boxes", [])
    if not word_boxes:
        return None

    heights = sorted(wb["h"] for wb in word_boxes if wb.get("h", 0) > 0)
    if not heights:
        return None

    # Use median height as reference (robust to outliers)
    median_h = heights[len(heights) // 2]

    if not font_path:
        # Without a specific font, use a scaling factor based on typical
        # TrueType metrics: glyph em-height ≈ 72% of bbox height
        return max(6, int(median_h * 0.75))

    # Binary search: find the font size whose rendered height matches median_h
    lo, hi = 6, max(median_h * 2, 20)
    best_size = max(6, int(median_h * 0.75))
    ref_text = "Hgpqy"  # includes ascenders and descenders

    for _ in range(15):  # max iterations
        mid = (lo + hi) // 2
        if mid < 6:
            break
        try:
            font = ImageFont.truetype(font_path, mid)
            bbox = font.getbbox(ref_text)
            rendered_h = bbox[3] - bbox[1]
        except Exception:
            break

        if rendered_h < median_h:
            best_size = mid
            lo = mid + 1
        elif rendered_h > median_h:
            hi = mid - 1
        else:
            best_size = mid
            break

    return max(6, best_size)
