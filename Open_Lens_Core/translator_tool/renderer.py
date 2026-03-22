"""
Translated text rendering.

For each block:
  1. Sample the original text colour via k-means clustering.
  2. Detect font weight via stroke-width analysis.
  3. Detect text alignment (left / centre / right).
  4. Calibrate font size using rendered glyph height comparison.
  5. Measure original line spacing and reproduce it.
  6. Auto-fit translated text with word wrapping.
  7. Handle RTL languages via python-bidi + arabic-reshaper (optional).
"""

from __future__ import annotations

import os
import sys
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .language_utils import is_rtl
from .font_analyzer import (
    extract_text_color,
    detect_font_weight,
    detect_alignment,
    measure_line_spacing,
    calibrate_font_size,
)


# ---------------------------------------------------------------------------
# Font discovery
# ---------------------------------------------------------------------------

# Ordered list of fonts with good Unicode / multilingual coverage
_WINDOWS_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\arialuni.ttf",   # Arial Unicode MS (best coverage)
    r"C:\Windows\Fonts\arial.ttf",
    r"C:\Windows\Fonts\calibri.ttf",
    r"C:\Windows\Fonts\tahoma.ttf",
    r"C:\Windows\Fonts\segoeui.ttf",
    r"C:\Windows\Fonts\verdana.ttf",
    r"C:\Windows\Fonts\times.ttf",
]

_WINDOWS_BOLD_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\arialbd.ttf",
    r"C:\Windows\Fonts\calibrib.ttf",
    r"C:\Windows\Fonts\tahomabd.ttf",
    r"C:\Windows\Fonts\segoeuib.ttf",
    r"C:\Windows\Fonts\verdanab.ttf",
    r"C:\Windows\Fonts\timesbd.ttf",
]

_WINDOWS_ITALIC_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\ariali.ttf",
    r"C:\Windows\Fonts\calibrii.ttf",
    r"C:\Windows\Fonts\segoeuii.ttf",
    r"C:\Windows\Fonts\verdanai.ttf",
    r"C:\Windows\Fonts\timesi.ttf",
]

_LINUX_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
]

_MACOS_FONT_CANDIDATES = [
    "/System/Library/Fonts/Helvetica.ttc",
    "/Library/Fonts/Arial Unicode.ttf",
    "/System/Library/Fonts/SFNS.ttf",
]


def find_system_font() -> Optional[str]:
    """Return the path to the best available system font, or None."""
    if sys.platform == "win32":
        candidates = _WINDOWS_FONT_CANDIDATES
    elif sys.platform == "darwin":
        candidates = _MACOS_FONT_CANDIDATES
    else:
        candidates = _LINUX_FONT_CANDIDATES

    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def _find_bold_font() -> Optional[str]:
    """Return path to a bold system font, or None."""
    if sys.platform == "win32":
        candidates = _WINDOWS_BOLD_FONT_CANDIDATES
    else:
        return None  # TODO: add Linux/macOS bold candidates
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def _find_italic_font() -> Optional[str]:
    """Return path to an italic system font, or None."""
    if sys.platform == "win32":
        candidates = _WINDOWS_ITALIC_FONT_CANDIDATES
    else:
        return None
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def _load_font(font_path: Optional[str], size: int) -> ImageFont.FreeTypeFont:
    if font_path:
        try:
            return ImageFont.truetype(font_path, size)
        except (OSError, IOError):
            pass
    # Fallback: Pillow's built-in bitmap font (fixed size, limited charset)
    return ImageFont.load_default()


# ---------------------------------------------------------------------------
# Text metrics
# ---------------------------------------------------------------------------

def _text_bbox(text: str, font: ImageFont.FreeTypeFont) -> Tuple[int, int]:
    """Return (width, height) of *text* rendered in *font*."""
    try:
        bbox = font.getbbox(text)   # (left, top, right, bottom)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    except AttributeError:
        # Older Pillow
        w, h = font.getsize(text)
        return w, h


def _wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> List[str]:
    """Word-wrap *text* so each line fits within *max_width* pixels."""
    words = text.split()
    if not words:
        return []

    lines: List[str] = []
    current: List[str] = []

    for word in words:
        candidate = " ".join(current + [word])
        w, _ = _text_bbox(candidate, font)
        if w <= max_width:
            current.append(word)
        else:
            if current:
                lines.append(" ".join(current))
            current = [word]

    if current:
        lines.append(" ".join(current))

    return lines or [text]


def _fit_font(
    text: str,
    max_width: int,
    max_height: int,
    font_path: Optional[str],
    min_size: int = 6,
    max_size: Optional[int] = None,
) -> Tuple[ImageFont.FreeTypeFont, List[str], int]:
    """Find the largest font size for which *text* fits in the bounding box.

    Returns (font, wrapped_lines, font_size).
    """
    if max_size is None:
        # Start from the height of the box as an upper bound
        max_size = max(max_height, 12)

    # Use a step-based search for speed: try larger steps first, then refine
    best = None
    for size in range(max_size, min_size - 1, -1):
        font = _load_font(font_path, size)
        lines = _wrap_text(text, font, max_width)
        _, line_h = _text_bbox("Ag", font)
        line_spacing = line_h + 2
        total_h = line_spacing * len(lines)
        max_line_w = max((_text_bbox(ln, font)[0] for ln in lines), default=0)

        if max_line_w <= max_width and total_h <= max_height:
            return font, lines, size

    # Minimum size fallback
    font = _load_font(font_path, min_size)
    lines = _wrap_text(text, font, max_width)
    return font, lines, min_size


def _estimate_original_font_size(block: Dict[str, Any], font_path: Optional[str] = None) -> Optional[int]:
    """Estimate original font size using glyph-height calibration.

    Delegates to :func:`font_analyzer.calibrate_font_size` which binary-searches
    for the font size whose rendered height matches the OCR bounding box height.
    """
    return calibrate_font_size(block, font_path)


# ---------------------------------------------------------------------------
# Colour sampling  (delegates to font_analyzer)
# ---------------------------------------------------------------------------

def _sample_text_color(
    original_image: Image.Image,
    block: Dict[str, Any],
) -> Tuple[int, int, int]:
    """Extract ink colour via k-means clustering."""
    return extract_text_color(original_image, block)


def _detect_font_weight_legacy(
    original_image: Image.Image,
    block: Dict[str, Any],
) -> str:
    """Detect bold/italic via stroke-width analysis."""
    return detect_font_weight(original_image, block)


# ---------------------------------------------------------------------------
# RTL support (optional)
# ---------------------------------------------------------------------------

def _prepare_rtl_text(text: str, lang_code: str) -> str:
    """Reshape and reorder Arabic/Hebrew text for correct rendering."""
    if not is_rtl(lang_code):
        return text
    try:
        import arabic_reshaper  # type: ignore
        from bidi.algorithm import get_display  # type: ignore
        reshaped = arabic_reshaper.reshape(text)
        return get_display(reshaped)
    except ImportError:
        # Libraries not installed – render as-is (may look reversed)
        return text


# ---------------------------------------------------------------------------
# Main render function
# ---------------------------------------------------------------------------

def render_translated_blocks(
    image: Image.Image,
    blocks: List[Dict[str, Any]],
    original_image: Optional[Image.Image] = None,
    font_path: Optional[str] = None,
    target_lang: str = "en",
    default_text_color: Tuple[int, int, int] = (0, 0, 0),
) -> Image.Image:
    """Draw translated text onto *image* inside each block's bounding box.

    Args:
        image:              The inpainted (background-only) PIL Image to draw on.
        blocks:             Block dicts that have ``translated_text`` populated.
        original_image:     Original image used to sample ink colour (optional).
        font_path:          Path to a .ttf font file. Auto-detected if None.
        target_lang:        ISO language code of the translated text (for RTL).
        default_text_color: Fallback text colour as RGB tuple.

    Returns:
        A new PIL Image with translated text rendered in place.
    """
    if font_path is None:
        font_path = find_system_font()

    bold_font_path = _find_bold_font()
    italic_font_path = _find_italic_font()

    result = image.copy()
    draw = ImageDraw.Draw(result)
    img_width = image.size[0]

    rtl = is_rtl(target_lang)

    for block in blocks:
        translated = block.get("translated_text", block.get("text", ""))
        if not translated or not translated.strip():
            continue

        x, y = block["x"], block["y"]
        w, h = block["w"], block["h"]

        # --- Colour ---
        if original_image is not None:
            text_color = _sample_text_color(original_image, block)
        else:
            text_color = default_text_color

        # --- Font weight ---
        weight = "regular"
        if original_image is not None:
            weight = _detect_font_weight_legacy(original_image, block)

        if weight == "bold" and bold_font_path:
            active_font_path = bold_font_path
        elif weight == "italic" and italic_font_path:
            active_font_path = italic_font_path
        else:
            active_font_path = font_path

        # --- RTL ---
        display_text = _prepare_rtl_text(translated.strip(), target_lang)

        # --- Font size (calibrated) ---
        estimated_size = _estimate_original_font_size(block, active_font_path)
        if estimated_size is None:
            original_text = block.get("text", "")
            if original_text and original_text.strip():
                _, _, estimated_size = _fit_font(original_text.strip(), w, h, active_font_path)

        # --- Alignment ---
        alignment = detect_alignment(block, img_width)
        if rtl:
            alignment = "right"

        # --- Line spacing ---
        original_spacing = measure_line_spacing(block)

        # Allow up to 60% extra height for longer translations
        fit_h = int(h * 1.6)
        min_font = max(6, int(estimated_size * 0.70)) if estimated_size else 6
        font, lines, font_size = _fit_font(
            display_text, w, fit_h, active_font_path,
            min_size=min_font, max_size=estimated_size,
        )

        # Line height: use measured original spacing if available
        _, glyph_h = _text_bbox("Ag", font)
        if original_spacing and original_spacing > glyph_h:
            line_spacing = int(original_spacing)
        else:
            line_spacing = glyph_h + 2

        # --- Vertical positioning ---
        total_text_h = line_spacing * len(lines)
        # Top-align by default; centre if block is significantly taller
        if total_text_h < h * 0.7 and len(lines) == 1:
            # Centre single short lines vertically
            current_y = y + (h - total_text_h) // 2
        else:
            current_y = y

        overflow_limit = y + h + int(h * 0.6)

        for line in lines:
            if current_y + glyph_h > overflow_limit:
                break

            line_w, _ = _text_bbox(line, font)
            if alignment == "right":
                draw_x = x + w - line_w
            elif alignment == "center":
                draw_x = x + (w - line_w) // 2
            else:
                draw_x = x

            draw.text((draw_x, current_y), line, font=font, fill=text_color)
            current_y += line_spacing

    return result
