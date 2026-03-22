"""
Document layout detection module.

Provides semantic region detection (Text, Title, Table, Figure, List) using
either Layout Parser (with Detectron2 backend) or PaddleOCR's PP-Structure.
Falls back gracefully when neither is available.

All models are downloaded once on first use and cached locally.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class LayoutRegion:
    """A detected region on a page."""
    x: int
    y: int
    w: int
    h: int
    x2: int
    y2: int
    region_type: str          # "Text", "Title", "List", "Table", "Figure"
    confidence: float = 0.0

    @property
    def area(self) -> int:
        return self.w * self.h


# ---------------------------------------------------------------------------
# Layout Parser backend  (Detectron2)
# ---------------------------------------------------------------------------

_LP_MODEL = None  # singleton cache


def _get_layoutparser_model(model_label: str = "auto", device: str = "cpu"):
    """Return a cached layoutparser model instance."""
    global _LP_MODEL
    if _LP_MODEL is not None:
        return _LP_MODEL

    import layoutparser as lp  # type: ignore

    if model_label == "auto":
        # PubLayNet Faster R-CNN — good balance of speed and accuracy
        model_label = "lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config"

    extra_config = []
    if device == "cpu":
        extra_config = ["MODEL.DEVICE", "cpu"]

    _LP_MODEL = lp.Detectron2LayoutModel(
        model_label,
        extra_config=extra_config,
        label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
    )
    log.info("Layout Parser model loaded: %s (device=%s)", model_label, device)
    return _LP_MODEL


def detect_layout_layoutparser(
    image: Image.Image,
    model_label: str = "auto",
    device: str = "cpu",
    min_confidence: float = 0.5,
) -> List[LayoutRegion]:
    """Detect layout regions using Layout Parser + Detectron2."""
    import layoutparser as lp  # type: ignore

    model = _get_layoutparser_model(model_label, device)
    img_np = np.array(image)
    layout = model.detect(img_np)

    regions: List[LayoutRegion] = []
    for block in layout:
        if block.score < min_confidence:
            continue
        x1, y1, x2, y2 = (
            int(block.block.x_1), int(block.block.y_1),
            int(block.block.x_2), int(block.block.y_2),
        )
        regions.append(LayoutRegion(
            x=x1, y=y1, w=x2 - x1, h=y2 - y1, x2=x2, y2=y2,
            region_type=block.type,
            confidence=block.score,
        ))

    log.info("Layout Parser detected %d region(s)", len(regions))
    return regions


# ---------------------------------------------------------------------------
# PaddleOCR PP-Structure backend
# ---------------------------------------------------------------------------

_PADDLE_LAYOUT = None


def _get_paddle_layout_model(device: str = "cpu"):
    """Return a cached PaddleOCR layout model."""
    global _PADDLE_LAYOUT
    if _PADDLE_LAYOUT is not None:
        return _PADDLE_LAYOUT

    from paddleocr import PPStructure  # type: ignore

    use_gpu = device != "cpu"
    _PADDLE_LAYOUT = PPStructure(
        table=False,
        ocr=False,
        layout=True,
        show_log=False,
        use_gpu=use_gpu,
    )
    log.info("PaddleOCR layout model loaded (device=%s)", device)
    return _PADDLE_LAYOUT


# PaddleOCR label → normalised region type
_PADDLE_TYPE_MAP = {
    "text": "Text",
    "title": "Title",
    "figure": "Figure",
    "table": "Table",
    "list": "List",
    "header": "Title",
    "footer": "Text",
    "reference": "Text",
    "equation": "Figure",
}


def detect_layout_paddleocr(
    image: Image.Image,
    device: str = "cpu",
    min_confidence: float = 0.5,
) -> List[LayoutRegion]:
    """Detect layout regions using PaddleOCR PP-Structure."""
    model = _get_paddle_layout_model(device)
    img_np = np.array(image)
    result = model(img_np)

    regions: List[LayoutRegion] = []
    for item in result:
        region_info = item.get("type", "text").lower()
        score = item.get("score", 0.0) if isinstance(item.get("score"), (int, float)) else 0.0
        if score < min_confidence:
            continue

        bbox = item.get("bbox", item.get("region", []))
        if not bbox or len(bbox) < 4:
            continue

        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        rtype = _PADDLE_TYPE_MAP.get(region_info, "Text")

        regions.append(LayoutRegion(
            x=x1, y=y1, w=x2 - x1, h=y2 - y1, x2=x2, y2=y2,
            region_type=rtype,
            confidence=score,
        ))

    log.info("PaddleOCR layout detected %d region(s)", len(regions))
    return regions


# ---------------------------------------------------------------------------
# Unified interface
# ---------------------------------------------------------------------------

def detect_layout(
    image: Image.Image,
    engine: str = "auto",
    device: str = "cpu",
    model_label: str = "auto",
    min_confidence: float = 0.5,
) -> List[LayoutRegion]:
    """Detect layout regions using the best available backend.

    Args:
        image:          RGB PIL Image.
        engine:         ``"layoutparser"``, ``"paddleocr"``, ``"none"``, or ``"auto"``.
        device:         ``"cpu"`` or ``"cuda"``.
        model_label:    Layout Parser model config (or ``"auto"``).
        min_confidence: Minimum detection confidence.

    Returns:
        List of :class:`LayoutRegion`.  Empty list when engine is ``"none"``.
    """
    if engine == "none":
        return []

    if engine in ("layoutparser", "auto"):
        try:
            return detect_layout_layoutparser(image, model_label, device, min_confidence)
        except Exception as exc:
            if engine == "layoutparser":
                raise
            log.warning("Layout Parser unavailable (%s), trying PaddleOCR…", exc)

    if engine in ("paddleocr", "auto"):
        try:
            return detect_layout_paddleocr(image, device, min_confidence)
        except Exception as exc:
            if engine == "paddleocr":
                raise
            log.warning("PaddleOCR layout unavailable (%s), skipping layout detection.", exc)

    return []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def filter_text_regions(regions: List[LayoutRegion]) -> List[LayoutRegion]:
    """Keep only regions that should be OCR'd (Text, Title, List — not Figure)."""
    return [r for r in regions if r.region_type in ("Text", "Title", "List", "Table")]


def crop_region(image: Image.Image, region: LayoutRegion) -> Image.Image:
    """Crop the image to the given region (clamped to image bounds)."""
    w, h = image.size
    x1 = max(0, region.x)
    y1 = max(0, region.y)
    x2 = min(w, region.x2)
    y2 = min(h, region.y2)
    return image.crop((x1, y1, x2, y2))
