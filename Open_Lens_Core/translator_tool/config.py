"""
Centralised configuration for the translation pipeline.

Auto-detects which backends (layout, OCR, inpainting, translation) are
available on the current machine and picks the best one.  Every setting
can be overridden by the caller.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Backend availability probes
# ---------------------------------------------------------------------------

def _has_layoutparser() -> bool:
    try:
        import layoutparser  # noqa: F401
        return True
    except Exception:
        return False


def _has_easyocr() -> bool:
    try:
        import easyocr  # noqa: F401
        return True
    except Exception:
        return False


def _has_paddleocr() -> bool:
    try:
        from paddleocr import PaddleOCR  # noqa: F401
        return True
    except Exception:
        return False


def _has_tesseract() -> bool:
    try:
        import pytesseract  # noqa: F401
        return True
    except Exception:
        return False


def _has_lama() -> bool:
    try:
        from simple_lama_inpainting import SimpleLama  # noqa: F401
        return True
    except Exception:
        return False


def _has_torch() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class TranslationConfig:
    """All pipeline settings in one place.

    Set any field to ``"auto"`` (or ``None``) to let auto-detection decide.
    """

    # --- backend selection (auto | specific name) --------------------------
    layout_engine: str = "auto"       # "layoutparser" | "paddleocr" | "none"
    ocr_engine: str = "auto"          # "easyocr" | "paddleocr" | "tesseract"
    inpaint_engine: str = "auto"      # "lama" | "telea"
    translator_engine: str = "auto"   # "argos" | "ollama"

    # --- device -------------------------------------------------------------
    device: str = "auto"              # "auto" | "cuda" | "cpu"

    # --- language -----------------------------------------------------------
    target_lang: str = "en"
    source_lang: str = "auto"

    # --- paths / overrides --------------------------------------------------
    tesseract_cmd: Optional[str] = None
    font_path: Optional[str] = None
    ollama_url: Optional[str] = "http://localhost:11434"
    ollama_model: Optional[str] = "qwen2.5:7b"

    # --- layout parser options ----------------------------------------------
    layout_model: str = "auto"        # auto-select or specific model label
    layout_confidence: float = 0.5    # min detection confidence

    # --- glossary -----------------------------------------------------------
    glossary_path: Optional[str] = None  # TSV file: source_term<TAB>target_term

    # --- resolved (filled by resolve()) ------------------------------------
    _resolved: bool = field(default=False, repr=False)

    # -----------------------------------------------------------------------
    def resolve(self) -> "TranslationConfig":
        """Probe the system and fill in ``"auto"`` values with concrete choices."""
        if self._resolved:
            return self

        # Device
        if self.device == "auto":
            self.device = "cuda" if _has_cuda() else "cpu"

        # Layout engine
        if self.layout_engine == "auto":
            if _has_layoutparser():
                self.layout_engine = "layoutparser"
            elif _has_paddleocr():
                self.layout_engine = "paddleocr"
            else:
                self.layout_engine = "none"

        # OCR engine
        if self.ocr_engine == "auto":
            if _has_easyocr():
                self.ocr_engine = "easyocr"
            elif _has_paddleocr():
                self.ocr_engine = "paddleocr"
            elif _has_tesseract():
                self.ocr_engine = "tesseract"
            else:
                self.ocr_engine = "tesseract"  # will fail later with a clear msg

        # Inpainter
        if self.inpaint_engine == "auto":
            if _has_lama():
                self.inpaint_engine = "lama"
            else:
                self.inpaint_engine = "telea"

        # Translator
        if self.translator_engine == "auto":
            self.translator_engine = "argos"   # always available after pip install

        self._resolved = True
        log.info(
            "Resolved config: layout=%s  ocr=%s  inpaint=%s  translate=%s  device=%s",
            self.layout_engine, self.ocr_engine, self.inpaint_engine,
            self.translator_engine, self.device,
        )
        return self

    def summary(self) -> str:
        """Human-readable one-liner."""
        return (
            f"layout={self.layout_engine}  ocr={self.ocr_engine}  "
            f"inpaint={self.inpaint_engine}  translate={self.translator_engine}  "
            f"device={self.device}"
        )
