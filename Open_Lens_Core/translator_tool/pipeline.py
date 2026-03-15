"""
Main translation pipeline.

Orchestrates: load → OCR → language detection → translate → inpaint → render → save.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Callable, Any

from PIL import Image

from .file_handler import load_document, save_output, guess_output_path
from .ocr_extractor import extract_text_blocks, extract_full_text, configure_tesseract_path
from .language_utils import (
    detect_language,
    langdetect_to_tesseract,
    translate_blocks,
)
from .inpainter import remove_text_blocks
from .renderer import render_translated_blocks, find_system_font


# ---------------------------------------------------------------------------
# Per-page processing
# ---------------------------------------------------------------------------

def process_page(
    page_image: Image.Image,
    target_lang: str,
    source_lang: str = "auto",
    tesseract_cmd: Optional[str] = None,
    font_path: Optional[str] = None,
    log: Callable[[str], Any] = print,
) -> Image.Image:
    """Run the full translation pipeline on a single page image.

    Args:
        page_image:    RGB PIL Image of one document page.
        target_lang:   Target language code (e.g. ``"en"``, ``"fr"``).
        source_lang:   Source language code or ``"auto"`` to detect.
        tesseract_cmd: Path to the Tesseract binary (optional override).
        font_path:     Path to a .ttf font file (optional override).
        log:           Callable used for progress messages.

    Returns:
        Processed PIL Image with translated text in place.
    """
    if tesseract_cmd:
        configure_tesseract_path(tesseract_cmd)

    # ---- Step 1: First-pass OCR (English) to gather text for language detection
    if source_lang == "auto":
        log("    Detecting language…")
        sample_text = extract_full_text(page_image, ocr_lang="eng")
        detected_lang = detect_language(sample_text)
        ocr_lang = langdetect_to_tesseract(detected_lang)
        src_for_translation = detected_lang
        log(f"    Detected language: {detected_lang}  →  Tesseract pack: {ocr_lang}")
    else:
        ocr_lang = langdetect_to_tesseract(source_lang)
        src_for_translation = source_lang
        log(f"    Using source language: {source_lang}  →  Tesseract pack: {ocr_lang}")

    # ---- Step 2: Proper OCR pass with the correct language pack
    log(f"    Running OCR (lang={ocr_lang})…")
    blocks = extract_text_blocks(page_image, ocr_lang=ocr_lang)

    # If the language-specific pack fails or finds nothing, fall back to English
    if not blocks and ocr_lang != "eng":
        log("    Language pack returned no blocks, retrying with 'eng'…")
        blocks = extract_text_blocks(page_image, ocr_lang="eng")

    if not blocks:
        log("    No text found on this page – returning unchanged.")
        return page_image.copy()

    log(f"    Found {len(blocks)} text block(s).")

    # ---- Step 3: Translate
    log(f"    Translating to '{target_lang}'…")
    blocks = translate_blocks(
        blocks,
        target_lang=target_lang,
        source_lang=src_for_translation,
        log=log,
    )

    # ---- Step 4: Inpaint (remove original text)
    log("    Removing original text (inpainting)…")
    inpainted = remove_text_blocks(page_image, blocks)

    # ---- Step 5: Render translated text
    log("    Rendering translated text…")
    final = render_translated_blocks(
        image=inpainted,
        blocks=blocks,
        original_image=page_image,
        font_path=font_path,
        target_lang=target_lang,
    )

    return final


# ---------------------------------------------------------------------------
# Document-level entry point
# ---------------------------------------------------------------------------

def process_document(
    input_path: str,
    target_lang: str,
    output_path: Optional[str] = None,
    source_lang: str = "auto",
    tesseract_cmd: Optional[str] = None,
    font_path: Optional[str] = None,
    verbose: bool = True,
    log_callback: Optional[Callable[[str], Any]] = None,
) -> str:
    """Translate all text in a document file and save the result.

    Supports PDF, JPG, PNG (and other common image formats).

    Args:
        input_path:    Path to the input file.
        target_lang:   ISO 639-1 target language code (e.g. ``"en"``).
        output_path:   Where to save the result. Auto-generated if ``None``.
        source_lang:   Source language or ``"auto"`` for automatic detection.
        tesseract_cmd: Override path to Tesseract binary.
        font_path:     Override path to a TrueType font file.
        verbose:       Print progress messages.
        log_callback:  Optional callable that receives each log message.
                       When provided, overrides the built-in print logging.

    Returns:
        Path to the saved output file (or directory for multi-page images).
    """
    def log(msg: str):
        if log_callback is not None:
            log_callback(msg)
        elif verbose:
            print(msg)

    input_path = str(input_path)
    if output_path is None:
        output_path = guess_output_path(input_path, target_lang)

    if font_path is None:
        font_path = find_system_font()

    log(f"Loading document: {input_path}")
    pages: List[Image.Image] = load_document(input_path)
    log(f"  Loaded {len(pages)} page(s).")

    processed: List[Image.Image] = []

    for idx, page in enumerate(pages):
        log(f"\n  ── Page {idx + 1} / {len(pages)} ─────────────────")
        result = process_page(
            page_image=page,
            target_lang=target_lang,
            source_lang=source_lang,
            tesseract_cmd=tesseract_cmd,
            font_path=font_path,
            log=lambda msg, _i=idx: log(msg),
        )
        processed.append(result)

    log(f"\nSaving output → {output_path}")
    written = save_output(processed, output_path)
    log("Done!  Written files:")
    for f in written:
        log(f"  {f}")

    return output_path
