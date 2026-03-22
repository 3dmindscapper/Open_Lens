"""
Main translation pipeline.

Orchestrates:
    layout detect → OCR → language detection → translate → inpaint → render → save.

All backend choices (layout engine, OCR engine, inpainter, translator) are
controlled by :class:`config.TranslationConfig` with automatic fallback chains.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Callable, Any

from PIL import Image

from .config import TranslationConfig
from .file_handler import load_document, save_output, guess_output_path
from .ocr_extractor import (
    extract_text_blocks,
    extract_text_blocks_unified,
    extract_full_text,
    configure_tesseract_path,
)
from .language_utils import (
    detect_language,
    langdetect_to_tesseract,
    translate_blocks,
    translate_blocks_batch,
)
from .inpainter import remove_text_blocks, remove_text
from .renderer import render_translated_blocks, find_system_font
from .layout_detector import detect_layout, filter_text_regions, crop_region
from .text_classifier import classify_blocks

log = logging.getLogger(__name__)


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
    config: Optional[TranslationConfig] = None,
) -> Image.Image:
    """Run the full translation pipeline on a single page image.

    Args:
        page_image:    RGB PIL Image of one document page.
        target_lang:   Target language code (e.g. ``"en"``, ``"fr"``).
        source_lang:   Source language code or ``"auto"`` to detect.
        tesseract_cmd: Path to the Tesseract binary (optional override).
        font_path:     Path to a .ttf font file (optional override).
        log:           Callable used for progress messages.
        config:        Pipeline configuration. Auto-detected if ``None``.

    Returns:
        Processed PIL Image with translated text in place.
    """
    # Resolve config
    if config is None:
        config = TranslationConfig(
            target_lang=target_lang,
            source_lang=source_lang,
            tesseract_cmd=tesseract_cmd,
            font_path=font_path,
        )
    config.resolve()

    if config.tesseract_cmd:
        configure_tesseract_path(config.tesseract_cmd)

    log(f"    Config: {config.summary()}")

    # ---- Step 1: Layout detection (new) -----------------------------------
    regions = []
    if config.layout_engine != "none":
        log(f"    Detecting layout ({config.layout_engine})…")
        try:
            regions = detect_layout(
                page_image,
                engine=config.layout_engine,
                device=config.device,
                model_label=config.layout_model,
                min_confidence=config.layout_confidence,
            )
            text_regions = filter_text_regions(regions)
            log(f"    Layout: {len(regions)} region(s), {len(text_regions)} text region(s).")
        except Exception as exc:
            log(f"    Layout detection failed ({exc}), continuing without layout.")
            regions = []

    # ---- Step 2: Language detection ----------------------------------------
    if source_lang == "auto":
        log("    Detecting language…")
        sample_text = extract_full_text(page_image, ocr_lang="eng", engine=config.ocr_engine)
        detected_lang = detect_language(sample_text)
        ocr_lang = langdetect_to_tesseract(detected_lang)
        src_for_translation = detected_lang
        log(f"    Detected language: {detected_lang}  →  OCR pack: {ocr_lang}")
    else:
        ocr_lang = langdetect_to_tesseract(source_lang)
        src_for_translation = source_lang
        log(f"    Using source language: {source_lang}  →  OCR pack: {ocr_lang}")

    # ---- Step 3: OCR -------------------------------------------------------
    text_regions = filter_text_regions(regions) if regions else []
    all_blocks = []

    if text_regions:
        # Region-aware OCR: process each layout region independently
        log(f"    Running OCR per-region ({config.ocr_engine}, lang={ocr_lang})…")
        for region in text_regions:
            cropped = crop_region(page_image, region)
            region_blocks = extract_text_blocks_unified(
                cropped,
                ocr_lang=ocr_lang,
                engine=config.ocr_engine,
                device=config.device,
                region_offset=(region.x, region.y),
            )
            # Attach region type metadata to each block
            for b in region_blocks:
                b["_region_type"] = region.region_type
            all_blocks.extend(region_blocks)
    else:
        # No layout regions — full page OCR (original behaviour)
        log(f"    Running OCR ({config.ocr_engine}, lang={ocr_lang})…")
        all_blocks = extract_text_blocks_unified(
            page_image,
            ocr_lang=ocr_lang,
            engine=config.ocr_engine,
            device=config.device,
        )

    # Fallback: if primary engine produces nothing, try Tesseract
    if not all_blocks and config.ocr_engine != "tesseract":
        log("    Primary OCR returned no blocks, retrying with Tesseract…")
        all_blocks = extract_text_blocks(page_image, ocr_lang=ocr_lang)

    if not all_blocks and ocr_lang != "eng":
        log("    Language pack returned no blocks, retrying with 'eng'…")
        all_blocks = extract_text_blocks(page_image, ocr_lang="eng")

    if not all_blocks:
        log("    No text found on this page – returning unchanged.")
        return page_image.copy()

    log(f"    Found {len(all_blocks)} text block(s).")

    # ---- Step 3b: Classify blocks (watermark / stamp / noise filter) ------
    log("    Classifying text blocks…")
    all_blocks = classify_blocks(all_blocks, page_image)
    skip_count = sum(1 for b in all_blocks if b.get("_skip"))
    if skip_count:
        skip_details = {}
        for b in all_blocks:
            if b.get("_skip"):
                cls = b.get("_classification", "unknown")
                skip_details[cls] = skip_details.get(cls, 0) + 1
        detail_str = ", ".join(f"{v} {k}" for k, v in skip_details.items())
        log(f"    Filtered {skip_count} block(s): {detail_str}")
    active_blocks = [b for b in all_blocks if not b.get("_skip")]
    if not active_blocks:
        log("    All blocks filtered — returning unchanged.")
        return page_image.copy()

    # ---- Step 4: Translate -------------------------------------------------
    log(f"    Translating to '{target_lang}' ({config.translator_engine})…")
    active_blocks = translate_blocks_batch(
        active_blocks,
        target_lang=target_lang,
        source_lang=src_for_translation,
        engine=config.translator_engine,
        ollama_url=config.ollama_url or "http://localhost:11434",
        ollama_model=config.ollama_model or "qwen2.5:7b",
        glossary_path=config.glossary_path,
        log_fn=log,
    )

    # ---- Step 5: Separate data-only vs. translatable blocks ----------------
    blocks_to_render = [b for b in active_blocks if not b.get("_data_only")]
    preserved_count = len(active_blocks) - len(blocks_to_render)

    if preserved_count:
        log(f"    Preserving {preserved_count} data-only block(s) unchanged.")

    # ---- Step 6: Inpaint ---------------------------------------------------
    log(f"    Removing original text ({config.inpaint_engine})…")
    inpainted = remove_text(
        page_image,
        blocks_to_render,
        engine=config.inpaint_engine,
    )

    # ---- Step 7: Render translated text ------------------------------------
    log("    Rendering translated text…")
    final = render_translated_blocks(
        image=inpainted,
        blocks=blocks_to_render,
        original_image=page_image,
        font_path=config.font_path or font_path,
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
    config: Optional[TranslationConfig] = None,
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
        config:        Pipeline configuration. Built from other args if ``None``.

    Returns:
        Path to the saved output file (or directory for multi-page images).
    """
    def log_msg(msg: str):
        if log_callback is not None:
            log_callback(msg)
        elif verbose:
            print(msg)

    # Build config if not provided
    if config is None:
        config = TranslationConfig(
            target_lang=target_lang,
            source_lang=source_lang,
            tesseract_cmd=tesseract_cmd,
            font_path=font_path,
        )
    config.resolve()

    input_path = str(input_path)
    if output_path is None:
        output_path = guess_output_path(input_path, target_lang)

    if font_path is None and config.font_path is None:
        config.font_path = find_system_font()
    elif font_path:
        config.font_path = font_path

    log_msg(f"Loading document: {input_path}")
    log_msg(f"  Pipeline: {config.summary()}")
    pages: List[Image.Image] = load_document(input_path)
    log_msg(f"  Loaded {len(pages)} page(s).")

    processed: List[Image.Image] = []

    for idx, page in enumerate(pages):
        log_msg(f"\n  ── Page {idx + 1} / {len(pages)} ─────────────────")
        result = process_page(
            page_image=page,
            target_lang=target_lang,
            source_lang=source_lang,
            tesseract_cmd=tesseract_cmd,
            font_path=config.font_path,
            log=lambda msg, _i=idx: log_msg(msg),
            config=config,
        )
        processed.append(result)

    log_msg(f"\nSaving output → {output_path}")
    written = save_output(processed, output_path)
    log_msg("Done!  Written files:")
    for f in written:
        log_msg(f"  {f}")

    return output_path
