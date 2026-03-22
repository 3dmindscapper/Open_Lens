"""
Command-line interface for the document translation tool.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


_EXAMPLE_LANGS = "en, fr, de, es, it, pt, ru, ja, zh-CN, ar, ko, nl, pl, tr …"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="translate_doc",
        description=(
            "Detect, translate, and replace text in images and PDF documents "
            "while preserving the original layout."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument(
        "input",
        metavar="INPUT",
        help="Path to input file (PDF, JPG, PNG, TIFF, …)",
    )
    p.add_argument(
        "-t", "--target",
        default="en",
        metavar="LANG",
        dest="target_lang",
        help=f"Target language code (default: en). Examples: {_EXAMPLE_LANGS}",
    )
    p.add_argument(
        "-s", "--source",
        default="auto",
        metavar="LANG",
        dest="source_lang",
        help=(
            "Source language code (default: auto-detect).\n"
            "Set this when auto-detection gives wrong results."
        ),
    )
    p.add_argument(
        "-o", "--output",
        default=None,
        metavar="PATH",
        dest="output_path",
        help=(
            "Output file path (default: <input>_translated_<lang>.<ext>).\n"
            "For multi-page images the tool appends _page1, _page2, … automatically."
        ),
    )
    p.add_argument(
        "--tesseract",
        default=None,
        metavar="PATH",
        dest="tesseract_cmd",
        help=(
            "Path to the Tesseract binary if it is not on your PATH.\n"
            "Example (Windows): --tesseract \"C:\\Program Files\\Tesseract-OCR\\tesseract.exe\""
        ),
    )
    p.add_argument(
        "--font",
        default=None,
        metavar="PATH",
        dest="font_path",
        help="Path to a TrueType (.ttf) font file. Auto-detected from system fonts if omitted.",
    )
    p.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress progress output.",
    )

    # --- Engine selection ---------------------------------------------------
    engine_group = p.add_argument_group("Engine selection (auto-detected by default)")
    engine_group.add_argument(
        "--layout-engine",
        default="auto",
        choices=["auto", "layoutparser", "paddleocr", "none"],
        dest="layout_engine",
        help="Layout detection engine (default: auto).",
    )
    engine_group.add_argument(
        "--ocr-engine",
        default="auto",
        choices=["auto", "paddleocr", "tesseract"],
        dest="ocr_engine",
        help="OCR engine (default: auto — prefers PaddleOCR).",
    )
    engine_group.add_argument(
        "--inpaint-engine",
        default="auto",
        choices=["auto", "lama", "telea"],
        dest="inpaint_engine",
        help="Inpainting engine (default: auto — prefers LaMa).",
    )
    engine_group.add_argument(
        "--translator",
        default="auto",
        choices=["auto", "argos", "ollama"],
        dest="translator_engine",
        help="Translation engine (default: auto — uses Argos Translate).",
    )
    engine_group.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Compute device (default: auto).",
    )
    engine_group.add_argument(
        "--ollama-url",
        default="http://localhost:11434",
        metavar="URL",
        dest="ollama_url",
        help="Ollama server URL (default: http://localhost:11434).",
    )
    engine_group.add_argument(
        "--ollama-model",
        default="qwen2.5:7b",
        metavar="MODEL",
        dest="ollama_model",
        help="Ollama model name (default: qwen2.5:7b).",
    )
    engine_group.add_argument(
        "--glossary",
        default=None,
        metavar="PATH",
        dest="glossary_path",
        help="Path to a TSV glossary file (source_term<TAB>target_term).",
    )
    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    if not input_path.exists():
        parser.error(f"Input file not found: {args.input}")

    # Import here so startup is fast even if deps are missing
    try:
        from .pipeline import process_document
        from .config import TranslationConfig
    except ImportError as exc:
        print(f"[ERROR] Missing dependency: {exc}", file=sys.stderr)
        sys.exit(1)

    config = TranslationConfig(
        layout_engine=args.layout_engine,
        ocr_engine=args.ocr_engine,
        inpaint_engine=args.inpaint_engine,
        translator_engine=args.translator_engine,
        device=args.device,
        target_lang=args.target_lang,
        source_lang=args.source_lang,
        tesseract_cmd=args.tesseract_cmd,
        font_path=args.font_path,
        ollama_url=args.ollama_url,
        ollama_model=args.ollama_model,
        glossary_path=args.glossary_path,
    )

    try:
        out = process_document(
            input_path=str(input_path),
            target_lang=args.target_lang,
            output_path=args.output_path,
            source_lang=args.source_lang,
            tesseract_cmd=args.tesseract_cmd,
            font_path=args.font_path,
            verbose=not args.quiet,
            config=config,
        )
        if args.quiet:
            print(out)
    except Exception as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
