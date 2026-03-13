"""
Handles loading of PDF, JPG, and PNG documents into lists of PIL Images,
and saving processed images back to file.
"""

from pathlib import Path
from typing import List, Optional

from PIL import Image


SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"}


def load_document(input_path: str) -> List[Image.Image]:
    """Load a document file and return a list of PIL RGB images (one per page).

    Supports PDF, JPG, PNG, TIFF, BMP, and WebP.

    Args:
        input_path: Path to the input file.

    Returns:
        List of PIL Images in RGB mode.

    Raises:
        ValueError: If the file format is not supported.
        ImportError: If pdf2image is not installed when loading a PDF.
        FileNotFoundError: If the input path does not exist.
    """
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return _load_pdf(str(path))
    elif suffix in SUPPORTED_IMAGE_EXTS:
        img = Image.open(str(path)).convert("RGB")
        return [img]
    else:
        raise ValueError(
            f"Unsupported file format '{suffix}'. "
            f"Supported: PDF, {', '.join(sorted(SUPPORTED_IMAGE_EXTS))}"
        )


def _load_pdf(pdf_path: str) -> List[Image.Image]:
    """Convert each page of a PDF to a PIL Image at 200 DPI."""
    try:
        from pdf2image import convert_from_path
    except ImportError:
        raise ImportError(
            "pdf2image is required for PDF support.\n"
            "Install it with:  pip install pdf2image\n"
            "You also need Poppler on your system:\n"
            "  Windows: https://github.com/oschwartz10612/poppler-windows/releases\n"
            "  macOS:   brew install poppler\n"
            "  Linux:   sudo apt-get install poppler-utils"
        )

    images = convert_from_path(pdf_path, dpi=200)
    return [img.convert("RGB") for img in images]


def save_output(
    images: List[Image.Image],
    output_path: str,
    original_suffix: Optional[str] = None,
) -> List[str]:
    """Save processed images to disk.

    - Single-page PDF/image → one output file.
    - Multi-page PDF → one multi-page PDF.
    - Multi-page image input  → numbered files (page_1.ext, page_2.ext, …).

    Args:
        images:         Processed PIL Images.
        output_path:    Desired output file path.
        original_suffix: The suffix of the input file (used to decide format).

    Returns:
        List of paths that were written.
    """
    path = Path(output_path)
    suffix = path.suffix.lower()
    written = []

    if suffix == ".pdf":
        if len(images) == 1:
            images[0].save(str(path), "PDF", resolution=200)
        else:
            images[0].save(
                str(path),
                "PDF",
                save_all=True,
                append_images=images[1:],
                resolution=200,
            )
        written.append(str(path))

    elif len(images) == 1:
        images[0].save(str(path))
        written.append(str(path))

    else:
        # Multiple images saved as numbered files
        stem = path.stem
        parent = path.parent
        ext = path.suffix
        for i, img in enumerate(images, start=1):
            out = parent / f"{stem}_page{i}{ext}"
            img.save(str(out))
            written.append(str(out))

    return written


def guess_output_path(input_path: str, target_lang: str) -> str:
    """Generate a default output path from the input path and target language."""
    p = Path(input_path)
    return str(p.parent / f"{p.stem}_translated_{target_lang}{p.suffix}")
