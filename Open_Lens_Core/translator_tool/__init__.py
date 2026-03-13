"""
Document Translation Tool
Intakes JPG, PNG, and PDF files, detects text and its language,
translates the text, removes the original, and renders the
translated text back in the same position with matching formatting.
"""

from .pipeline import process_document

__all__ = ["process_document"]
