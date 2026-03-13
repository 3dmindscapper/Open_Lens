#!/usr/bin/env python
"""
Root-level entry point.  Run from the project directory:

    python translate.py document.pdf -t en
    python translate.py scan.jpg -t fr -s de -o output.jpg
    python translate.py invoice.png -t es --tesseract "C:\Program Files\Tesseract-OCR\tesseract.exe"
"""

import sys
import os

# Ensure the package root is importable whether installed or run in-tree
sys.path.insert(0, os.path.dirname(__file__))

from translator_tool.main import main

if __name__ == "__main__":
    main()
