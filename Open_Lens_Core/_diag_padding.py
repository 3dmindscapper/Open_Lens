"""Check dynamic padding values and whether word boxes cover their ink."""
import sys
sys.path.insert(0, ".")
import numpy as np
import cv2
from PIL import Image
from translator_tool.ocr_extractor import extract_text_blocks
from translator_tool.language_utils import is_data_only_block
from translator_tool.inpainter import _dynamic_padding

INPUT = (
    r"c:\Users\migne\Desktop\DEBlock\Open_Lens\examples"
    r"\france-certificate-of-good-standing-example.jpg"
)

img = Image.open(INPUT)
blocks = extract_text_blocks(img, ocr_lang="fra")
translatable = [b for b in blocks if not is_data_only_block(b["text"])]

# Check the problematic blocks: 18(idx in translatable), 27, 38
problem_indices = [18, 27, 38]  # from the original block list

for b in translatable:
    pad = _dynamic_padding(b)
    wbs = b.get("word_boxes", [])
    heights = [wb["h"] for wb in wbs] if wbs else [b["h"]]
    avg_h = sum(heights) / len(heights)
    txt = b["text"][:60].replace("\n", " ")
    bbox_h = b["y2"] - b["y"]
    if pad < 12 or bbox_h > 25:  # Show blocks where padding might be insufficient
        print(f"pad={pad:2d}  avg_h={avg_h:.0f}  bbox_h={bbox_h:3d}  "
              f"bbox=({b['x']},{b['y']},{b['x2']},{b['y2']})  "
              f"text={txt!r}")
