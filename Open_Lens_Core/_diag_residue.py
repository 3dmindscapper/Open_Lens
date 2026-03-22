"""Diagnostic: find text residue — pixels that are dark in the original but
NOT covered by the inpainting mask. Shows where ghost text will appear."""
import sys
sys.path.insert(0, ".")
import numpy as np
import cv2
from PIL import Image
from translator_tool.ocr_extractor import extract_text_blocks
from translator_tool.language_utils import is_data_only_block
from translator_tool.inpainter import _build_mask

INPUT = (
    r"c:\Users\migne\Desktop\DEBlock\Open_Lens\examples"
    r"\france-certificate-of-good-standing-example.jpg"
)

img = Image.open(INPUT)
blocks = extract_text_blocks(img, ocr_lang="fra")
translatable = [b for b in blocks if not is_data_only_block(b["text"])]

mask = _build_mask(img, translatable)
img_np = np.array(img)
gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

# Look at each translatable block's bounding box and find dark pixels NOT in mask
print("=== Blocks with potential text residue (dark pixels outside mask) ===\n")
for i, b in enumerate(translatable):
    x1, y1, x2, y2 = b["x"], b["y"], b["x2"], b["y2"]
    # Expand a little to check near boundaries
    pad = 10
    rx1 = max(0, x1 - pad)
    ry1 = max(0, y1 - pad)
    rx2 = min(gray.shape[1], x2 + pad)
    ry2 = min(gray.shape[0], y2 + pad)

    roi_gray = gray[ry1:ry2, rx1:rx2]
    roi_mask = mask[ry1:ry2, rx1:rx2]

    # Dark pixels (likely text)
    dark = roi_gray < 100
    # Pixels in mask
    covered = roi_mask > 0
    # Uncovered dark pixels
    leaked = dark & ~covered
    total_dark = int(np.sum(dark))
    total_leaked = int(np.sum(leaked))

    if total_dark > 0 and total_leaked > 5:
        pct = 100 * total_leaked / total_dark
        txt = b["text"][:70].replace("\n", " ")
        print(f"[{i:2d}] leaked={total_leaked:5d}/{total_dark:5d} ({pct:.1f}%)  "
              f"bbox=({x1},{y1},{x2},{y2})")
        print(f"     text={txt!r}")
        # Find where the leaks are
        leak_ys, leak_xs = np.where(leaked)
        if len(leak_ys) > 0:
            # Convert back to image coords
            abs_leak_x_min = rx1 + int(leak_xs.min())
            abs_leak_x_max = rx1 + int(leak_xs.max())
            abs_leak_y_min = ry1 + int(leak_ys.min())
            abs_leak_y_max = ry1 + int(leak_ys.max())
            print(f"     leak region: ({abs_leak_x_min},{abs_leak_y_min})-"
                  f"({abs_leak_x_max},{abs_leak_y_max})")
        print()

# Also check: dark text pixels that are NOT inside ANY block bbox at all
print("\n=== Global: dark pixels not inside any block bbox ===")
all_block_mask = np.zeros_like(mask)
for b in blocks:
    x1, y1, x2, y2 = b["x"], b["y"], b["x2"], b["y2"]
    all_block_mask[y1:y2, x1:x2] = 255

# Global dark pixels
all_dark = gray < 80
outside_blocks = all_dark & (all_block_mask == 0)
total_outside = int(np.sum(outside_blocks))
print(f"Dark pixels outside ALL block bboxes: {total_outside}")
print("(These are watermark/stamp pixels, lines, or OCR-missed text)")
