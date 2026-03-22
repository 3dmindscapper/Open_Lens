"""Diagnostic: generate a mask overlay + block boxes image for visual inspection."""
import sys
sys.path.insert(0, ".")
import numpy as np
import cv2
from PIL import Image
from translator_tool.ocr_extractor import extract_text_blocks
from translator_tool.language_utils import is_data_only_block, translate_blocks_batch
from translator_tool.inpainter import _build_mask, _dynamic_padding

INPUT = (
    r"c:\Users\migne\Desktop\DEBlock\Open_Lens\examples"
    r"\france-certificate-of-good-standing-example.jpg"
)
OUTDIR = r"c:\Users\migne\Desktop\DEBlock\Open_Lens\examples"

img = Image.open(INPUT)
blocks = extract_text_blocks(img, ocr_lang="fra")

# Separate data-only
translatable = []
data_only = []
for b in blocks:
    if is_data_only_block(b["text"]):
        data_only.append(b)
    else:
        translatable.append(b)

print(f"Total: {len(blocks)}, Translatable: {len(translatable)}, Data-only: {len(data_only)}")

# Build the mask that the inpainter would use
mask = _build_mask(img, translatable)

# Create overlay image
img_np = np.array(img)
overlay = img_np.copy()

# Draw mask regions in semi-transparent red
red_overlay = overlay.copy()
red_overlay[mask > 0] = [255, 0, 0]
overlay = cv2.addWeighted(overlay, 0.6, red_overlay, 0.4, 0)

# Draw translatable block boxes in green, data-only in yellow
for b in translatable:
    cv2.rectangle(overlay, (b["x"], b["y"]), (b["x2"], b["y2"]), (0, 255, 0), 2)
    # Draw word boxes in cyan
    for wb in b.get("word_boxes", []):
        x2 = wb.get("x2", wb["x"] + wb["w"])
        y2 = wb.get("y2", wb["y"] + wb["h"])
        cv2.rectangle(overlay, (wb["x"], wb["y"]), (x2, y2), (0, 255, 255), 1)

for b in data_only:
    cv2.rectangle(overlay, (b["x"], b["y"]), (b["x2"], b["y2"]), (0, 255, 255), 2)

# Save
out_path = OUTDIR + r"\debug_mask_overlay.jpg"
Image.fromarray(overlay).save(out_path, quality=95)
print(f"Saved: {out_path}")

# Also save just the mask
mask_path = OUTDIR + r"\debug_mask.png"
Image.fromarray(mask).save(mask_path)
print(f"Saved: {mask_path}")
