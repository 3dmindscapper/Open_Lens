"""Diagnostic: dump all OCR blocks and their classification."""
import sys
sys.path.insert(0, ".")
from PIL import Image
from translator_tool.ocr_extractor import extract_text_blocks
from translator_tool.language_utils import is_data_only_block

img = Image.open(
    r"c:\Users\migne\Desktop\DEBlock\Open_Lens\examples"
    r"\france-certificate-of-good-standing-example.jpg"
)
blocks = extract_text_blocks(img, ocr_lang="fra")
print(f"Total blocks: {len(blocks)}\n")

for i, b in enumerate(blocks):
    data = is_data_only_block(b["text"])
    wc = len(b.get("word_boxes", []))
    x, y, x2, y2 = b["x"], b["y"], b["x2"], b["y2"]
    txt = b["text"][:90].replace("\n", " ")
    print(f"[{i:2d}] data={str(data):5s}  words={wc:2d}  "
          f"bbox=({x},{y},{x2},{y2})  text={txt!r}")
