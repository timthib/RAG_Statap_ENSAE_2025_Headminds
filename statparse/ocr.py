"""Step 5: OCR — extract text from image regions using Tesseract.

Uses pytesseract as a Python wrapper around the Tesseract OCR engine.
Requires: pip install pytesseract
System:   brew install tesseract  (macOS)  /  apt install tesseract-ocr  (Linux)
"""
import numpy as np
import pytesseract


def recognize_text(blocks: list[dict], page_image: np.ndarray) -> list[dict]:
    """Run Tesseract OCR on each block's bounding box region.

    Args:
        blocks: list of block dicts with at least 'bbox' (x, y, w, h) and 'label'.
        page_image: the original RGB page image as a numpy array.

    Returns:
        Same blocks, each enriched with a 'text' key.
    """
    h_img, w_img = page_image.shape[:2]

    for block in blocks:
        # Skip non-text regions
        if block.get("label") in ("figure",):
            block["text"] = ""
            continue

        x, y, w, h = block["bbox"]

        # Clamp to image bounds
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(w_img, x + w)
        y2 = min(h_img, y + h)

        crop = page_image[y1:y2, x1:x2]

        if crop.size == 0:
            block["text"] = ""
            continue

        # Run Tesseract — PSM 6 = assume a single uniform block of text
        text = pytesseract.image_to_string(crop, config="--psm 6")
        block["text"] = text.strip()

    return blocks
