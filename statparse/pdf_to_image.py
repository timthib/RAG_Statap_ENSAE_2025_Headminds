"""Step 1: PDF to Image rendering.

Uses pdf2image (Poppler wrapper) to rasterize PDF pages into numpy arrays.
Requires: pip install pdf2image
System:   brew install poppler  (macOS)  /  apt install poppler-utils  (Linux)
"""
import numpy as np
from pdf2image import convert_from_path


def render_pdf(pdf_path: str, dpi: int = 300) -> list[np.ndarray]:
    """Convert a PDF file to a list of page images (numpy arrays, RGB).

    Args:
        pdf_path: path to the PDF file.
        dpi: rendering resolution. 300 is standard for OCR.

    Returns:
        List of numpy arrays of shape (H, W, 3), dtype uint8, in RGB order.
    """
    pil_images = convert_from_path(pdf_path, dpi=dpi)
    return [np.array(img) for img in pil_images]
