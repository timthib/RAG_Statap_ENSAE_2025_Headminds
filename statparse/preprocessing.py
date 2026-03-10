"""Step 2: Image preprocessing — binarization, deskew, denoising."""
import numpy as np


def preprocess(image: np.ndarray) -> np.ndarray:
    """Take an RGB page image, return a cleaned binary image.

    Steps:
        1. Convert to grayscale
        2. Binarize (Otsu or Sauvola thresholding)
        3. Deskew (projection profile variance maximization)
        4. Denoise (morphological operations)
    """
    raise NotImplementedError
