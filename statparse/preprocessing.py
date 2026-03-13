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
    
    gray = convertgrayscale(image)
    #1. grayscale
    binarizer = SauvolaBinarizer()
    #2. binarizer
    binary = binarizer.binarize(gray)
    raise NotImplementedError


def convertgrayscale(image:np.ndarray) -> np.ndarray:
  
    if image.ndim == 2:
        # déjà en grayscale
        return image.astype(np.float64)

    if image.shape[2] != 3:
        raise ValueError("Image RGB attendue")

    # formule standard de luminance
    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]

    gray = 0.299 * r + 0.587 * g + 0.114 * b

    return gray.astype(np.float64)

class SauvolaBinarizer:

    def __init__(self, window_size=31, k=0.35, R=128):

        self.window_size = window_size
        self.k = k
        self.R = R
        # k et R sont arbitraires selon ce qui marche le mieux dans la littérature
    
    def binarize(self, image : np.ndarray) -> np.ndarray:

        if image.ndim != 2:
            raise ValueError("Image grayscale requise")

        r = self.window_size // 2
        padded = np.pad(image, r, mode="reflect")
        H, W = padded.shape

        integral = np.zeros((H + 1, W + 1))
        integral_sq = np.zeros((H + 1, W + 1))

        integral[1:, 1:] = np.cumsum(np.cumsum(padded, axis=0), axis=1)
        integral_sq[1:, 1:] = np.cumsum(np.cumsum(padded**2, axis=0), axis=1)

        h, w = image.shape
        area = self.window_size * self.window_size

        y0 = np.arange(h)
        x0 = np.arange(w)
        y1 = y0
        x1 = x0
        y2 = y0 + self.window_size
        x2 = x0 + self.window_size

        sum_ = (
            integral[y2[:, None], x2]
            - integral[y1[:, None], x2]
            - integral[y2[:, None], x1]
            + integral[y1[:, None], x1]
        )

        sum_sq = (
            integral_sq[y2[:, None], x2]
            - integral_sq[y1[:, None], x2]
            - integral_sq[y2[:, None], x1]
            + integral_sq[y1[:, None], x1]
        )

        mean = sum_ / area
        var = sum_sq / area - mean**2
        std = np.sqrt(np.maximum(var, 0))

        T = mean * (1 + self.k * (std / self.R - 1))
        binary = np.where(image < T, 0, 255).astype(np.uint8)
        return binary
