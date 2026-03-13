"""Step 3a: Geometric segmentation — connected components + hierarchical grouping."""
import numpy as np


def segment(binary_image: np.ndarray) -> list[dict]:
    """Segment a binary image into blocks using statistical spatial clustering.

    Pipeline:
        1. Extract connected components
        2. Compute k-NN distances and angles between components
        3. Model distance distributions (KDE / GMM)
        4. Find valley thresholds to separate intra-group vs inter-group gaps
        5. Group hierarchically: characters → words → lines → blocks → columns

    Returns:
        List of block dicts, each with keys:
            - 'bbox': (x, y, w, h)
            - 'components': list of connected component bboxes inside this block
    """
    raise NotImplementedError
