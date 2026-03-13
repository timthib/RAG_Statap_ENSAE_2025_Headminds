"""Step 3b: Semantic classification — label each block by type."""


def classify(blocks: list[dict]) -> list[dict]:
    """Assign a semantic label to each block.

    Features used:
        - Font size ratio (relative to page median)
        - Vertical position on page
        - Aspect ratio of bounding box
        - Internal spacing density

    Labels: 'title', 'paragraph', 'table', 'figure', 'caption', 'equation', 'header', 'footer'

    Returns:
        Same list of blocks, each enriched with a 'label' key.
    """
    raise NotImplementedError
