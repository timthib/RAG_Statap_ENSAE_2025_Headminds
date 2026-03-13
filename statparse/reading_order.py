"""Step 4: Reading order — sort blocks in natural reading sequence."""


def order_blocks(blocks: list[dict]) -> list[dict]:
    """Sort labeled blocks in reading order.

    Strategy:
        1. Detect columns (using block x-coordinates clustering)
        2. Within each column: sort top-to-bottom
        3. Across columns: sort left-to-right

    Returns:
        Same blocks, reordered.
    """
    raise NotImplementedError
