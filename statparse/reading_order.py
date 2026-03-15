"""Step 4: Reading order — sort blocks in natural reading sequence."""
"""sans utiliser layoutparser dans les autres étapes, il n'y a pas de libraire qui fasse spécifiquement le reading order :
c'est intégré dans d'autres étapes au sein d'une pipeline plus grosse.
je choisis donc une solution hard codée. Source : Claude """

import numpy as np

def order_blocks(blocks: list[dict]) -> list[dict]:
    """Sort labeled blocks in natural reading sequence.

    Strategy:
        1. Detect columns (using block x-coordinates clustering)
        2. Within each column: sort top-to-bottom
        3. Across columns: sort left-to-right

    Returns:
        Same blocks, reordered.
    """
    if not blocks:
        return blocks

    # headers et footers en premier et dernier
    headers = [b for b in blocks if b.get("label") == "header"]
    footers = [b for b in blocks if b.get("label") == "footer"]
    body = [b for b in blocks if b.get("label") not in ("header", "footer")]

    if not body:
        return headers + footers

    columns = _detect_columns(body)

    # trier chaque colonne de haut en bas
    for col in columns:
        col.sort(key=lambda b: b["bbox"][1])

    # trier les colonnes de gauche à droite par x median
    columns.sort(key=lambda col: np.median([b["bbox"][0] for b in col]))

    ordered_body = [block for col in columns for block in col]

    return headers + ordered_body + footers


# ─────────────────────────── helpers ────────────────────────────


def _detect_columns(blocks: list[dict]) -> list[list[dict]]:
    """Cluster blocks into columns by their x-center using 1D gap detection."""
    if not blocks:
        return []

    # x-center de chaque bloc
    x_centers = np.array([b["bbox"][0] + b["bbox"][2] / 2 for b in blocks])
    order = np.argsort(x_centers)
    sorted_centers = x_centers[order]
    sorted_blocks = [blocks[i] for i in order]

    # détecter les gaps significatifs entre x-centers consécutifs
    gaps = np.diff(sorted_centers)
    if len(gaps) == 0:
        return [blocks]

    threshold = _gap_threshold(gaps)

    # couper aux gaps > threshold
    columns = []
    current_col = [sorted_blocks[0]]
    for i, gap in enumerate(gaps):
        if gap > threshold:
            columns.append(current_col)
            current_col = []
        current_col.append(sorted_blocks[i + 1])
    columns.append(current_col)

    return columns


def _gap_threshold(gaps: np.ndarray) -> float:
    """Seuil = moyenne + 1 écart-type des gaps.
    Sépare les petits gaps intra-colonne des grands gaps inter-colonnes.
    """
    return float(np.mean(gaps) + np.std(gaps))