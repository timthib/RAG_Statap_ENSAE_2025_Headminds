"J'utilise dans un premier temps des 'règles heuristiques' et non la librairie layoutparser car lourde à installer"

"""Step 3b: Semantic classification — label each block by type."""
import numpy as np


def classify(blocks: list[dict], image_shape: tuple) -> list[dict]:
    """Assign a semantic label to each block.

    Features used:
        - Font size ratio (relative to page median)
        - Vertical position on page
        - Aspect ratio of bounding box
        - Internal spacing density

    Labels: 'title', 'paragraph', 'table', 'figure', 'caption', 'equation', 'header', 'footer'

    Args:
        blocks:      output of segment(), list of dicts with 'bbox' and 'components'
        image_shape: (H, W) of the original page image

    Returns:
        Same list of blocks, each enriched with a 'label' key.
    """
    if not blocks:
        return blocks

    page_h, page_w = image_shape

    # ── feature extraction ──────────────────────────────────────
    features = [_extract_features(b, page_h, page_w) for b in blocks]

    # page median component height → proxy for body font size
    all_heights = [f["median_comp_h"] for f in features if f["median_comp_h"] > 0]
    median_font = float(np.median(all_heights)) if all_heights else 10.0

    for block, feat in zip(blocks, features):
        block["label"] = _classify_block(feat, median_font, page_h, page_w)

    return blocks


# ─────────────────────────── helpers ────────────────────────────


def _extract_features(block: dict, page_h: int, page_w: int) -> dict:
    x, y, w, h = block["bbox"]
    components = block.get("components", [])

    comp_heights = [c["bbox"][3] for c in components if c["bbox"][3] > 0]
    median_comp_h = float(np.median(comp_heights)) if comp_heights else 0.0

    aspect_ratio = w / h if h > 0 else 0.0
    rel_y_top = y / page_h          # 0 = top, 1 = bottom
    rel_y_bot = (y + h) / page_h
    rel_width = w / page_w
    rel_height = h / page_h

    # density: fraction of bbox area covered by components
    comp_area = sum(c["bbox"][2] * c["bbox"][3] for c in components)
    bbox_area = w * h if w * h > 0 else 1
    density = comp_area / bbox_area

    # horizontal regularity → table heuristic
    # tables tend to have components aligned on a grid (low x-variance per row)
    row_variance = _row_alignment_variance(components) if len(components) >= 4 else 1.0

    return {
        "median_comp_h": median_comp_h,
        "aspect_ratio": aspect_ratio,
        "rel_y_top": rel_y_top,
        "rel_y_bot": rel_y_bot,
        "rel_width": rel_width,
        "rel_height": rel_height,
        "density": density,
        "row_variance": row_variance,
        "n_components": len(components),
        "bbox_w": w,
        "bbox_h": h,
    }


def _row_alignment_variance(components: list[dict]) -> float:
    """Low variance → components aligned in rows (table-like)."""
    ys = [c["bbox"][1] for c in components]
    if not ys:
        return 1.0
    # bin components into rows by y proximity
    ys_arr = np.array(sorted(ys))
    diffs = np.diff(ys_arr)
    return float(np.std(diffs)) / (float(np.mean(diffs)) + 1e-6)


def _classify_block(feat: dict, median_font: float, page_h: int, page_w: int) -> str:

    mh = feat["median_comp_h"]
    font_ratio = mh / median_font if median_font > 0 else 1.0

    # ── header / footer (position-based) ────────────────────────
    if feat["rel_y_top"] < 0.05 and feat["rel_height"] < 0.08:
        return "header"
    if feat["rel_y_bot"] > 0.95 and feat["rel_height"] < 0.08:
        return "footer"

    # ── figure (large bbox, very low text density) ───────────────
    if feat["density"] < 0.08 and feat["rel_height"] > 0.1:
        return "figure"

    # ── table (wide, moderate density, regular row alignment) ────
    if (
        feat["rel_width"] > 0.4
        and feat["density"] > 0.05
        and feat["row_variance"] < 0.5
        and feat["aspect_ratio"] > 1.5
    ):
        return "table"

    # ── title (large font, short block, near top half) ───────────
    if (
        font_ratio > 1.4
        and feat["rel_height"] < 0.12
        and feat["rel_y_top"] < 0.6
        and feat["rel_width"] > 0.2
    ):
        return "title"

    # ── caption (small font, short block, below figure/table) ────
    if font_ratio < 0.85 and feat["rel_height"] < 0.06 and feat["n_components"] > 3:
        return "caption"

    # ── equation (narrow, low component count, centered) ─────────
    if (
        feat["n_components"] < 10
        and feat["rel_width"] < 0.6
        and 0.3 < (feat["bbox_w"] / page_w + feat["rel_y_top"]) < 1.4  # roughly centered
        and feat["rel_height"] < 0.05
    ):
        return "equation"

    # ── default: paragraph ───────────────────────────────────────
    return "paragraph"