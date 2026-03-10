"""Step 6: Markdown serialization — convert labeled text blocks to Markdown.

No external dependencies. Deterministic mapping from labels to Markdown syntax.
"""


def to_markdown(blocks: list[dict]) -> str:
    """Convert ordered, labeled, OCR'd blocks into a Markdown string.

    Each block must have keys: 'label', 'text', and optionally 'heading_level'.

    Returns:
        A markdown-formatted string for the full page.
    """
    lines = []

    for block in blocks:
        text = block.get("text", "").strip()
        if not text:
            continue

        label = block.get("label", "paragraph")

        if label == "title":
            level = block.get("heading_level", 1)
            prefix = "#" * level
            lines.append(f"{prefix} {text}")

        elif label == "paragraph":
            lines.append(text)

        elif label == "equation":
            lines.append(f"$$\n{text}\n$$")

        elif label == "table":
            # Text from OCR on a table region — pass through as-is
            # (proper table reconstruction is a harder problem)
            lines.append(text)

        elif label == "caption":
            lines.append(f"*{text}*")

        elif label in ("header", "footer"):
            # Usually not useful for RAG, skip
            continue

        elif label == "figure":
            # No text content
            continue

        else:
            # Fallback: treat as paragraph
            lines.append(text)

    return "\n\n".join(lines)
