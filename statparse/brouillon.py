from preprocessing import preprocess
from segmentation import segment
from classification import classify
from reading_order import order_blocks
from ocr import recognize_text
from PIL import Image
import cv2
import numpy as np


#img = Image.open("/home/onyxia/RAG_Statap_ENSAE_2025_Headminds/data/images/docstructbench_enbook-zlib-o.O-17761417.pdf_894.jpg")
#image = np.array(img)
#image2 = preprocess(image)
#seg = segment(image2)
#Image.fromarray(image2).save("/home/onyxia/RAG_Statap_ENSAE_2025_Headminds/output_preprocessed.png")

def visualize_segments(image: np.ndarray, blocks: list[dict], output_path: str):
    vis = image.copy()
    if vis.ndim == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    COLORS = {
        "title":     (255, 0,   0),
        "paragraph": (0,   200, 0),
        "table":     (0,   0,   255),
        "figure":    (255, 165, 0),
        "caption":   (128, 0,   128),
        "header":    (0,   255, 255),
        "footer":    (255, 255, 0),
        "equation":  (255, 20,  147),
    }

    for block in blocks:
        x, y, w, h = block["bbox"]
        label = block.get("label", "?")
        color = COLORS.get(label, (200, 200, 200))
        cv2.rectangle(vis, (x, y), (x + w, y + h), color=color, thickness=2)
        cv2.putText(vis, label, (x, max(y - 5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)).save(output_path)


def visualize_reading_order(image: np.ndarray, blocks: list[dict], output_path: str):
    """Dessine les blocs numérotés dans l'ordre de lecture."""
    vis = image.copy()
    if vis.ndim == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    for i, block in enumerate(blocks):
        x, y, w, h = block["bbox"]
        cv2.rectangle(vis, (x, y), (x + w, y + h), color=(0, 0, 255), thickness=2)
        # numéro d'ordre en grand au centre du bloc
        cx, cy = x + w // 2, y + h // 2
        cv2.putText(vis, str(i + 1), (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2, cv2.LINE_AA)

    Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)).save(output_path)

def visualize_ocr(image: np.ndarray, blocks: list[dict], output_path: str):
    """Affiche le texte OCR extrait au-dessus de chaque bloc."""
    vis = image.copy()
    if vis.ndim == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    for block in blocks:
        x, y, w, h = block["bbox"]
        text = block.get("text", "").strip()
        # première ligne seulement pour ne pas surcharger
        first_line = text.split("\n")[0][:60]

        cv2.rectangle(vis, (x, y), (x + w, y + h), color=(0, 180, 0), thickness=2)
        cv2.putText(vis, first_line, (x, max(y - 5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 180, 0), 1, cv2.LINE_AA)

    Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)).save(output_path)


img = Image.open("/home/onyxia/RAG_Statap_ENSAE_2025_Headminds/output_preprocessed.png")
image = np.array(img)
image2 = preprocess(image)
blocks = segment(image2)
blocks = classify(blocks, image_shape=image2.shape)
blocks = order_blocks(blocks)
blocks = recognize_text(blocks, image)



visualize_ocr(image, blocks, "/home/onyxia/RAG_Statap_ENSAE_2025_Headminds/output_ocr.png")
#visualize_reading_order(image, blocks, "/home/onyxia/RAG_Statap_ENSAE_2025_Headminds/output_reading_order.png")
#visualize_segments(image, blocks, "/home/onyxia/RAG_Statap_ENSAE_2025_Headminds/output_segments.png")

