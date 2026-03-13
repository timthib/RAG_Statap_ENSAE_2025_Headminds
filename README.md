# StatParse: A Statistical Document Parsing Pipeline for RAG Systems

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## Overview

**StatParse** is a lightweight, fully statistical document parsing pipeline that converts PDF documents into structured Markdown — without relying on deep learning or GPU resources.

The core idea: at every level of document structure (characters → words → lines → blocks → columns), spatial relationships follow **statistical distributions**. By modeling these distributions explicitly (using kernel density estimation, Gaussian mixture models, and adaptive thresholding), we can segment and classify document elements with **zero training data** and **zero GPU compute**.

StatParse is designed to plug into Retrieval-Augmented Generation (RAG) systems as the parsing front-end.

## Motivation

Modern document parsing methods (LayoutParser, Docling, DiT) achieve high accuracy but depend on large pretrained models. We explore how far **classical statistics and computer vision** can go, producing an interpretable, reproducible, and resource-free alternative suitable for constrained environments.

This is a research project. We benchmark StatParse against [Docling](https://github.com/DS4SD/docling) (our baseline) on standard document layout analysis benchmarks, then integrate both into a RAG system for end-to-end comparison.

## Pipeline Architecture
PDF ──► Page Images ──► Preprocessing ──► Geometric Segmentation ──► Semantic Classification ──► Reading Order ──► OCR ──► Markdown

| Stage | Method | Key Technique |
|-------|--------|---------------|
| **1. PDF to Image** | Rendering at 300 DPI | `pdf2image` / `PyMuPDF` |
| **2. Preprocessing** | Binarization, deskew, noise removal | Otsu/Sauvola, projection profile variance maximization |
| **3a. Geometric Segmentation** | Hierarchical spatial clustering | Connected components → k-NN distance/angle distributions → KDE/GMM-based grouping at each level (characters → words → lines → blocks → columns) |
| **3b. Semantic Classification** | Rule-based + lightweight statistical classifier | Features: font size ratios, position, aspect ratio, spacing — classified via decision tree or logistic regression |
| **4. Reading Order** | Topological sort on spatial graph | Column-aware top-to-bottom, left-to-right ordering |
| **5. OCR** | Tesseract (legacy mode) | Template matching + n-gram language model correction |
| **6. Markdown Serialization** | Deterministic mapping | Label → Markdown syntax with heading level inference via clustering |

### Geometric Segmentation Detail

The core novelty lies in step 3a. At each grouping level, the same statistical principle applies:

1. Compute pairwise distances between elements
2. Model the distance distribution (KDE or GMM)
3. Identify the valley between intra-group and inter-group distance modes
4. Use the valley as an adaptive, document-specific threshold
5. Group elements below the threshold
Connected Components
        │
        ▼
   k-NN distances + angles
        │
        ├── angle histogram ──► orientation detection
        │
        ├── horizontal distance distribution ──► word grouping (intra-word vs inter-word gap)
        │
        ├── vertical distance distribution ──► line grouping (intra-line vs inter-line gap)
        │
        └── block-level gap distribution ──► block grouping + column detection

This approach is inspired by the **Docstrum algorithm** (O'Gorman, 1993) but extends it with explicit distribution modeling and a unified hierarchical framework.

## Project Structure
statparse/
├── src/
│   ├── pdf_to_image/          # Step 1: PDF rendering
│   ├── preprocessing/         # Step 2: Binarization, deskew, denoising
│   ├── segmentation/          # Step 3a: Geometric segmentation
│   ├── classification/        # Step 3b: Semantic classification
│   ├── reading_order/         # Step 4: Block ordering
│   ├── ocr/                   # Step 5: Tesseract integration
│   ├── serialization/         # Step 6: Markdown output
│   └── pipeline.py            # End-to-end pipeline
├── baselines/
│   └── docling_baseline.py    # Docling baseline wrapper
├── evaluation/
│   ├── benchmarks/            # Benchmark datasets and loaders
│   ├── metrics.py             # Evaluation metrics
│   └── compare.py             # StatParse vs Docling comparison
├── rag/
│   ├── chunking.py            # Document chunking strategies
│   ├── retrieval.py           # Retrieval pipeline
│   └── generation.py          # Generation with retrieved context
├── notebooks/                 # Exploratory analysis and demos
├── tests/                     # Unit tests per module
├── docs/                      # Documentation and literature notes
├── requirements.txt
└── README.md

## Installation

```bash
git clone https://github.com/<your-org>/statparse.git
cd statparse
pip install -r requirements.txt
Dependencies
pdf2image>=1.16.3
PyMuPDF>=1.23.0
opencv-python>=4.8.0
numpy>=1.24.0
scipy>=1.11.0
scikit-learn>=1.3.0
pytesseract>=0.3.10
matplotlib>=3.7.0
System Requirements
# Tesseract OCR engine
sudo apt-get install tesseract-ocr    # Ubuntu/Debian
brew install tesseract                  # macOS

# Poppler (for pdf2image)
sudo apt-get install poppler-utils     # Ubuntu/Debian
brew install poppler                    # macOS
No GPU required. StatParse runs entirely on CPU.
Usage
Basic
from statparse import Pipeline

pipeline = Pipeline()
result = pipeline.parse("document.pdf")

# Save as markdown
with open("output.md", "w") as f:
    f.write(result.to_markdown())
Step-by-Step
from statparse.pdf_to_image import render_pdf
from statparse.preprocessing import preprocess
from statparse.segmentation import segment
from statparse.classification import classify
from statparse.reading_order import order_blocks
from statparse.ocr import recognize_text
from statparse.serialization import to_markdown

images = render_pdf("document.pdf", dpi=300)

for page_image in images:
    clean = preprocess(page_image)
    blocks = segment(clean)
    labeled_blocks = classify(blocks)
    ordered_blocks = order_blocks(labeled_blocks)
    text_blocks = recognize_text(ordered_blocks, page_image)
    markdown = to_markdown(text_blocks)
Evaluation
Benchmarks
We evaluate on:
Copier le tableau


Benchmark
Task
Reference



PRImA Layout Analysis
Layout segmentation
Antonacopoulos et al., ICDAR 2009


DocLayNet
Document layout analysis
Pfitzmann et al., KDD 2022


PubLayNet
Scientific document layout
Zhong et al., ICDAR 2019


Metrics
Copier le tableau


Metric
What it measures



IoU (Intersection over Union)
Geometric segmentation accuracy


mAP@0.5 / mAP@0.75
Detection quality at different overlap thresholds


Block classification accuracy
Semantic label correctness


Reading order Kendall's τ
Ordering quality vs ground truth


Levenshtein / BLEU on Markdown
End-to-end output quality


Running Evaluation
# Run StatParse on benchmark
python evaluation/compare.py --method statparse --dataset prima

# Run Docling baseline
python evaluation/compare.py --method docling --dataset prima

# Compare results
python evaluation/compare.py --compare statparse docling --dataset prima
Roadmap

 Literature review and pipeline design
 Implement Docling baseline and compute baseline metrics
 Implement preprocessing (binarization, deskew, denoising)
 Implement geometric segmentation (connected components + hierarchical grouping)
 Implement semantic classification
 Implement reading order
 Integrate Tesseract OCR
 Implement Markdown serialization
 Benchmark StatParse vs Docling on layout analysis
 Integrate both methods into a RAG system
 End-to-end RAG evaluation (parsing → retrieval → generation)
 Write and submit paper

Key References
Copier le tableau


Paper
Relevance



O'Gorman, "The Document Spectrum for Page Layout Analysis," IEEE TPAMI, 1993. DOI
Core method: k-NN distance/angle distributions for layout analysis


Ha, Haralick & Phillips, "Recursive X-Y Cut," ICDAR, 1995. DOI
Hierarchical top-down segmentation


Wong, Casey & Wahl, "Document Analysis System," IBM JRD, 1982. DOI
RLSA and projection profiles


Breuel, "Two Geometric Algorithms for Layout Analysis," DAS, 2002. DOI
Whitespace-based segmentation


Kise, Sato & Iwata, "Segmentation Using Area Voronoi Diagram," CVIU, 1998. DOI
Voronoi-based approach for complex layouts


Binmakhashen & Mahmoud, "Document Layout Analysis: A Comprehensive Survey," ACM CSUR, 2019. DOI
Modern survey covering classical and DL methods


Mao, Rosenfeld & Kanungo, "Document Structure Analysis Algorithms: A Literature Survey," SPIE, 2003. DOI
Foundational survey


Auer et al., "Docling Technical Report," arXiv, 2024. arXiv
Our baseline system


Contributing
This is an academic research project. If you want to contribute:

Fork the repository
Create a feature branch (git checkout -b feature/step-3a-segmentation)
Write tests for your module
Submit a pull request with a description of what you implemented and why

Please follow the module structure. Each pipeline step is an independent module with clear input/output contracts.
License
MIT License. See LICENSE for details.
Acknowledgments
This project is conducted as part of a student research initiative exploring lightweight statistical alternatives to deep learning for document understanding in RAG systems.