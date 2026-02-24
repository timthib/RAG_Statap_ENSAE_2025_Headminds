# Projet

Créer et implémenter sur un dataset une nouvelle méthode de RAG en innovant au moins sur la méthode de parsing. 

# Livrables

- Github documenté
- Rapport final + ppt

# Roadmap

1. Revue scientifique - 4sem 
    1. LLM, RAG, parsing embedding, reranking, generation
    2. Synthèse de l’état de l’art 
2. Construction d’un dataset - 2sem 
    1. opensource 
    2. différentes sources de données 
    3. différents types de données (images, pdf, xlsx) 
3. Définition de métriques d’évaluation - 2sem
    1. efficacité du RAG 
    2. évaluation de la méthode de parsing 
4. Expérimentation de RAGs - 4sem 
    1. sur le dataset créé 
    2. tester des méthodes de parsing 
    3. évaluation des performances par rapport aux métriques d’évaluation 
5. Développement d’un RAG - 4 à 6 sem 
    1. mise au point d’une méthode de parsing 
    2. évaluation des performances par rapport aux métriques d’évaluation 
    3. front avec chat bot ?

# Évaluation de Docling sur OmniDocBench

Ce dépôt contient un environnement d'évaluation autonome et léger pour évaluer les performances du modèle **Docling** sur la benchmark **OmniDocBench**. Il inclut les scripts d'évaluation, les configurations et les dépendances nécessaires pour lancer une validation de bout en bout.

## 📋 Table des matières

- [Prérequis](#prérequis)
- [Installation de l'environnement](#installation-de-lenvironnement)
- [Données nécessaires](#données-nécessaires)
- [Structure du projet](#structure-du-projet)
- [Configuration](#configuration)
- [Lancer l'évaluation](#lancer-lévaluation)
- [Visualisation des résultats](#visualisation-des-résultats)

---

## Prérequis

- **Python 3.10** ou **3.11**
- **pip** (gestionnaire de paquets standard)
- **Git**

---

## Installation de l'environnement

### 1. Cloner le dépôt

```bash
git clone https://github.com/timthib/RAG_Statap_ENSAE_2025_Headminds.git
cd RAG_Statap_ENSAE_2025_Headminds
```

### 2. Créer et activer un environnement virtuel

Il est recommandé d'utiliser un environnement virtuel pour isoler les dépendances.

```bash
# Créer l'environnement (exemple avec venv)
python -m venv .venv

# Activer l'environnement
# Sur Linux/Mac :
source .venv/bin/activate
# Sur Windows :
.venv\Scripts\activate
```

### 3. Installer les dépendances

Installez les paquets nécessaires listés dans `requirements.txt`. Cela inclut les outils de parsing, les métriques (TEDS, BLEU, etc.) et la logique de matching.

```bash
pip install -r requirements.txt
```

**Note :** Si votre modèle sort des tableaux spécifiquement au format **LaTeX** (nécessitant une conversion), vous pourriez avoir besoin d'installer [LaTeXML](https://math.nist.gov/~BMiller/LaTeXML/) séparément. Pour une sortie Markdown/HTML standard de Docling, les dépendances ci-dessus sont suffisantes.

---

## Données nécessaires

⚠️ **Important :** Les fichiers de données brutes (images, PDFs, annotations OmniDocBench et les fichiers markdown générés par Docling) ne sont pas inclus dans ce dépôt GitHub en raison de leur taille importante.

Pour faire fonctionner le script d'évaluation, vous devez télécharger ces données et les placer dans le dossier du projet.

### Structure attendue

Assurez-vous que votre dossier de travail ressemble à ceci avant de lancer l'évaluation. Les dossiers marqués d'un astérisque `*` sont ceux que vous devez créer ou remplir :

```text
RAG_Statap_ENSAE_2025_Headminds/
├── configs/
│   └── end2end.yaml            # Fichier de configuration (à vérifier)
├── dataset/                    # Code de chargement des données
├── images/ *                   # [À AJOUTER] Annotations de vérité terrain (GT)
├── metrics/                    # Code de calcul des métriques
├── pdfs/ *                     # [À AJOUTER] PDFs du benchmark
├── registry/
├── result/ *                   # Résultats de l'évaluation (JSONs)
│   └── docling/ *              # [À AJOUTER] Fichiers .md générés par Docling
│   └── docling_quick_match_...
│   └── docling_quick_match_...
│   └── ...
├── task/
├── tools/
│   └── generate_result_tables.ipynb  # Notebook pour visualiser les résultats
├── utils/
├── gitignore           
├── OmniDocBench.json           # [À AJOUTER] Annotations de vérité terrain (GT)
├── pdf_validation.py           # Script principal d'évaluation
├── requirements.txt
└── README.md
```

---

## Configuration

Avant de lancer l'évaluation, vous devez vérifier et potentiellement modifier les chemins dans le fichier `configs/end2end.yaml`.

La section critique se trouve sous `dataset` :

```yaml
dataset:
  dataset_name: end2end_dataset
  ground_truth:
    # Chemin vers le fichier JSON de vérité terrain (OmniDocBench.json)
    data_path: ./OmniDocBench.json
  prediction:
    # Chemin vers le dossier contenant les fichiers .md de Docling
    data_path: ./result/docling
  match_method: quick_match
```

*   **`ground_truth/data_path`** : Doit pointer vers votre fichier `OmniDocBench.json`.
*   **`prediction/data_path`** : Doit pointer vers le dossier contenant vos résultats de prédiction (les fichiers `.md`).

---

## Lancer l'évaluation

Une fois l'environnement activé et la configuration vérifiée, lancez la commande suivante à la racine du dépôt :

```bash
python pdf_validation.py --config configs/end2end.yaml
```

Une barre de progression (`tqdm`) apparaîtra pour indiquer le traitement des pages.

---

## Visualisation des résultats

Les fichiers JSON résultants de l'évaluation (contenant les scores Edit Distance, TEDS, CDM, etc.) sont déjà inclus dans ce dépôt GitHub.

Pour analyser et visualiser ces résultats sous forme de tableaux de bord :

1.  Assurez-vous d'avoir **Jupyter** installé (`pip install jupyter`) ou utilisez VS Code avec l'extension Jupyter.
2.  Ouvrez le notebook situé dans le dossier outils :
    ```bash
    jupyter notebook tools/generate_result_tables.ipynb
    ```
3.  Exécutez les cellules du notebook pour générer les tableaux de scores finaux.

# Documents

## Veille scientifique 
HF : https://huggingface.co/papers/trending  #replaced paperswithcode
Arxiv : https://arxiv.org
Google scholar

- Connaissance globale LLM https://datascientest.com/large-language-models-tout-savoir
- Détail sur le fonctionnement des Transformers https://arxiv.org/pdf/1706.03762
- Présentation RAG https://arxiv.org/pdf/2410.12837
- Discussion sur le chunking de documents https://medium.com/@alexisperrier/dans-le-rag-limportant-c-est-le-chunk-0d541b4f4bbe
- Cours sur l'utilisation de modèles https://huggingface.co/learn/llm-course/chapter1/1
- Prise en main LangChain https://docs.langchain.com/oss/python/langchain/overview


