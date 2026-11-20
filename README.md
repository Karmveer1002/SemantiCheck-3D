<div align="center">

# 💠 SemantiCheck 3D
### Deep Semantic Text Similarity & Duplicate Detection Studio

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Sentence-BERT](https://img.shields.io/badge/Sentence--BERT-all--MiniLM--L6--v2-FF6F00?style=for-the-badge&logo=huggingface&logoColor=white)](https://www.sbert.net/)
[![Three.js](https://img.shields.io/badge/Three.js-WebGL%203D-000000?style=for-the-badge&logo=threedotjs&logoColor=white)](https://threejs.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)

*Bridging the gap between surface-level keyword matching and transformer-based contextual understanding across raw text and multi-format documents (`.txt`, `.docx`, `.pdf`).*

**Developed by:** Karmveer Singh Gehlot 

</div>

---

## 📌 Overview & Problem Statement

Traditional duplicate-detection systems rely on exact string or keyword overlap. When two documents convey the exact same meaning using different vocabulary—such as synonyms, paraphrasing, or restructured sentences—lexical algorithms fail completely:

> **Example:**  
> *Text A:* *"How do I learn Python programming quickly?"*  
> *Text B:* *"What is the fastest way to master Python?"*  
> Even though both questions carry identical user intent, a standard keyword matcher scores them near zero.

**SemantiCheck 3D** solves this lexical-semantic gap using a **dual-engine NLP architecture** paired with a sliding-window document chunker and an interactive **60fps Three.js WebGL 3D studio interface**.

---

## ✨ Key Features

* **🧠 Dual-Engine Similarity Analysis:** Simultaneously computes **Deep Semantic Meaning** (`Sentence-BERT`) alongside **Exact Wording Overlap** (`TF-IDF + Cosine Similarity`) and **Vocabulary Overlap** (`Jaccard Index`).
* **📄 Universal Document Ingestion:** Compare pasted text snippets or upload multi-page **Notepad (`.txt`)**, **Microsoft Word (`.docx`)**, and **Adobe PDF (`.pdf`)** files seamlessly.
* **🪟 Sliding-Window Document Chunking:** Automatically splits long documents into **350-word overlapping chunks (50-word overlap)** and computes bi-directional max-chunk similarity so multi-page files are analyzed without transformer token truncation.
* **🌌 Interactive 3D WebGL Studio:** Built with **Three.js** and **FastAPI**, featuring a 2,600-particle 3D sculpture that dynamically morphs between a *Cube Mesh*, *Code Brackets (`</>`) *, *DNA Helix*, and *Neural Star* with mouse-parallax physics.
* **📊 Reproducible Benchmarking Pipeline:** Includes end-to-end data cleaning, Exploratory Data Analysis (EDA), and model evaluation (`Accuracy`, `Precision`, `Recall`, `F1-Score`) on the **Quora Question Pairs** dataset.

---

## 🔄 System Architecture & Process Flowchart

```mermaid
flowchart TD
    A["📥 Input Ingestion<br/>Source A & Target B (Raw Text / .TXT / .DOCX / .PDF)"] --> B["📄 Universal File Parser<br/>Multi-encoding TXT, python-docx, pypdf"]
    B --> C["🪟 Sliding-Window Chunker<br/>350-word chunks with 50-word overlap"]

    C --> D1["🧹 Lexical Preprocessing<br/>Lowercase, Regex, NLTK Stopwords, Lemmatization"]
    C --> D2["🔤 Contextual Preservation<br/>Full grammar & punctuation retained"]

    D1 --> E1["📊 TF-IDF Vectorizer<br/>Sublinear TF, (1,2) n-grams, 8k features"]
    D2 --> E2["🤖 Sentence-BERT Encoder<br/>all-MiniLM-L6-v2 → 384-d unit vectors"]

    E1 --> F1["📐 Sparse Cosine & Jaccard Overlap"]
    E2 --> F2["📐 Bi-Directional Max Chunk Cosine Alignment"]

    F1 --> G{"🎯 Threshold Classifier"}
    F2 --> G

    G -->|"Score ≥ 0.75"| H1["🚨 DUPLICATE<br/>Same core intent & meaning"]
    G -->|"0.45 – 0.74"| H2["⚠️ SIMILAR<br/>Shared themes, differing details"]
    G -->|"Score < 0.45"| H3["✅ DIFFERENT<br/>Unrelated topics"]

    H1 & H2 & H3 --> I["🌐 FastAPI JSON Response → Three.js 3D Web Studio"]
