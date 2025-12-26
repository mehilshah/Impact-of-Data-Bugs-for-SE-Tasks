# Understanding the Impact of Data Bugs in Deep Learning for Software Engineering

**Empirical Software Engineering (EMSE), 2025**

This repository contains the **artifacts, datasets, and analysis code** for our EMSE 2025 paper investigating the **impact of data quality issues (data bugs) on deep learning models used in software engineering tasks**.

📄 **Published paper (EMSE 2025):**
[https://link.springer.com/article/10.1007/s10664-025-10717-y](https://link.springer.com/article/10.1007/s10664-025-10717-y)

📄 **Preprint (arXiv):**
[https://arxiv.org/pdf/2411.12137](https://arxiv.org/pdf/2411.12137)

---

## 📌 Overview

Deep learning (DL) techniques have achieved remarkable success in software engineering tasks such as vulnerability detection, just-in-time defect prediction, and code analysis. However, DL systems are **highly sensitive to training data**, and bugs in datasets—such as noise, inconsistencies, missing preprocessing, and label issues—are both **common and poorly understood** in the software engineering domain.

This project presents a **comprehensive empirical study** on the **impact and symptoms of data bugs** across three data types widely used in software engineering:

* **Code-based data**
* **Text-based data**
* **Metric-based data**

Using **state-of-the-art deep learning models**, we compare:

* Models trained on **clean, well-preprocessed datasets**
* Models trained on datasets with **data quality issues and inadequate preprocessing**

By analysing **gradients, weights, and biases** during training, we identify **distinct failure symptoms** caused by data bugs. Our findings show that:

* **Code data issues** lead to *biased learning* and *gradient instability*
* **Text data issues** cause *overfitting* and *poor generalization*
* **Metric data issues** result in *exploding gradients* and *severe overfitting*
* **Inadequate preprocessing** amplifies these problems across all data types

We further validate the **generalizability** of our findings using **six additional datasets**, demonstrating that the observed symptoms are consistent across tasks and models.

---

## 🔬 Research Questions & Repository Structure

Experiments are organized around **four research questions (RQs)**, each targeting different data types and learning scenarios commonly found in software engineering.

```
.
├── RQ1   # Code-based data
├── RQ2   # Text-based data
├── RQ3   # Metric-based data
├── RQ4   # Cross-model statistical analysis
├── Analysis   # Aggregated analysis per RQ
└── Discussion # DataGuardian framework and discussion artifacts
```

---

### RQ1 – Code-Based Data

**Goal:** Understand how data quality issues in source code affect DL models.

* **Models**

  * CodeBERT
  * LineVul
* **Key Focus**

  * Gradient instability
  * Biased learning due to noisy or malformed code
* **Location**

  ```
  RQ1/
    ├── CodeBERT/
    └── LineVul/
  ```

---

### RQ2 – Text-Based Data

**Goal:** Analyze the impact of data bugs in natural language artifacts (e.g., commit messages, issue descriptions).

* **Models**

  * BERT-MLP
  * DCCNN
* **Key Focus**

  * Overfitting
  * Poor generalization caused by noisy or unprocessed text
* **Location**

  ```
  RQ2/
    ├── BERT-MLP/
    └── DCCNN/
  ```

---

### RQ3 – Metric-Based Data

**Goal:** Study how numerical and metric-based data quality issues affect DL training dynamics.

* **Models**

  * CodeBERTJIT
  * DeepJIT
* **Key Focus**

  * Exploding gradients
  * Model overfitting due to inconsistent or unnormalized metrics
* **Location**

  ```
  RQ3/
    ├── CodeBERTJIT/
    └── DeepJIT/
  ```

---

### RQ4 – Statistical Validation

**Goal:** Statistically validate the impact of data bugs across models and datasets.

* **Methods**

  * Non-parametric statistical tests
  * Effect size analysis
* **Location**

  ```
  RQ4/
    └── StatisticalTest_RQ4.ipynb
  ```

---

## 📊 Analysis Artifacts

The `Analysis/` directory contains scripts for:

* Aggregating experimental results
* Inspecting gradients, weights, and biases
* Reproducing tables and figures from the paper

```
Analysis/
├── RQ1/
├── RQ2/
├── RQ3/
└── RQ4/
```

---

## 🛡️ Discussion: DataGuardian

The `Discussion/` directory includes **DataGuardian**, a framework proposed in the paper to help **detect symptoms of data bugs** by monitoring model training behavior.

```
Discussion/
└── dataguardian/
```

Details are provided in:

```
Discussion/README.md
```

---

## ⚙️ Running the Experiments

Each model directory includes a **self-contained README** describing:

* Environment setup
* Dataset preparation
* Training and evaluation commands
* Notes on introduced data quality issues and preprocessing variants

➡️ **Please refer to the README inside each model directory** for execution details.

---

## 📁 Artifact Integrity

* Original implementations are preserved as much as possible
* Any necessary modifications are documented
* All experiments correspond to the configurations reported in the EMSE paper

---

## 📖 Citation

If you use this repository or build upon this work, please cite:

```bibtex
@article{shah2025towards,
  title={Towards understanding the impact of data bugs on deep learning models in software engineering},
  author={Shah, Mehil B and Rahman, Mohammad Masudur and Khomh, Foutse},
  journal={Empirical Software Engineering},
  volume={30},
  number={6},
  pages={168},
  year={2025},
  publisher={Springer}
}
```

---

## ⭐ Impact

This work provides actionable insights for:

* **Researchers** studying DL robustness in SE
* **Practitioners** building DL-based SE tools
* **Tool builders** designing data monitoring and cleaning pipelines