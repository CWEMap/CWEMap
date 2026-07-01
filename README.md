# 🛠️ CWEMap: Graph-Guided Reasoning for Fine-Grained CWE Classification of Security Patches

This repository contains the replication package for **CWEMap**, the framework proposed in our ICSE submission:

> **CWEMap: Graph-Guided Reasoning for Fine-Grained CWE Classification of Security Patches**

![CWEMap Approach Overview](https://github.com/CWEMap/CWEMap/blob/d5f697b026bbc702f28bb528c013e9ed04648087/CWEMap.png?raw=true)

CWEMap addresses fine-grained commit-level Common Weakness Enumeration (CWE) classification by combining patch-evolution evidence, structurally similar historical vulnerability cases, and taxonomy-aware hierarchical reasoning. Instead of directly classifying raw patch text, CWEMap abstracts vulnerability-fixing commits into phase-aware patch graphs, aligns them with retrieved historical vulnerability cases, and performs constrained top-down prediction over the CWE taxonomy.

The replication package supports the experiments reported in the ICSE manuscript, including effectiveness comparison, ablation analysis, cross-backbone generalizability, and efficiency analysis.

---

## 🧭 1. Framework Overview

Given a vulnerability-fixing commit, CWEMap operates across four core stages:

### 🔍 Phase 1: Patch-Aware Vulnerability Retrieval (PVR)
Retrieves patch-relevant historical vulnerability cases from the training corpus while strictly preventing train-test data leakage.

### 🌿 Phase 2: Phase-Aware Patch Graph Construction (PGC)
Converts each raw code patch into phase-aware security triples:
* `T_before`: Vulnerable pre-patch state
* `T_delta`: Repair transformation
* `T_after`: Mitigated post-patch state

### 🤝 Phase 3: Agent-Based Evidence Alignment (AEA)
Aligns target patch graphs with retrieved reference graphs using constrained subgraph matching and multi-dimensional evidence scoring.

### 🧠 Phase 4: Agent-Based Hierarchical Reasoning (AHR)
Performs taxonomy-constrained CWE path prediction and confidence-guided refinement over the hierarchical CWE structure.

> **Key Innovation:** Unlike traditional approaches that rely solely on lexical patch similarity or unconstrained direct LLM prompting, CWEMap grounds its CWE predictions in explicit, graph-mapped patch semantics and valid structural taxonomy paths.

---

## 📂 2. Repository Structure

```text
CWEMap/
├── README.md               # Setup and replication instructions
├── requirements.txt       # Python dependencies (pip)
├── environment.yml        # Anaconda environment specification
├── configs/               # Hyperparameter and configuration files
│   ├── default.yaml
│   ├── treevul.yaml
│   ├── primevul.yaml
│   └── llm_backbones.yaml
├── data/                  # Datasets and taxonomies
│   ├── raw/               # Raw benchmarks (TreeVul, PrimeVul)
│   │   ├── treevul/
│   │   └── primevul/
│   ├── processed/         # Tokenized, parsed, and graph-ready inputs
│   │   ├── treevul/
│   │   └── primevul/
│   └── cwe/
│       └── cwe_taxonomy.json
├── src/                   # Core implementation codebase
│   ├── retrieval/         # Code for PVR phase
│   ├── graph_construction/# Code for PGC phase
│   ├── evidence_alignment/# Code for AEA phase
│   ├── reasoning/         # Code for AHR phase
│   ├── evaluation/        # Evaluation metric calculation
│   └── utils/             # Helper utilities and loggers
├── scripts/               # Single-task execution scripts
│   ├── preprocess_data.py
│   ├── build_retrieval_index.py
│   ├── run_cwemap.py
│   ├── run_baselines.py
│   ├── run_ablation.py
│   ├── run_backbone_study.py
│   ├── run_efficiency.py
│   └── aggregate_results.py
├── outputs/               # Evaluation artifacts
│   ├── logs/              # Runtime execution logs
│   ├── predictions/       # Model output inference JSONs
│   ├── metrics/           # Calculated precision, recall, F1 scores
│   ├── tables/            # LaTeX/CSV tables generated for the paper
│   └── figures/           # Plot diagrams (PDF/PNG format)
└── replication/           # One-click reproduction workflows
    ├── run_all.sh         # Complete pipeline execution
    ├── reproduce_rq1.sh   # Main effectiveness comparison
    ├── reproduce_rq2.sh   # Ablation study variants
    ├── reproduce_rq3.sh   # Cross-backbone evaluation
    └── reproduce_rq4.sh   # Execution time & resource footprint analysis
