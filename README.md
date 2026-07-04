# 🛠️ CWEMap Replication Package 


> **CWEMap: Graph-Guided Reasoning for Fine-Grained CWE Classification of Security Patches**

![CWEMap Approach Overview](https://github.com/CWEMap/CWEMap/blob/d5f697b026bbc702f28bb528c013e9ed04648087/CWEMap.png?raw=true)

The replication package supports the experiments reported in the manuscript, including dataset preprocessing, execution scripts, cross-backbone generalizability evaluation, and efficiency analysis.

## 🧭 1. Framework Overview

CWEMap follows a four-stage graph-guided workflow for fine-grained commit-level CWE classification: it retrieves relevant historical cases, constructs phase-aware patch graphs, aligns structurally compatible evidence, and decodes a valid CWE path over the frozen CWE taxonomy graph.


```text
🎯 [Target Vulnerability-Fixing Commit]
        │ 
        ▼
🔎 1. Patch-Aware Vulnerability Retrieval (PVR)
   └── Retrieves top-k patch-relevant historical vulnerability cases
       from a leakage-free training corpus
        │
        ▼
🧱 2. Phase-Aware Patch Graph Construction (PGC)
   └── Extracts security triples and materializes phase-aware graphs
       using T_before, T_delta, and T_after
        │
        ▼
🔗 3. Agent-Based Evidence Alignment (AEA)
   └── Verifies structurally compatible historical cases through
       constrained subgraph matching and resonance evidence scoring
        │
        ▼
🌳 4. Agent-Based Hierarchical Reasoning (AHR)
   └── Performs top-down CWE path prediction over the frozen
       CWE taxonomy graph G_CWE with confidence-guided refinement
        │
        ▼
🏁 [Predicted Terminal CWE Path]
```


## 📊 Datasets

We evaluate **CWEMap** on two public vulnerability benchmarks: **TREEVUL** and **PRIMEVUL**.

The dataset and preprocessing scripts are available in [`Dataset_Preprocessing/`](./Dataset_Preprocessing).

**TREEVUL** contains real-world security patches annotated with hierarchical CWE paths, making it suitable for evaluating fine-grained commit-level vulnerability type classification.

**PRIMEVUL** is a large-scale vulnerability benchmark collected from multiple open-source ecosystems. It covers 140 Common Weakness Enumeration (CWE) classes among vulnerability-labeled training instances; in CWEMap, CWE-path evaluation is conducted only on vulnerability-labeled samples with valid CWE annotations.

The datasets are publicly available through Google Drive below.

Download Dataset: [Click here to access the dataset](https://drive.google.com/drive/folders/1ZNNrLlSb7GIvuvNFKMNDHEvGxci6WppK?usp=sharing)

## 📂 Datasets Structure

```text
Datasets/
├── dataset/
│   ├── treevul/
│   │   ├── train_set.json
│   │   ├── validation_set.json
│   │   └── test_set.json
│   │
│   └── primevul/
│       ├── primevul_train.jsonl
│       ├── primevul_valid.jsonl
│       └── primevul_test.jsonl
```

## 🚀 Prerequisites & Environment Setup

It is recommended to use a virtual environment to manage dependencies and ensure reproducibility.

### System Requirements

- **Operating System:** Ubuntu 22.04 LTS
- **Python:** >= 3.9
- **Perl:** >= 5.12
- **Java:** 1.8

### Create Virtual Environment

Using `venv`:

```bash
python3 -m venv cwemap_env
source cwemap_env/bin/activate
```

CWEMap follows phase evaluation pipeline. Each phase takes the output of the previous phase as input and produces structured evidence for the next stage.

## 🔄 Detailed Workflow of CWEMap

📜 **Phase 1: Patch-Aware Vulnerability Retrieval (PVR):**

- `PatchAwareVulnerabilityRetriever` retrieves top-k historical vulnerability cases from the training-only corpus.
- The `E_top-k`, the top-k retrieved historical cases, are passed to `PhaseAwarePatchGraphConstructor`.

📜 **Phase 2: Phase-Aware Patch Graph Construction (PGC):**

- `PhaseAwarePatchGraphConstructor` transforms the target patch and retrieved cases into phase-aware security triples (`T_before`, `T∆`, and `T_after`).
- `PatchGraphBuilder` materializes these triples into `G_input` for the target patch and `K_Gexamples` for retrieved cases.
- The graph workspace `{G_input, K_Gexamples, G_CWE}` is passed to `EvidenceAlignmentAgent`.

📜 **Phase 3: Agent-Based Evidence Alignment (AEA):**

- `EvidenceAlignmentAgent` verifies whether retrieved reference graphs are structurally compatible with the target patch graph.
- It performs subgraph isomorphism matching using relation and phase compatibility checks.
- `ResonanceEvidenceScoring` ranks the matched subgraphs using Hierarchical Node-Edge Resonance Proximity.
- The aligned evidence package `Z` is passed to `HierarchicalReasoningAgent`.

📜 **Phase 4: Agent-Based Hierarchical Reasoning (AHR):**

- `HierarchicalReasoningAgent` decodes the CWE path over the frozen taxonomy graph `G_CWE`.
- The final high-confidence predicted CWE path $\hat{P}$ is saved for evaluation using Weighted F1, Macro F1, MCC, and Path Fraction.



## Phase-Aware Security Triple Extraction Prompt

**Role:**  
You are an expert software security analyst and code-understanding engine.

**Task:**  
Analyze the provided code diff/commit to extract directed knowledge triples describing the underlying security vulnerability and its corresponding repair transformation.

To prevent semantic shortcut learning and token-level noise, the extraction must be strictly mapped to a rigid three-phase temporal schema:

- `T_before`: vulnerable state before the repair
- `T_delta`: security-relevant repair transformation
- `T_after`: corrected or safer state after the repair

The extraction should not rely on generic API-call relationships. Instead, each predicate/relation must represent an abstract security-critical invariant.

**Output Format:**  
Output each extracted triple on a new line, grouped strictly under the corresponding temporal phase comments:

```text
# T_before
(Head, Relation, Tail)

# T_delta
(Head, Relation, Tail)

# T_after
(Head, Relation, Tail)
```

Each line inside a phase must be formatted exactly as a three-variable graph tuple:

```text
(Head, Relation, Tail)
```

Do not output conversational prose, explanations, markdown wrappers, or any additional text outside the triples.


## ⚙️ Dependencies

Install the required dependencies using pip:

```bash
pip install -r requirements.txt
```

## 🔑 LLM API Keys

Set the required API key according to the selected backbone:

```bash
export DEEPSEEK_API_KEY="your-api-key-here"
export OPENAI_API_KEY="your-api-key-here"
export GEMINI_API_KEY="your-api-key-here"
```
## 🖥️ Hardware Requirements

A GPU is recommended for faster retrieval, graph construction, and LLM-assisted reasoning.

- **Recommended GPU:** NVIDIA GeForce RTX 3090
- **Minimum RAM:** 16 GB
- **Recommended OS:** Ubuntu 22.04 LTS
