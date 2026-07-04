## CWEMap layout

```
CWEMap/
├── main.py                                # single pipeline entrypoint
├── configs/
│   ├── default.yaml                       # validation-oriented default config
│   ├── primevul_training.yaml             # training split
│   ├── primevul_validation.yaml           # validation split
│   ├── primevul_testing.yaml              # held-out testing split
├── phases/
│   ├── phase1_patch_aware_retrieval.py    # Phase 1: CodeLlama-7B patch-aware retrieval
│   ├── phase2_patch_graph_construction.py # Phase 2: ChatGPT-4o triples + materialisation
│   ├── phase3_evidence_alignment.py       # Phase 3: VF2++ alignment + HNERP evidence scoring
│   └── phase4_hierarchical_reasoning.py   # Phase 4: DeepSeek-V4-Flash CWE path reasoning
└── outputs/                               # JSON/Excel/CSV 
```

## dataset split

The source code and configs now explicitly use the original split names:

| Split | Relative dataset name | Intended role |
|---|---|---|
| Training | `datasetname/datasetname_training.jsonl` | Historical retrieval corpus / reference examples |
| Validation | `datasetname/datasetname.jsonl` | Unseen validation queries for approach tuning |
| Testing | `datasetname/datasetname.jsonl` | Held-out final test queries |

The default full paths are:

```text
Dataset_Preprocessing/dataset/datasetname/dataset_training.jsonl
Dataset_Preprocessing/dataset/datasetname/dataset_testing.jsonl
Dataset_Preprocessing/dataset/datasetname/dataset_validation.jsonl
```


## Phase-to-phase data flow

The pipeline is wired as a strict chain:

```text
PrimeVul split JSONL files
  -> Phase 1 PVR output JSON
  -> Phase 2 triples/materialised patch knowledge JSON
  -> Phase 3 graph-aligned evidence JSON
  -> Phase 4 final CWE path prediction JSON/CSV/XLSX
```

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
