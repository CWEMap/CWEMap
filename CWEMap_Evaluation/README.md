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

## dataset split names

The source code and configs now explicitly use the original split names:

| Split | Relative dataset name | Intended role |
|---|---|---|
| Training | `datasetname/datasetname_training.jsonl` | Historical retrieval corpus / reference examples |
| Validation | `datasetname/datasetname.jsonl` | Unseen validation queries for approach tuning |
| Testing | `datasetname/datasetname.jsonl` | Held-out final test queries |

The default full paths are:

```text
Dataset_Preprocessing/dataset/datasetname/datasetname.jsonl
Dataset_Preprocessing/dataset/datasetname/datasetname.jsonl
Dataset_Preprocessing/dataset/datasetname/datasetname.jsonl
```

Recommended evaluation protocol:

```text
Validation run: input = datasetname.jsonl, corpus = datasetname.jsonl
Testing run:    input = datasetname.jsonl,    corpus = datasetname.jsonl
```

This avoids retrieval leakage because validation/testing samples are never used as the retrieval corpus.


## Phase-to-phase data flow

The pipeline is wired as a strict chain. In `main.py`, each phase returns an in-memory JSON object and the next phase consumes that object directly:

```text
PrimeVul split JSONL files
  -> Phase 1 PVR output JSON
  -> Phase 2 triples/materialised patch knowledge JSON
  -> Phase 3 graph-aligned evidence JSON
  -> Phase 4 final CWE path prediction JSON/CSV/XLSX
```

The same chain is also written explicitly in the configs for standalone/partial reruns:

| Config key | Meaning |
|---|---|
| `phase1.input` | query split: `datasetname_training.jsonl`, `datasetname_validation.jsonl`, or `datasetname_testing.jsonl` |
| `phase1.corpus` | retrieval corpus, normally `datasetname_training.jsonl` |
| `phase2.phase1_json` | saved Phase 1 output consumed by Phase 2 |
| `phase3.phase2_json` | saved Phase 2 output consumed by Phase 3 |
| `phase4.phase3_json` | saved Phase 3 output consumed by Phase 4 |

