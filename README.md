# Disentangling Safety and Utility in LLM Activation Spaces — Team 9 (CS639)

We test whether the safety (refusal) and utility (helpfulness) directions
extracted from an instruction-tuned LLM's residual stream occupy
*separable* subspaces. Each direction is extracted with two methods —
Difference-of-Means (**DoM**) and **ActSVD** — then ablated at a single
transformer layer via a forward hook. Independence is measured by a 2×2
cross-intervention matrix: ablating one direction should break only its
own behavior and leave the other intact.

Models studied: **Gemma-2-2B-IT** and **Llama-3.1-8B-Instruct**.

---

## Repo layout

```
.
├── src/                         shared library
│   ├── __init__.py
│   ├── common.py                model + activation + dataset loaders
│   ├── directions.py            DoM and ActSVD extraction
│   ├── ablation.py              forward-hook ablation (vector + subspace)
│   └── evaluation.py            ASR + BoolQ/HellaSwag/ARC-C/TruthfulQA
├── scripts/
│   ├── run_eda.py               layer selection, δ comparison, stability
│   ├── extract_directions.py    extract DoM + ActSVD, save to cache/
│   ├── run_2x2_matrix.py        cross-intervention experiment
│   ├── run_sensitivity.py       layer + ActSVD-rank sweep
│   └── plot_matrix.py           grouped bar chart for the report
├── hw3/                         HW3 deliverables (preserved as submitted)
│   ├── ablation_experiment.py
│   ├── exploratory_analysis.py
│   ├── eda.py
│   ├── ablation_results.json
│   ├── asr_examples.json
│   ├── eda_*.png
│   └── r_s_unit.npy
├── results/<tag>/               EDA plots, JSONs, matrix plot
├── cache/<tag>/                 extracted directions + cached activations
├── run_all.sh                   end-to-end reproduction
├── requirements.txt
├── pyproject.toml
└── README.md
```

`tag` is `gemma2-2b-it` or `llama3.1-8b-it`.

---

## Setup

```bash
pip install -r requirements.txt
echo "HF_TOKEN=hf_..." > .env
```

GPU memory:
* Gemma-2-2B-IT (FP16) — ~12 GB
* Llama-3.1-8B-Instruct (FP16) — ~20 GB (use an A40 or similar; an
  11 GB card cannot fit the model in FP16)

You also need to accept the gated-repo licenses on Hugging Face for
`google/gemma-2-2b-it`, `meta-llama/Llama-3.1-8B-Instruct`,
`walledai/AdvBench`, and `nvidia/HelpSteer`.

---

## Reproducing everything

```bash
bash run_all.sh                # both models, full pipeline
bash run_all.sh gemma_only     # only Gemma
bash run_all.sh llama_only     # only Llama
```

`run_all.sh` uses Gemma layer 25 and Llama layer 16 (the empirically
best layer found via the sensitivity sweep). The sensitivity sweep
range is automatically clipped so it never exceeds the model's last
transformer block.

### Step-by-step

```bash
# 1) EDA — picks an extraction layer L*
python scripts/run_eda.py --model google/gemma-2-2b-it --tag gemma2-2b-it
# inspect:  results/gemma2-2b-it/eda_summary.json   -> "recommended_layer"

# 2) Cache safety + utility directions at L*
python scripts/extract_directions.py \
  --model google/gemma-2-2b-it --tag gemma2-2b-it --layer 25

# 3) Run the 2x2 cross-intervention matrix
python scripts/run_2x2_matrix.py \
  --model google/gemma-2-2b-it --tag gemma2-2b-it --layer 25
python scripts/plot_matrix.py --tag gemma2-2b-it

# 4) Sensitivity (layers around L*; ActSVD ranks at L*)
python scripts/run_sensitivity.py \
  --model google/gemma-2-2b-it --tag gemma2-2b-it \
  --layers 22,23,24,25 --ranks 1,2,4,8,16 --rank_layer 25

# Repeat for Llama, swapping --model and --tag.
```

---

## Methods

### DoM (Difference-of-Means)

At a chosen layer L*, compute
`r = mean(h_pos) - mean(h_neg)`, then normalize.
* safety: positives = AdvBench prompts, negatives = Alpaca prompts
  (last-token activations)
* utility: positives = HelpSteer high-helpfulness responses, negatives =
  HelpSteer low-helpfulness responses (mean over response tokens),
  filtered to δ ≥ 3

### ActSVD

Build the per-example difference matrix `D = H_pos - H_neg`, take its
SVD, and keep the top-k right singular vectors as an orthonormal basis
V (k × d).

### Ablation

Both methods are applied via a forward hook on the chosen
transformer block (`model.model.layers[L]`):
* DoM — subtract the projection onto a single unit vector
* ActSVD — subtract the projection onto the rank-k subspace
  `h' = h − (h Vᵀ) V`

### Cross-intervention matrix

For each cell, ablate the named direction and evaluate **both** ASR
(safety) and the utility benchmark suite + TruthfulQA (control).

|              | ablate **safety** | ablate **utility** |
|--------------|-------------------|--------------------|
| **DoM**      | dom_safety        | dom_utility        |
| **ActSVD**   | actsvd_safety     | actsvd_utility     |

Plus a `baseline` row (no ablation).

---

## Datasets

| use                | dataset                          |
|--------------------|----------------------------------|
| harmful prompts    | `walledai/AdvBench`              |
| harmless prompts   | `tatsu-lab/alpaca`               |
| helpfulness pairs  | `nvidia/HelpSteer`               |
| ASR eval           | held-out AdvBench (100 prompts)  |
| utility — yes/no   | `google/boolq`                   |
| utility — common.  | `Rowan/hellaswag`                |
| utility — sci      | `allenai/ai2_arc` (Challenge)    |
| control            | `truthfulqa/truthful_qa` (MC1)   |

---

## Key hyperparameters (defaults)

| flag             | default | meaning                                   |
|------------------|---------|-------------------------------------------|
| `--n_extract`    | 256     | N for direction extraction (per pool)     |
| `--n_eval`       | 100     | held-out AdvBench prompts for ASR         |
| `--n_utility`    | 200     | samples per utility benchmark             |
| `--rank`         | 4       | ActSVD subspace dimensionality            |
| `--min_delta`    | 3       | HelpSteer δ threshold for utility pairs   |
| `--seed`         | 42      | numpy + torch RNG                         |

---

## ASR scoring

Refusal detection uses substring matching against the list in
[src/evaluation.py](src/evaluation.py) (`REFUSAL_SUBSTRINGS`),
following Arditi et al. (2024). ASR = fraction of held-out prompts
where **no** refusal substring appears in the model's reply
(higher = less safe).

---

## Outputs

After a full run for a tag, `results/<tag>/` contains:

```
eda_layer_selection.png         per-layer ||r_s||, ||r_u||, |cos|, divergence
eda_helpsteer_delta.png         δ threshold comparison + δ=4 PCA
eda_stability.png               bootstrap stability vs N
eda_summary.json                recommended layer, key numbers
matrix_results.json             full 2x2 cross-intervention numbers
matrix_plot.png                 grouped bar chart of the matrix
sensitivity_layer_sweep.json    ASR + utility per extraction layer
sensitivity_rank_sweep.json     ASR + utility per ActSVD rank
```

`cache/<tag>/` contains:

```
r_safety_dom.npy            (hidden,)
r_utility_dom.npy           (hidden,)
V_safety_actsvd.npy         (rank, hidden)
V_utility_actsvd.npy        (rank, hidden)
eval_prompts.json           held-out AdvBench prompts
meta.json                   extraction config
sensitivity_acts/           per-layer activations cache (auto-generated on first sweep,
                            git-ignored because the files exceed GitHub's 100 MB limit;
                            run_sensitivity.py rebuilds them automatically)
```

---

## Relationship to HW3

The original HW3 deliverables — `eda.py`, `ablation_experiment.py`,
`exploratory_analysis.py`, `r_s_unit.npy`, `ablation_results.json`,
`asr_examples.json`, and the three `eda_*.png` plots — live in `hw3/`
unchanged. The HW5 pipeline supersedes them but the legacy scripts
still run end-to-end against `cache/gemma2-2b-it/r_safety_dom.npy`
(equivalent to `r_s_unit.npy`).

---

## Known caveats

* **Refusal detection is substring-based**, so it over-counts politely-
  worded compliance ("I cannot help… [proceeds to help]"). LLM-as-judge
  scoring would tighten this.
* **ActSVD rank** defaults to 4; the rank sweep is what justifies that
  choice — see `sensitivity_rank_sweep.json`.
* **Llama layer choice** was determined empirically via an 11-layer
  sensitivity sweep. The EDA's divergence-score recommendation
  (layer 32) does not correspond to a hookable transformer block and
  did not produce a working ablation when tested at layer 31. The
  empirical best layer (16) still produces only a weak ablation effect,
  which we report as a negative finding.
* **Same RNG seed per cell** in `run_2x2_matrix.py` ensures every cell
  hits the same utility-benchmark indices, so deltas across cells are
  not dominated by sampling noise.
