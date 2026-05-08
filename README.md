# Disentangling Safety and Utility in LLM Activation Spaces — Team 9 (CS639)

Instruction-tuned LLMs encode refusal (safety) and helpfulness (utility)
as directions in their activation space. We test whether these
directions live in separable subspaces or share structure.

We extract each direction with two methods: Difference-of-Means (DoM)
and ActSVD. We then ablate the direction at a single transformer layer
using a forward hook. To check independence, we run a 2×2
cross-intervention matrix. The idea is simple: ablating one direction
should break only its own behavior.

Models tested: **Gemma-2-2B-IT** and **Llama-3.1-8B-Instruct**.

---

## Repo layout

```
.
├── src/                         shared library
│   ├── __init__.py
│   ├── common.py                model + activation + dataset loaders
│   ├── directions.py            DoM and ActSVD extraction
│   ├── ablation.py              forward-hook ablation
│   └── evaluation.py            ASR + utility benchmarks
├── scripts/
│   ├── run_eda.py               layer selection, δ comparison, stability
│   ├── extract_directions.py    extract DoM + ActSVD, save to cache/
│   ├── run_2x2_matrix.py        cross-intervention experiment
│   ├── run_sensitivity.py       layer + ActSVD-rank sweep
│   ├── plot_matrix.py           grouped bar chart
│   └── compute_subspace_angles.py  φ(U_s, U_u) + principal angles
├── hw3/                         HW3 deliverables (preserved as submitted)
├── results/<tag>/               EDA plots, JSONs, matrix plot
├── cache/<tag>/                 extracted directions
├── run_all.sh                   end-to-end driver
├── requirements.txt
├── pyproject.toml
└── README.md
```

`tag` is `gemma2-2b-it` or `llama3.1-8b-it`.

---

## Setup

Requires Python 3.10 or newer.

```bash
pip install -r requirements.txt
printf "HF_TOKEN=hf_YOUR_TOKEN\n" > .env
```

If `> .env` errors with "cannot overwrite existing file", your shell
has `noclobber` set. Use `>|` instead:

```bash
printf "HF_TOKEN=hf_YOUR_TOKEN\n" >| .env
```

GPU memory:
* Gemma-2-2B-IT (FP16) — ~12 GB
* Llama-3.1-8B-Instruct (FP16) — ~20 GB. An 11 GB card cannot fit
  Llama in FP16. Use an A40 or larger.

You also need to accept the gated-repo licenses on Hugging Face for:
- `google/gemma-2-2b-it`
- `meta-llama/Llama-3.1-8B-Instruct`
- `walledai/AdvBench`
- `nvidia/HelpSteer`

---

## Reproducing everything

```bash
bash run_all.sh                # both models, full pipeline
bash run_all.sh gemma_only     # only Gemma
bash run_all.sh llama_only     # only Llama
```

For the canonical results we use **Gemma layer 25** and **Llama layer
11**. Gemma's layer is the same one the EDA divergence-score
recommends, and is also the empirical best in the layer sweep. Llama's
layer is *not* the EDA recommendation: the divergence score peaks at
hidden_states index 32, which is the post-final-block residual stream
and not a hookable transformer block. The nearest hookable block (L=31)
produced no ablation effect, and a layer sweep covering L=13–28
similarly maxed out around 14% ASR. Extending the sweep to early
layers revealed L=11 as the true safety-causal layer, with DoM
ablation lifting ASR from 6% to 55%. We treat the EDA layer-selection
mismatch on larger models as itself a finding (see Section 5 of the
report).

`run_all.sh` defaults to `LLAMA_LAYER=11` for Llama. The sweep range in
`run_all.sh` is auto-clipped so it never exceeds the model's last block.

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
```

Repeat for Llama by swapping `--model` and `--tag` and passing
`--layer 11` (not 16). For Llama the sensitivity sweep that locates the
right layer needed to cover early layers, e.g.:

```bash
python scripts/run_sensitivity.py \
  --model meta-llama/Llama-3.1-8B-Instruct --tag llama3.1-8b-it \
  --layers 9,10,11,12 --ranks 4 --rank_layer 11
```

After the matrix completes, compute the principal angles between the
ActSVD safety and utility subspaces at the chosen layer:

```bash
python scripts/compute_subspace_angles.py --tag gemma2-2b-it
python scripts/compute_subspace_angles.py --tag llama3.1-8b-it
```

---

## Methods

### DoM (Difference-of-Means)

At a chosen layer L*, compute `r = mean(h_pos) - mean(h_neg)` and
normalize.
* Safety: positives are AdvBench prompts, negatives are Alpaca prompts.
  We use the activation at the last prompt token.
* Utility: positives are HelpSteer high-helpfulness responses,
  negatives are low-helpfulness responses. We average activations over
  the response tokens. Pairs are filtered to δ ≥ 3.

### ActSVD

Build the per-example difference matrix `D = H_pos - H_neg`. Take its
SVD. For safety ablation, keep the top-k right singular vectors as an
orthonormal basis
V (k × d).

For the current Llama utility pivot, `actsvd_utility` keeps the
bottom-k right singular vectors from the safety difference matrix
instead of extracting raw HelpSteer utility ActSVD ranks.

### Ablation

Both methods are applied via a forward hook on the chosen transformer
block (`model.model.layers[L]`):
* DoM — subtract the projection onto a single unit vector.
* ActSVD — subtract the projection onto the rank-k subspace.
  `h' = h − (h Vᵀ) V`.

### Cross-intervention matrix

For each cell, ablate the named direction. Then evaluate both ASR
(safety) and the utility benchmark suite plus TruthfulQA (control).

|              | ablate **safety** | ablate **utility** |
|--------------|-------------------|--------------------|
| **DoM**      | dom_safety        | dom_utility        |
| **ActSVD**   | actsvd_safety     | actsvd_utility     |

In the current pivot, `actsvd_utility` is intentionally implemented as
bottom safety-rank ablation: it loads `V_utility_actsvd.npy`, but that
file is generated from the least safety-relevant AdvBench-vs-Alpaca
ActSVD ranks.

Plus a `baseline` row (no ablation) and two random-direction control
rows. `random_direction` ablates a random unit vector at the same
layer; `random_subspace` ablates a random rank-4 orthonormal basis.
These controls test whether the ASR / utility movements observed in
the four real cells are direction-specific or just generic
ablation-noise artifacts. Both controls reproduce the baseline numbers
on both models.

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
following Arditi et al. (2024). ASR is the fraction of held-out
prompts where no refusal substring appears in the model's reply.
Higher means less safe.

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
subspace_angles.json            ActSVD subspace similarity φ(U_s, U_u)
                                and principal angles between safety
                                and utility subspaces

# Llama tag also has:
matrix_results_layer16.json     2x2 matrix from an earlier sweep that
                                bottomed out around L=16 (max ASR ≈
                                14%). Kept as evidence that
                                mid-network layers are not where the
                                Llama safety direction lives.
matrix_results_layer31.json     2x2 matrix at the layer the EDA
                                divergence score recommended
                                (hidden_states index 32, whose nearest
                                hookable transformer block is L=31).
                                Produced no measurable ablation
                                effect; kept as evidence that the EDA
                                recommendation is wrong on Llama.
sensitivity_layer_sweep_18to31.json    earlier mid/late-layer Llama
                                       sweep (L=18,20,22,25,28). Max
                                       ASR ≈ 14%.
sensitivity_layer_sweep_L13to17.json   earlier near-L16 Llama sweep.
                                       Max ASR ≈ 14%.

# Gemma tag also has:
matrix_results_no_random.json   snapshot of the Gemma matrix before
                                the random-direction control cells
                                were added. Kept for reproducibility
                                of the original 5-cell tables in the
                                proposal.
```

The current `matrix_results.json` for both tags contains all seven
cells (baseline, dom_safety, dom_utility, actsvd_safety,
actsvd_utility, random_direction, random_subspace).

`cache/<tag>/` contains:

```
r_safety_dom.npy            (hidden,)
r_utility_dom.npy           (hidden,)
V_safety_actsvd.npy         (rank, hidden)
V_utility_actsvd.npy        (rank, hidden), bottom safety ranks for
                            the current actsvd_utility pivot
eval_prompts.json           held-out AdvBench prompts
meta.json                   extraction config
sensitivity_acts/           per-layer activations cache. Auto-generated
                            on first sweep. Git-ignored because the
                            files exceed GitHub's 100 MB limit. The
                            script rebuilds them automatically.
```

---

## Relationship to HW3

The original HW3 deliverables live in `hw3/` and are kept unchanged
for reference. They include `eda.py`, `ablation_experiment.py`,
`exploratory_analysis.py`, `r_s_unit.npy`, `ablation_results.json`,
`asr_examples.json`, and the three `eda_*.png` plots. The HW5 pipeline
in `src/` and `scripts/` is the current code.

---

## Known caveats

* **Refusal detection is substring-based.** It can over-count politely-
  worded compliance (e.g. "I cannot help… [proceeds to help]"). An
  LLM-as-judge evaluator would tighten this.
* **ActSVD rank defaults to 4.** The rank sweep in
  `sensitivity_rank_sweep.json` justifies that choice.
* **Llama layer choice was empirical.** The EDA's divergence-score
  recommendation (hidden_states index 32) sits past the last hookable
  transformer block (Llama-3.1-8B has 32 blocks indexed 0-31). We
  tried L=31, L=18-28, and L=13-17 — all produced ASR ≤ 14% (~+8pp
  above baseline). Extending the sweep to early layers found L=11,
  where DoM ablation lifts ASR from 6% to 55% (+49pp). The EDA
  divergence score is biased toward late layers because activation
  norms grow with depth; for larger models this surfaces a layer that
  is causally inert. We treat this as a methodological finding and
  document it in Section 5 of the report.
* **Random-direction controls.** The matrix runner produces two
  control cells in addition to the four real ones. Both reproduce the
  baseline (Gemma 1%, Llama 6%) on every metric, ruling out "any
  ablation breaks safety" as an alternative explanation for the
  dom_safety and actsvd_safety jumps.
* **Same RNG seed per cell.** `run_2x2_matrix.py` resets the seed
  before each cell so all cells use the same utility-benchmark
  indices. Deltas across cells are not caused by sampling noise.
