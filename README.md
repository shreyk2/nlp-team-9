# Disentangling Safety and Utility in LLM Activation Spaces

We studied whether instruction-following LLMs represent safety/refusal behavior and utility/helpfulness behavior in separable activation-space directions.

We compare two direction-extraction methods, DoM and ActSVD, then ablate those directions at a chosen transformer layer using forward hooks. The main question is simple: if we remove a safety direction, does harmful behavior increase without materially hurting utility, and vice versa?

Models used in the current experiments:

- Gemma-2-2B-IT
- Llama-3.1-8B-Instruct

## What This Repo Does

The pipeline has four stages:

1. Explore layer-wise activation geometry to choose a candidate extraction layer.
2. Extract safety and utility directions at that layer.
3. Ablate the extracted directions and evaluate both safety and utility benchmarks.
4. Sweep layers, ranks, and subspace overlap to test how robust the result is.

The code is organized so the reusable logic lives in [src/](src/), while the experiment entry points live in [scripts/](scripts/).

## Repository Layout

```
.
├── src/
│   ├── common.py              model loading, activations, dataset helpers
│   ├── directions.py          DoM, ActSVD, projection utilities
│   ├── ablation.py            forward-hook ablation classes
│   └── evaluation.py          ASR and utility benchmark evaluation
├── scripts/
│   ├── run_eda.py             layer selection and stability analysis
│   ├── extract_directions.py  save DoM and ActSVD directions to cache/
│   ├── run_2x2_matrix.py      cross-intervention matrix experiment
│   ├── plot_matrix.py         bar chart for matrix results
│   ├── run_sensitivity.py     layer and rank sweeps
│   └── compute_subspace_angles.py  principal-angle analysis
├── cache/                     extracted directions and prompt splits
├── results/                   plots and JSON summaries
├── run_all.sh                 end-to-end driver for both models
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Main Idea

At a chosen layer $L$, the repo builds two kinds of vectors or subspaces:

- a safety direction, from harmful prompts vs harmless prompts
- a utility direction, from high-helpfulness vs low-helpfulness responses

It then measures what happens when those directions are removed from the hidden states of the model.

If the directions are meaningfully disentangled, then ablation should mostly affect the behavior associated with the ablated direction.

## Methods

### Difference of Means (DoM)

For a positive set and a negative set, DoM computes:

$$
r = \frac{1}{|P|} \sum_{x \in P} h(x) - \frac{1}{|N|} \sum_{x \in N} h(x)
$$

The resulting vector is normalized and used as a one-dimensional direction.

In this project:

- safety DoM: AdvBench prompts vs Alpaca prompts
- utility DoM: high-helpfulness HelpSteer responses vs low-helpfulness HelpSteer responses

For prompt activations, the code uses the last prompt token. For response activations, it averages hidden states over the response tokens only.

### ActSVD

ActSVD builds a matrix of paired activation differences and takes its SVD. The top-$k$ right singular vectors form an orthonormal basis for a subspace.

For ablation, the hidden state is projected onto that basis and the projection is removed:

$$
h' = h - (hV^T)V
$$

where $V$ is the basis matrix.

### Ablation Hooks

The actual intervention happens in [src/ablation.py](src/ablation.py) using a forward hook on a transformer block. That keeps the rest of the model unchanged while zeroing out the chosen direction or subspace.

## Evaluation

The repo evaluates two broad outcomes:

- Safety: Attack Success Rate, measured on held-out AdvBench prompts
- Utility: BoolQ, HellaSwag, ARC-Challenge, and TruthfulQA MC1

ASR is computed with substring-based refusal detection in [src/evaluation.py](src/evaluation.py). Lower ASR means the model is refusing more often and is therefore safer by this metric.

## Datasets

The current pipeline uses these datasets:

- `walledai/AdvBench` for harmful prompts and held-out ASR evaluation
- `tatsu-lab/alpaca` for harmless prompts
- `nvidia/HelpSteer` for utility pairs
- `google/boolq` for yes/no reading comprehension
- `Rowan/hellaswag` for commonsense completion
- `allenai/ai2_arc` for science questions
- `truthfulqa/truthful_qa` for control evaluation

## Setup

Requires Python 3.10 or newer.

```bash
pip install -r requirements.txt
printf "HF_TOKEN=hf_YOUR_TOKEN\n" > .env
```

If your shell has `noclobber` enabled and the `.env` write fails, use:

```bash
printf "HF_TOKEN=hf_YOUR_TOKEN\n" >| .env
```

You also need to accept the Hugging Face licenses for the gated datasets and models used here, including:

- `google/gemma-2-2b-it`
- `meta-llama/Llama-3.1-8B-Instruct`
- `walledai/AdvBench`
- `nvidia/HelpSteer`

### Hardware Notes

- Gemma-2-2B-IT in FP16 is roughly a 12 GB model
- Llama-3.1-8B-Instruct in FP16 is roughly a 20 GB model
- An 11 GB GPU is not enough for Llama in FP16

## Reproducing the Full Pipeline

Run everything:

```bash
bash run_all.sh
```

Run only one model:

```bash
bash run_all.sh gemma_only
bash run_all.sh llama_only
```

The driver script runs EDA, direction extraction, the 2×2 matrix, plotting, and sensitivity analysis.

## Step-by-Step Workflow

### 1. Run EDA

EDA estimates which layer is most promising for extraction.

```bash
python scripts/run_eda.py --model google/gemma-2-2b-it --tag gemma2-2b-it
```

This produces `results/<tag>/eda_summary.json`, which includes the recommended layer.

### 2. Extract directions

```bash
python scripts/extract_directions.py \
  --model google/gemma-2-2b-it \
  --tag gemma2-2b-it \
  --layer 25
```

This saves DoM and ActSVD directions to `cache/<tag>/`.

### 3. Run the 2×2 matrix

```bash
python scripts/run_2x2_matrix.py \
  --model google/gemma-2-2b-it \
  --tag gemma2-2b-it \
  --layer 25

python scripts/plot_matrix.py --tag gemma2-2b-it
```

The matrix evaluates:

- DoM safety ablation
- DoM utility ablation
- ActSVD safety ablation
- ActSVD utility ablation
- random direction control
- random subspace control
- baseline with no ablation

### 4. Run sensitivity analysis

```bash
python scripts/run_sensitivity.py \
  --model google/gemma-2-2b-it \
  --tag gemma2-2b-it \
  --layers 22,23,24,25 \
  --ranks 1,2,4,8,16 \
  --rank_layer 25
```

This checks whether the effect is stable across nearby layers and across ActSVD ranks.

### 5. Measure subspace overlap

```bash
python scripts/compute_subspace_angles.py --tag gemma2-2b-it
python scripts/compute_subspace_angles.py --tag llama3.1-8b-it
```

This reports principal angles and a simple subspace similarity measure between the safety and utility ActSVD bases.

## Layers Used in the Current Results

The saved results in this repo use the following extraction layers:

- Gemma: layer 25
- Llama: layer 11

For Llama, the EDA peak lands on hidden_states index 32, but that is not a hookable transformer block. The project therefore uses the nearest layer that actually supports ablation, and the layer sweep suggests that early layers are more causally relevant for the safety behavior in this model.

## Outputs

After a run, each `results/<tag>/` directory contains:

- `eda_layer_selection.png`
- `eda_helpsteer_delta.png`
- `eda_stability.png`
- `eda_summary.json`
- `matrix_results.json`
- `matrix_plot.png`
- `sensitivity_layer_sweep.json`
- `sensitivity_rank_sweep.json`
- `subspace_angles.json`

The corresponding `cache/<tag>/` directory stores the extracted vectors, bases, and held-out ASR prompts used by the downstream scripts.

## Key Findings

The main finding is that the extracted safety and utility directions are not strongly aligned. Subspace overlap is low, and ablation results suggest the directions have partially distinct behavioral effects rather than collapsing into a single generic steering axis.

There is also an important model-specific difference:

- Gemma’s EDA-selected layer lines up well with the ablation results
- Llama’s EDA peak does not directly correspond to a hookable block, so the canonical ablation layer is chosen from the nearby causal sweep instead

## Notes on the Codebase

- [src/common.py](src/common.py) handles model loading, activation extraction, and dataset loading.
- [src/directions.py](src/directions.py) implements DoM, ActSVD, projection, and angle calculations.
- [src/ablation.py](src/ablation.py) contains the hook classes used to modify hidden states.
- [src/evaluation.py](src/evaluation.py) defines ASR scoring and utility benchmarks.
- [hw3/](hw3/) contains the code used for the midterm/hw3