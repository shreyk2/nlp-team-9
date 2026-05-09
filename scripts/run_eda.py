# exploratory analysis: layer selection, delta thresholds, direction stability
import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.common import (
    cosine_similarity,
    get_prompt_activations,
    get_response_activations,
    load_advbench_prompts,
    load_alpaca_prompts,
    load_helpsteer_pairs,
    load_model,
    set_seed,
)


def parse_command_line_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--n_layer_select", type=int, default=100)
    parser.add_argument("--n_safety_pool", type=int, default=512)
    parser.add_argument("--n_utility_pool", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results_root", default="results")
    return parser.parse_args()


# get high/low response activations grouped by helpfulness delta
def collect_activations_per_delta(pairs_by_delta, model, tokenizer, device, max_count_per_delta):
    high_activations_by_delta = {}
    low_activations_by_delta = {}
    for delta_value in [1, 2, 3, 4]:
        if len(pairs_by_delta[delta_value]) == 0:
            continue
        sampled_pairs = pairs_by_delta[delta_value][:max_count_per_delta]
        high_activations_by_delta[delta_value] = get_response_activations(
            sampled_pairs, "high", model, tokenizer, device
        )
        low_activations_by_delta[delta_value] = get_response_activations(
            sampled_pairs, "low", model, tokenizer, device
        )
    return high_activations_by_delta, low_activations_by_delta


# compute per-layer norms, cosines, and divergence score for layer picking
def compute_layer_selection_metrics(
    harmful_sample, harmless_sample, high_sample, low_sample, layer_count
):
    safety_norms = np.zeros(layer_count)
    utility_norms = np.zeros(layer_count)
    cosines = np.zeros(layer_count)
    divergence_scores = np.zeros(layer_count)
    for layer_index in range(layer_count):
        safety_direction = (
            harmful_sample[:, layer_index, :].mean(0)
            - harmless_sample[:, layer_index, :].mean(0)
        )
        utility_direction = (
            high_sample[:, layer_index, :].mean(0)
            - low_sample[:, layer_index, :].mean(0)
        )
        safety_norms[layer_index] = np.linalg.norm(safety_direction)
        utility_norms[layer_index] = np.linalg.norm(utility_direction)
        cosines[layer_index] = cosine_similarity(safety_direction, utility_direction)
        divergence_scores[layer_index] = (
            (safety_norms[layer_index] * utility_norms[layer_index])
            / (abs(cosines[layer_index]) + 1e-8)
        )
    return safety_norms, utility_norms, cosines, divergence_scores


# 3-panel plot: direction norms, cosine curve, divergence with recommendation
def plot_layer_selection_panels(
    layer_indices,
    safety_norms,
    utility_norms,
    cosines,
    divergence_scores,
    recommended_layer,
    output_path,
):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    safety_norms_normalized = safety_norms / safety_norms.max()
    utility_norms_normalized = utility_norms / utility_norms.max()

    axes[0].plot(
        layer_indices, safety_norms_normalized,
        marker="o", label="||r_s|| (norm.)", color="red",
    )
    axes[0].plot(
        layer_indices, utility_norms_normalized,
        marker="s", label="||r_u|| (norm.)", color="blue",
    )
    axes[0].plot(
        layer_indices, np.abs(cosines),
        marker="^", linestyle="--", label="|cos(r_s,r_u)|", color="purple",
    )
    axes[0].set_xlabel("layer")
    axes[0].set_title("Direction properties per layer")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(layer_indices, cosines, color="black")
    axes[1].fill_between(
        layer_indices, cosines, 0,
        where=(cosines > 0), alpha=0.3, color="red",
    )
    axes[1].fill_between(
        layer_indices, cosines, 0,
        where=(cosines <= 0), alpha=0.3, color="blue",
    )
    axes[1].set_xlabel("layer")
    axes[1].set_title(f"cos(r_s, r_u)  (mean |cos|={np.mean(np.abs(cosines)):.3f})")
    axes[1].grid(alpha=0.3)

    axes[2].plot(layer_indices, divergence_scores, marker="o", color="black")
    axes[2].scatter(
        [recommended_layer],
        [divergence_scores[recommended_layer]],
        marker="*", s=400, color="gold", edgecolors="black", zorder=5,
    )
    axes[2].set_xlabel("layer")
    axes[2].set_title(f"Divergence score (recommended L={recommended_layer})")
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# 3-panel plot: cosine vs delta, norm vs delta, PCA scatter for delta=4
def plot_helpsteer_delta_panels(
    high_activations_by_delta,
    low_activations_by_delta,
    extraction_layer,
    output_path,
):
    deltas_present = sorted(high_activations_by_delta.keys())
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    mean_cosines = []
    cosine_standard_deviations = []
    direction_norms = []
    for delta_value in deltas_present:
        high_at_layer = high_activations_by_delta[delta_value][:, extraction_layer, :]
        low_at_layer = low_activations_by_delta[delta_value][:, extraction_layer, :]
        pair_count = min(len(high_at_layer), len(low_at_layer), 50)
        high_at_layer = high_at_layer[:pair_count]
        low_at_layer = low_at_layer[:pair_count]
        per_pair_cosines = []
        for pair_index in range(pair_count):
            per_pair_cosines.append(
                cosine_similarity(high_at_layer[pair_index], low_at_layer[pair_index])
            )
        mean_cosines.append(np.mean(per_pair_cosines))
        cosine_standard_deviations.append(np.std(per_pair_cosines))
        direction_norms.append(
            np.linalg.norm(high_at_layer.mean(0) - low_at_layer.mean(0))
        )

    delta_labels = []
    for delta_value in deltas_present:
        delta_labels.append(f"δ={delta_value}")

    axes[0].bar(
        range(len(deltas_present)), mean_cosines,
        yerr=cosine_standard_deviations, capsize=5,
    )
    axes[0].set_xticks(range(len(deltas_present)))
    axes[0].set_xticklabels(delta_labels)
    axes[0].set_title("Mean cos(high, low) per δ — lower=better")
    axes[0].grid(alpha=0.3, axis="y")

    axes[1].bar(range(len(deltas_present)), direction_norms)
    axes[1].set_xticks(range(len(deltas_present)))
    axes[1].set_xticklabels(delta_labels)
    axes[1].set_title("||r_u|| per δ — higher=better")
    axes[1].grid(alpha=0.3, axis="y")

    if 4 in high_activations_by_delta:
        high_at_layer_delta_4 = high_activations_by_delta[4][:, extraction_layer, :]
        low_at_layer_delta_4 = low_activations_by_delta[4][:, extraction_layer, :]
        stacked_vectors = np.vstack([high_at_layer_delta_4, low_at_layer_delta_4])
        projected_2d = PCA(n_components=2).fit_transform(stacked_vectors)
        high_count = len(high_at_layer_delta_4)
        axes[2].scatter(
            projected_2d[:high_count, 0], projected_2d[:high_count, 1],
            color="#ADD8E6", edgecolors="black", label="high (δ=4)",
        )
        axes[2].scatter(
            projected_2d[high_count:, 0], projected_2d[high_count:, 1],
            color="#FFA07A", edgecolors="black", label="low (δ=4)",
        )
        axes[2].set_title(f"PCA δ=4 only @ layer {extraction_layer}")
        axes[2].legend()
        axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return deltas_present, mean_cosines, direction_norms


# bootstrap resampled directions to test stability at a given sample size
def bootstrap_direction_samples(
    positive_activations, negative_activations, sample_size, layer_index, trial_count=10
):
    sampled_directions = []
    for _ in range(trial_count):
        actual_size = min(sample_size, len(positive_activations), len(negative_activations))
        positive_indices = np.random.choice(len(positive_activations), actual_size, replace=False)
        negative_indices = np.random.choice(len(negative_activations), actual_size, replace=False)
        direction = (
            positive_activations[positive_indices, layer_index, :].mean(0)
            - negative_activations[negative_indices, layer_index, :].mean(0)
        )
        sampled_directions.append(direction / (np.linalg.norm(direction) + 1e-8))
    return sampled_directions


# mean and std of all pairwise cosines between bootstrap samples
def compute_pairwise_cosine_statistics(directions):
    pairwise_cosines = []
    for first_index in range(len(directions)):
        for second_index in range(first_index + 1, len(directions)):
            pairwise_cosines.append(
                cosine_similarity(directions[first_index], directions[second_index])
            )
    return float(np.mean(pairwise_cosines)), float(np.std(pairwise_cosines))


# plot direction stability vs sample size N
def plot_stability_curve(
    sample_sizes,
    safety_stability,
    utility_stability,
    extraction_layer,
    output_path,
):
    fig, axis = plt.subplots(figsize=(8, 5))

    safety_means = []
    safety_stds = []
    utility_means = []
    utility_stds = []
    for sample_size in sample_sizes:
        safety_means.append(safety_stability[sample_size][0])
        safety_stds.append(safety_stability[sample_size][1])
        utility_means.append(utility_stability[sample_size][0])
        utility_stds.append(utility_stability[sample_size][1])

    axis.errorbar(
        sample_sizes, safety_means, yerr=safety_stds,
        marker="o", linewidth=2, label="safety", color="red",
    )
    axis.errorbar(
        sample_sizes, utility_means, yerr=utility_stds,
        marker="^", linewidth=2, label="utility (δ≥3)", color="green",
    )
    axis.axhline(0.95, linestyle="--", color="gray", label="stable threshold")
    axis.set_xlabel("N")
    axis.set_ylabel("mean pairwise cos")
    axis.set_xticks(sample_sizes)
    axis.set_ylim([0, 1.01])
    axis.set_title(f"Direction stability vs N (layer {extraction_layer})")
    axis.legend()
    axis.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    args = parse_command_line_arguments()
    set_seed(args.seed)

    output_directory = os.path.join(args.results_root, args.tag)
    os.makedirs(output_directory, exist_ok=True)

    model, tokenizer, device = load_model(args.model)

    print("\n[1] loading data")
    advbench_prompts = load_advbench_prompts()
    np.random.shuffle(advbench_prompts)
    harmful_prompts = advbench_prompts[: args.n_safety_pool]
    harmless_prompts = load_alpaca_prompts(max_count=args.n_safety_pool)
    all_helpsteer_pairs = load_helpsteer_pairs(min_helpfulness_delta=1)

    pairs_by_delta = {}
    for delta_value in [1, 2, 3, 4]:
        pairs_by_delta[delta_value] = []
        for pair in all_helpsteer_pairs:
            if pair["delta"] == delta_value:
                pairs_by_delta[delta_value].append(pair)

    print(f"  harmful={len(harmful_prompts)} harmless={len(harmless_prompts)}")
    for delta_value in [1, 2, 3, 4]:
        print(f"  helpsteer δ={delta_value}: {len(pairs_by_delta[delta_value])} pairs")

    print("\n[2] collecting activations across all layers")
    harmful_activations = get_prompt_activations(harmful_prompts, model, tokenizer, device)
    harmless_activations = get_prompt_activations(harmless_prompts, model, tokenizer, device)
    high_activations_by_delta, low_activations_by_delta = collect_activations_per_delta(
        pairs_by_delta, model, tokenizer, device, args.n_utility_pool
    )

    print("\n[3] layer selection")
    high_activations_geq3 = []
    low_activations_geq3 = []
    for delta_value in [3, 4]:
        if delta_value in high_activations_by_delta:
            high_activations_geq3.append(high_activations_by_delta[delta_value])
            low_activations_geq3.append(low_activations_by_delta[delta_value])
    high_activations_geq3 = np.vstack(high_activations_geq3)
    low_activations_geq3 = np.vstack(low_activations_geq3)

    sample_size = args.n_layer_select
    harmful_sample_indices = np.random.choice(
        len(harmful_activations),
        min(sample_size, len(harmful_activations)),
        replace=False,
    )
    harmless_sample_indices = np.random.choice(
        len(harmless_activations),
        min(sample_size, len(harmless_activations)),
        replace=False,
    )
    high_sample_indices = np.random.choice(
        len(high_activations_geq3),
        min(sample_size, len(high_activations_geq3)),
        replace=False,
    )
    low_sample_indices = np.random.choice(
        len(low_activations_geq3),
        min(sample_size, len(low_activations_geq3)),
        replace=False,
    )

    harmful_sample = harmful_activations[harmful_sample_indices]
    harmless_sample = harmless_activations[harmless_sample_indices]
    high_sample = high_activations_geq3[high_sample_indices]
    low_sample = low_activations_geq3[low_sample_indices]

    layer_count = harmful_activations.shape[1]
    safety_norms, utility_norms, cosines, divergence_scores = compute_layer_selection_metrics(
        harmful_sample, harmless_sample, high_sample, low_sample, layer_count
    )
    recommended_layer = int(np.argmax(divergence_scores))
    print(f"  recommended extraction layer: {recommended_layer}")

    layer_indices = np.arange(layer_count)
    plot_layer_selection_panels(
        layer_indices,
        safety_norms,
        utility_norms,
        cosines,
        divergence_scores,
        recommended_layer,
        os.path.join(output_directory, "eda_layer_selection.png"),
    )

    print("\n[4] HelpSteer δ comparison")
    deltas_present, mean_cosines_per_delta, direction_norms_per_delta = (
        plot_helpsteer_delta_panels(
            high_activations_by_delta,
            low_activations_by_delta,
            recommended_layer,
            os.path.join(output_directory, "eda_helpsteer_delta.png"),
        )
    )

    print("\n[5] stability vs sample size")
    candidate_sample_sizes = [
        32, 64, 128, 256,
        min(400, len(harmful_activations), len(harmless_activations)),
    ]
    sample_sizes = []
    for size in candidate_sample_sizes:
        if size > 0 and size not in sample_sizes:
            sample_sizes.append(size)
    sample_sizes = sorted(sample_sizes)

    bootstrap_trial_count = 10
    safety_stability = {}
    utility_stability_geq3 = {}
    for sample_size in sample_sizes:
        safety_directions = bootstrap_direction_samples(
            harmful_activations, harmless_activations,
            sample_size, recommended_layer, bootstrap_trial_count,
        )
        utility_directions = bootstrap_direction_samples(
            high_activations_geq3, low_activations_geq3,
            sample_size, recommended_layer, bootstrap_trial_count,
        )
        safety_stability[sample_size] = compute_pairwise_cosine_statistics(safety_directions)
        utility_stability_geq3[sample_size] = compute_pairwise_cosine_statistics(
            utility_directions
        )

    plot_stability_curve(
        sample_sizes,
        safety_stability,
        utility_stability_geq3,
        recommended_layer,
        os.path.join(output_directory, "eda_stability.png"),
    )

    safety_stability_serializable = {}
    for sample_size in sample_sizes:
        safety_stability_serializable[str(sample_size)] = list(safety_stability[sample_size])
    utility_stability_serializable = {}
    for sample_size in sample_sizes:
        utility_stability_serializable[str(sample_size)] = list(
            utility_stability_geq3[sample_size]
        )

    direction_norms_serializable = []
    for value in direction_norms_per_delta:
        direction_norms_serializable.append(float(value))
    mean_cosines_serializable = []
    for value in mean_cosines_per_delta:
        mean_cosines_serializable.append(float(value))

    summary = {
        "model": args.model,
        "tag": args.tag,
        "n_layers_with_embedding": int(layer_count),
        "n_layers_transformer": int(layer_count - 1),
        "recommended_layer": recommended_layer,
        "mean_abs_cos_su_all_layers": float(np.mean(np.abs(cosines))),
        "rec_layer_metrics": {
            "rs_norm": float(safety_norms[recommended_layer]),
            "ru_norm": float(utility_norms[recommended_layer]),
            "cos_su": float(cosines[recommended_layer]),
            "divergence_score": float(divergence_scores[recommended_layer]),
        },
        "stability": {
            "safety": safety_stability_serializable,
            "utility_geq3": utility_stability_serializable,
        },
        "delta_thresholds": {
            "deltas": deltas_present,
            "mean_cos_high_low": mean_cosines_serializable,
            "direction_norms": direction_norms_serializable,
        },
    }
    with open(os.path.join(output_directory, "eda_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nsaved EDA outputs to {output_directory}/")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
