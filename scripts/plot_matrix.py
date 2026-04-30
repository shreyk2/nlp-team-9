import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np


def parse_command_line_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--results_root", default="results")
    return parser.parse_args()


def main():
    args = parse_command_line_arguments()

    matrix_results_path = os.path.join(args.results_root, args.tag, "matrix_results.json")
    with open(matrix_results_path) as f:
        matrix_data = json.load(f)
    results_by_cell = matrix_data["results"]

    canonical_cell_order = [
        "baseline",
        "dom_safety",
        "actsvd_safety",
        "dom_utility",
        "actsvd_utility",
    ]
    present_cells = []
    for cell_name in canonical_cell_order:
        if cell_name in results_by_cell:
            present_cells.append(cell_name)

    metric_keys = ["asr", "boolq", "hellaswag", "arc_challenge", "truthfulqa_mc1"]
    metric_display_names = ["ASR", "BoolQ", "HellaSwag", "ARC-C", "TruthfulQA"]

    score_matrix = []
    for cell_name in present_cells:
        cell_scores = []
        for metric_key in metric_keys:
            cell_scores.append(results_by_cell[cell_name][metric_key])
        score_matrix.append(cell_scores)
    score_matrix = np.array(score_matrix)

    fig, axis = plt.subplots(figsize=(12, 6))
    metric_x_positions = np.arange(len(metric_keys))
    bar_width = 0.8 / len(present_cells)
    cell_colors = ["#888888", "#d62728", "#9467bd", "#1f77b4", "#17becf"]

    for cell_position, cell_name in enumerate(present_cells):
        bar_x_positions = metric_x_positions + cell_position * bar_width
        bar_heights = score_matrix[cell_position] * 100
        axis.bar(
            bar_x_positions,
            bar_heights,
            bar_width,
            label=cell_name,
            color=cell_colors[cell_position % len(cell_colors)],
            edgecolor="black",
            linewidth=0.6,
        )

    tick_offset = bar_width * (len(present_cells) - 1) / 2
    axis.set_xticks(metric_x_positions + tick_offset)
    axis.set_xticklabels(metric_display_names)
    axis.set_ylabel("score (%)")
    axis.set_title(f"Cross-intervention 2×2 matrix — {args.tag}")
    axis.legend(loc="best")
    axis.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    output_path = os.path.join(args.results_root, args.tag, "matrix_plot.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
