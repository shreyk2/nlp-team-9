import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.ablation import DirectionalAblationHook, NoOpHook, SubspaceAblationHook
from src.common import load_model, set_seed
from src.evaluation import evaluate_all_utility_benchmarks, evaluate_attack_success_rate


def build_hook_for_cell(extraction_method, ablation_target, cache_directory, layer):
    if extraction_method == "dom":
        path = os.path.join(cache_directory, f"r_{ablation_target}_dom.npy")
        direction = np.load(path)
        return DirectionalAblationHook(direction, layer)
    if extraction_method == "actsvd":
        path = os.path.join(cache_directory, f"V_{ablation_target}_actsvd.npy")
        subspace = np.load(path)
        return SubspaceAblationHook(subspace, layer)
    raise ValueError(extraction_method)


def run_single_cell(model, tokenizer, device, hook, harmful_evaluation_prompts, utility_sample_count):
    hook.attach(model)
    try:
        attack_success_rate, per_prompt_records = evaluate_attack_success_rate(
            model, tokenizer, device, harmful_evaluation_prompts, description="ASR"
        )
        utility_scores = evaluate_all_utility_benchmarks(
            model, tokenizer, device, sample_count=utility_sample_count
        )
    finally:
        hook.remove()
    return {
        "asr": attack_success_rate,
        **utility_scores,
        "asr_examples": per_prompt_records[:10],
    }


def parse_command_line_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--n_utility", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache_root", default="cache")
    parser.add_argument("--results_root", default="results")
    parser.add_argument("--cells", default="all")
    return parser.parse_args()


def main():
    args = parse_command_line_arguments()
    set_seed(args.seed)

    cache_directory = os.path.join(args.cache_root, args.tag)
    output_directory = os.path.join(args.results_root, args.tag)
    os.makedirs(output_directory, exist_ok=True)

    with open(os.path.join(cache_directory, "eval_prompts.json")) as f:
        harmful_evaluation_prompts = json.load(f)["harmful_eval"]
    print(f"loaded {len(harmful_evaluation_prompts)} held-out AdvBench eval prompts")

    model, tokenizer, device = load_model(args.model)

    full_cell_specifications = [
        ("baseline", None, None),
        ("dom_safety", "dom", "safety"),
        ("dom_utility", "dom", "utility"),
        ("actsvd_safety", "actsvd", "safety"),
        ("actsvd_utility", "actsvd", "utility"),
    ]

    if args.cells == "all":
        selected_cell_specifications = full_cell_specifications
    else:
        wanted_cell_names = set(args.cells.split(","))
        selected_cell_specifications = []
        for specification in full_cell_specifications:
            if specification[0] in wanted_cell_names:
                selected_cell_specifications.append(specification)

    results_by_cell = {}
    for cell_name, extraction_method, ablation_target in selected_cell_specifications:
        print("\n" + "=" * 70)
        print(f"CELL: {cell_name}")
        print("=" * 70)
        set_seed(args.seed)
        if cell_name == "baseline":
            hook = NoOpHook()
        else:
            hook = build_hook_for_cell(
                extraction_method, ablation_target, cache_directory, args.layer
            )
        cell_results = run_single_cell(
            model, tokenizer, device, hook, harmful_evaluation_prompts, args.n_utility
        )
        results_by_cell[cell_name] = cell_results
        with open(os.path.join(output_directory, "matrix_results.json"), "w") as f:
            json.dump({"meta": vars(args), "results": results_by_cell}, f, indent=2)
        print(
            f"  ASR={cell_results['asr']:.1%} | "
            f"avg_util={cell_results['avg_utility']:.1%} | "
            f"truthfulqa={cell_results['truthfulqa_mc1']:.1%}"
        )

    print("\n" + "=" * 70)
    print("RESULTS — cross-intervention 2x2 matrix")
    print("=" * 70)
    header_line = (
        f"{'cell':<20} {'ASR':>8} {'BoolQ':>8} {'HSwag':>8} "
        f"{'ARC-C':>8} {'AvgU':>8} {'TruthQA':>8}"
    )
    print(header_line)
    print("-" * len(header_line))
    for cell_name, _, _ in selected_cell_specifications:
        if cell_name not in results_by_cell:
            continue
        row = results_by_cell[cell_name]
        print(
            f"{cell_name:<20} {row['asr']:>7.1%} {row['boolq']:>7.1%} "
            f"{row['hellaswag']:>7.1%} {row['arc_challenge']:>7.1%} "
            f"{row['avg_utility']:>7.1%} {row['truthfulqa_mc1']:>7.1%}"
        )

    print(f"\nsaved: {os.path.join(output_directory, 'matrix_results.json')}")


if __name__ == "__main__":
    main()
