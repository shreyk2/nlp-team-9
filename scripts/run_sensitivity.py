# layer sweep and ActSVD rank sweep for sensitivity analysis
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.ablation import DirectionalAblationHook, SubspaceAblationHook
from src.common import (
    get_prompt_activations,
    load_advbench_prompts,
    load_alpaca_prompts,
    load_model,
    set_seed,
)
from src.directions import compute_actsvd_subspace, compute_dom_direction
from src.evaluation import evaluate_all_utility_benchmarks, evaluate_attack_success_rate


# load or compute all-layer activations for the safety pool
def get_or_cache_safety_pool_activations(cache_directory, model, tokenizer, device, n_extract, n_eval):
    sensitivity_directory = os.path.join(cache_directory, "sensitivity_acts")
    os.makedirs(sensitivity_directory, exist_ok=True)
    harmful_path = os.path.join(sensitivity_directory, "harmful_all_layers.npy")
    harmless_path = os.path.join(sensitivity_directory, "harmless_all_layers.npy")
    eval_path = os.path.join(sensitivity_directory, "harmful_eval.json")

    if (
        os.path.exists(harmful_path)
        and os.path.exists(harmless_path)
        and os.path.exists(eval_path)
    ):
        print("  using cached activations")
        with open(eval_path) as f:
            harmful_evaluation_prompts = json.load(f)["harmful_eval"]
        return (
            np.load(harmful_path),
            np.load(harmless_path),
            harmful_evaluation_prompts,
        )

    advbench_prompts = load_advbench_prompts()
    np.random.shuffle(advbench_prompts)
    harmful_extraction_prompts = advbench_prompts[:n_extract]
    harmful_evaluation_prompts = advbench_prompts[n_extract:n_extract + n_eval]
    harmless_prompts = load_alpaca_prompts(max_count=n_extract)

    harmful_activations = get_prompt_activations(
        harmful_extraction_prompts, model, tokenizer, device
    )
    harmless_activations = get_prompt_activations(
        harmless_prompts, model, tokenizer, device
    )

    np.save(harmful_path, harmful_activations)
    np.save(harmless_path, harmless_activations)
    with open(eval_path, "w") as f:
        json.dump({"harmful_eval": harmful_evaluation_prompts}, f)

    return harmful_activations, harmless_activations, harmful_evaluation_prompts


def parse_command_line_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--layers", default="")
    parser.add_argument("--ranks", default="1,2,4,8,16")
    parser.add_argument("--rank_layer", type=int, default=None)
    parser.add_argument("--n_extract", type=int, default=256)
    parser.add_argument("--n_eval", type=int, default=100)
    parser.add_argument("--n_utility", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache_root", default="cache")
    parser.add_argument("--results_root", default="results")
    return parser.parse_args()


# parse comma-separated layer list or default to all layers
def parse_layer_list(layers_argument, total_layer_count):
    if not layers_argument:
        return list(range(total_layer_count))
    parsed_layers = []
    for layer_string in layers_argument.split(","):
        parsed_layers.append(int(layer_string))
    return parsed_layers


# parse comma-separated rank list
def parse_rank_list(ranks_argument):
    parsed_ranks = []
    for rank_string in ranks_argument.split(","):
        parsed_ranks.append(int(rank_string))
    return parsed_ranks


# DoM safety ablation at each layer, record ASR + utility
def run_layer_sweep(
    model,
    tokenizer,
    device,
    harmful_activations,
    harmless_activations,
    harmful_evaluation_prompts,
    layers_to_test,
    seed,
    n_utility,
    output_path,
):
    layer_results = []
    for layer_index in layers_to_test:
        print(f"\n--- layer {layer_index} (DoM safety ablation) ---")
        set_seed(seed)
        safety_direction = compute_dom_direction(
            harmful_activations[:, layer_index, :],
            harmless_activations[:, layer_index, :],
        )
        hook = DirectionalAblationHook(safety_direction, layer_index)
        hook.attach(model)
        try:
            attack_success_rate, _ = evaluate_attack_success_rate(
                model, tokenizer, device,
                harmful_evaluation_prompts,
                description=f"L{layer_index}-ASR",
            )
            utility_scores = evaluate_all_utility_benchmarks(
                model, tokenizer, device, sample_count=n_utility
            )
        finally:
            hook.remove()
        row = {"layer": layer_index, "asr": attack_success_rate, **utility_scores}
        layer_results.append(row)
        print(f"  L={layer_index}: ASR={attack_success_rate:.1%} avg_util={utility_scores['avg_utility']:.1%}")
        with open(output_path, "w") as f:
            json.dump(layer_results, f, indent=2)
    return layer_results


# ActSVD ablation at varying ranks, record ASR + utility
def run_rank_sweep(
    model,
    tokenizer,
    device,
    harmful_activations,
    harmless_activations,
    harmful_evaluation_prompts,
    ranks_to_test,
    rank_layer,
    seed,
    n_utility,
    output_path,
):
    rank_results = []
    for rank_value in ranks_to_test:
        print(f"\n--- rank {rank_value} (ActSVD safety ablation @ layer {rank_layer}) ---")
        set_seed(seed)
        safety_subspace = compute_actsvd_subspace(
            harmful_activations[:, rank_layer, :],
            harmless_activations[:, rank_layer, :],
            rank=rank_value,
        )
        hook = SubspaceAblationHook(safety_subspace, rank_layer)
        hook.attach(model)
        try:
            attack_success_rate, _ = evaluate_attack_success_rate(
                model, tokenizer, device,
                harmful_evaluation_prompts,
                description=f"k{rank_value}-ASR",
            )
            utility_scores = evaluate_all_utility_benchmarks(
                model, tokenizer, device, sample_count=n_utility
            )
        finally:
            hook.remove()
        row = {
            "rank": rank_value,
            "layer": rank_layer,
            "asr": attack_success_rate,
            **utility_scores,
        }
        rank_results.append(row)
        print(f"  k={rank_value}: ASR={attack_success_rate:.1%} avg_util={utility_scores['avg_utility']:.1%}")
        with open(output_path, "w") as f:
            json.dump(rank_results, f, indent=2)
    return rank_results


def main():
    args = parse_command_line_arguments()
    set_seed(args.seed)

    cache_directory = os.path.join(args.cache_root, args.tag)
    output_directory = os.path.join(args.results_root, args.tag)
    os.makedirs(output_directory, exist_ok=True)

    model, tokenizer, device = load_model(args.model)
    total_layer_count = model.config.num_hidden_layers + 1

    print("\n[1] preparing safety pool activations (all layers)")
    harmful_activations, harmless_activations, harmful_evaluation_prompts = (
        get_or_cache_safety_pool_activations(
            cache_directory, model, tokenizer, device, args.n_extract, args.n_eval
        )
    )

    layers_to_test = parse_layer_list(args.layers, total_layer_count)
    print(f"  layers to test: {layers_to_test}")

    layer_sweep_path = os.path.join(output_directory, "sensitivity_layer_sweep.json")
    layer_results = run_layer_sweep(
        model,
        tokenizer,
        device,
        harmful_activations,
        harmless_activations,
        harmful_evaluation_prompts,
        layers_to_test,
        args.seed,
        args.n_utility,
        layer_sweep_path,
    )

    if args.rank_layer is None:
        best_layer_record = max(layer_results, key=lambda record: record["asr"])
        rank_sweep_layer = best_layer_record["layer"]
    else:
        rank_sweep_layer = args.rank_layer

    print(f"\n[2] ActSVD rank sweep at layer {rank_sweep_layer}")
    ranks_to_test = parse_rank_list(args.ranks)
    rank_sweep_path = os.path.join(output_directory, "sensitivity_rank_sweep.json")
    run_rank_sweep(
        model,
        tokenizer,
        device,
        harmful_activations,
        harmless_activations,
        harmful_evaluation_prompts,
        ranks_to_test,
        rank_sweep_layer,
        args.seed,
        args.n_utility,
        rank_sweep_path,
    )

    print(f"\nsaved: {layer_sweep_path}, {rank_sweep_path}")


if __name__ == "__main__":
    main()
