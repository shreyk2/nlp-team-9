# extract and cache DoM + ActSVD directions at a chosen layer
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.common import (
    get_prompt_activations,
    get_response_activations,
    load_advbench_prompts,
    load_alpaca_prompts,
    load_helpsteer_pairs,
    load_model,
    set_seed,
)
from src.directions import compute_actsvd_subspace, compute_dom_direction


def parse_command_line_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--n_extract", type=int, default=256)
    parser.add_argument("--n_eval", type=int, default=100)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--min_delta", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache_root", default="cache")
    return parser.parse_args()


def main():
    args = parse_command_line_arguments()
    set_seed(args.seed)

    cache_directory = os.path.join(args.cache_root, args.tag)
    os.makedirs(cache_directory, exist_ok=True)

    model, tokenizer, device = load_model(args.model)

    print("\n[1/4] loading AdvBench / Alpaca for safety direction")
    advbench_prompts = load_advbench_prompts()
    np.random.shuffle(advbench_prompts)
    harmful_extraction_prompts = advbench_prompts[: args.n_extract]
    harmful_evaluation_prompts = advbench_prompts[args.n_extract : args.n_extract + args.n_eval]
    harmless_prompts = load_alpaca_prompts(max_count=args.n_extract)
    print(
        f"  harmful_extract={len(harmful_extraction_prompts)} "
        f"harmful_eval={len(harmful_evaluation_prompts)} "
        f"harmless={len(harmless_prompts)}"
    )

    print(f"\n[2/4] loading HelpSteer pairs (min_delta={args.min_delta})")
    helpsteer_pairs = load_helpsteer_pairs(min_helpfulness_delta=args.min_delta)
    np.random.shuffle(helpsteer_pairs)
    helpsteer_pairs = helpsteer_pairs[: args.n_extract]
    print(f"  utility pairs used: {len(helpsteer_pairs)}")

    print(f"\n[3/4] collecting activations at layer {args.layer}")
    harmful_activations = get_prompt_activations(
        harmful_extraction_prompts, model, tokenizer, device, layer=args.layer
    )
    harmless_activations = get_prompt_activations(
        harmless_prompts, model, tokenizer, device, layer=args.layer
    )
    high_helpfulness_activations = get_response_activations(
        helpsteer_pairs, "high", model, tokenizer, device, layer=args.layer
    )
    low_helpfulness_activations = get_response_activations(
        helpsteer_pairs, "low", model, tokenizer, device, layer=args.layer
    )
    print(
        f"  shapes: harmful={harmful_activations.shape}, "
        f"harmless={harmless_activations.shape}"
    )
    print(
        f"          high={high_helpfulness_activations.shape}, "
        f"low={low_helpfulness_activations.shape}"
    )

    print("\n[4/4] extracting DoM + ActSVD directions")
    safety_direction_dom = compute_dom_direction(harmful_activations, harmless_activations)
    utility_direction_dom = compute_dom_direction(
        high_helpfulness_activations, low_helpfulness_activations
    )
    safety_subspace_actsvd = compute_actsvd_subspace(
        harmful_activations,
        harmless_activations,
        rank=args.rank,
        rank_position="top",
    )
    # Pivot: keep the actsvd_utility cell name, but source it from the
    # least safety-relevant safety ranks instead of raw HelpSteer utility ranks.
    utility_subspace_actsvd = compute_actsvd_subspace(
        harmful_activations,
        harmless_activations,
        rank=args.rank,
        rank_position="bottom",
    )

    cosine_dom = float(np.dot(safety_direction_dom, utility_direction_dom))
    print(f"  cos(r_s_dom, r_u_dom) at layer {args.layer} = {cosine_dom:.4f}")

    np.save(os.path.join(cache_directory, "r_safety_dom.npy"), safety_direction_dom)
    np.save(os.path.join(cache_directory, "r_utility_dom.npy"), utility_direction_dom)
    np.save(os.path.join(cache_directory, "V_safety_actsvd.npy"), safety_subspace_actsvd)
    np.save(os.path.join(cache_directory, "V_utility_actsvd.npy"), utility_subspace_actsvd)

    with open(os.path.join(cache_directory, "eval_prompts.json"), "w") as f:
        json.dump({"harmful_eval": harmful_evaluation_prompts}, f, indent=2)

    metadata = {
        "model": args.model,
        "tag": args.tag,
        "layer": args.layer,
        "n_extract": args.n_extract,
        "n_eval": args.n_eval,
        "rank": args.rank,
        "min_delta": args.min_delta,
        "seed": args.seed,
        "cos_dom_at_layer": cosine_dom,
        "actsvd_safety_source": "top_safety_ranks",
        "actsvd_safety_rank_position": "top",
        "actsvd_utility_source": "bottom_safety_ranks",
        "actsvd_utility_rank_position": "bottom",
    }
    with open(os.path.join(cache_directory, "meta.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nsaved everything to {cache_directory}/")


if __name__ == "__main__":
    main()
