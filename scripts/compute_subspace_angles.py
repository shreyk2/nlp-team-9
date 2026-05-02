import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.directions import principal_angles_between_subspaces


def parse_command_line_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--cache_root", default="cache")
    parser.add_argument("--results_root", default="results")
    return parser.parse_args()


def main():
    args = parse_command_line_arguments()

    cache_directory = os.path.join(args.cache_root, args.tag)
    output_directory = os.path.join(args.results_root, args.tag)
    os.makedirs(output_directory, exist_ok=True)

    safety_subspace = np.load(os.path.join(cache_directory, "V_safety_actsvd.npy"))
    utility_subspace = np.load(os.path.join(cache_directory, "V_utility_actsvd.npy"))

    safety_direction = np.load(os.path.join(cache_directory, "r_safety_dom.npy"))
    utility_direction = np.load(os.path.join(cache_directory, "r_utility_dom.npy"))

    angles_data = principal_angles_between_subspaces(safety_subspace, utility_subspace)

    cosine_dom = float(np.dot(safety_direction, utility_direction))

    output = {
        "tag": args.tag,
        "safety_subspace_shape": list(safety_subspace.shape),
        "utility_subspace_shape": list(utility_subspace.shape),
        "actsvd_subspace_similarity_phi": angles_data["subspace_similarity_phi"],
        "actsvd_cos_principal_angles": angles_data["cos_principal_angles"],
        "actsvd_principal_angles_degrees": angles_data["principal_angles_degrees"],
        "dom_cosine_safety_utility": cosine_dom,
    }

    print(f"\ntag = {args.tag}")
    print(f"  DoM cos(r_s, r_u) = {cosine_dom:.4f}")
    print(f"  ActSVD φ(U_s, U_u) = {angles_data['subspace_similarity_phi']:.4f}  (mean cos² of principal angles)")
    print(f"  principal angles (degrees) = " + ", ".join(f"{angle:.1f}°" for angle in angles_data["principal_angles_degrees"]))

    output_path = os.path.join(output_directory, "subspace_angles.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nsaved: {output_path}")


if __name__ == "__main__":
    main()
