import numpy as np


def compute_dom_direction(positive_activations, negative_activations):
    mean_difference = positive_activations.mean(axis=0) - negative_activations.mean(axis=0)
    direction_norm = np.linalg.norm(mean_difference) + 1e-8
    return mean_difference / direction_norm


def compute_actsvd_subspace(positive_activations, negative_activations, rank=4):
    sample_count = min(len(positive_activations), len(negative_activations))
    rng = np.random.RandomState(0)
    positive_indices = rng.choice(len(positive_activations), sample_count, replace=False)
    negative_indices = rng.choice(len(negative_activations), sample_count, replace=False)
    difference_matrix = positive_activations[positive_indices] - negative_activations[negative_indices]
    _, _, right_singular_vectors = np.linalg.svd(difference_matrix, full_matrices=False)
    subspace_basis = right_singular_vectors[:rank]
    row_norms = np.linalg.norm(subspace_basis, axis=1, keepdims=True) + 1e-8
    subspace_basis = subspace_basis / row_norms
    return subspace_basis


def project_out_direction(activations, unit_direction):
    projection_magnitudes = activations @ unit_direction
    projection_vectors = projection_magnitudes[..., None] * unit_direction[None, ...]
    return activations - projection_vectors


def project_out_subspace(activations, subspace_basis):
    coefficients = activations @ subspace_basis.T
    projection_vectors = coefficients @ subspace_basis
    return activations - projection_vectors


def principal_angles_between_subspaces(first_basis, second_basis):
    overlap_matrix = first_basis @ second_basis.T
    singular_values = np.linalg.svd(overlap_matrix, compute_uv=False)
    clipped_singular_values = np.clip(singular_values, -1.0, 1.0)
    angles_in_radians = np.arccos(clipped_singular_values)
    angles_in_degrees = np.degrees(angles_in_radians)
    mean_squared_cosine = float(np.mean(clipped_singular_values ** 2))
    return {
        "cos_principal_angles": clipped_singular_values.tolist(),
        "principal_angles_radians": angles_in_radians.tolist(),
        "principal_angles_degrees": angles_in_degrees.tolist(),
        "subspace_similarity_phi": mean_squared_cosine,
    }
