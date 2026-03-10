import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from collections import defaultdict
from tqdm import tqdm
from sklearn.decomposition import PCA
import os
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

MODEL_ID = "google/gemma-2-2b-it"


def get_prompt_activations(prompts, model, tokenizer, device):
    """collect activations at last prompt token for each layer"""
    all_acts = []
    for prompt in tqdm(prompts, desc="collecting prompt activations"):
        text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        tokens = tokenizer(text, return_tensors="pt")
        if tokens.input_ids.shape[1] > 512:
            continue
        inputs = tokens.to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        acts = torch.stack([h[0, -1, :] for h in outputs.hidden_states])
        all_acts.append(acts.cpu().float().numpy())
    return np.array(all_acts)


def get_response_activations(pairs_list, key_high_or_low, model, tokenizer, device):
    """collect activations averaged over response tokens"""
    all_acts = []
    for pair in tqdm(pairs_list, desc=f"collecting {key_high_or_low} response activations"):
        prompt = pair['prompt']
        response = pair[f'response_{key_high_or_low}']
        prompt_only = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        prompt_len = tokenizer(prompt_only, return_tensors="pt").input_ids.shape[1]
        full_text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}, {"role": "assistant", "content": response}], tokenize=False, add_generation_prompt=False)
        tokens = tokenizer(full_text, return_tensors="pt")
        if tokens.input_ids.shape[1] > 512:
            continue
        inputs = tokens.to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        acts = torch.stack([h[0, prompt_len:, :].mean(dim=0) for h in outputs.hidden_states])
        all_acts.append(acts.cpu().float().numpy())
    return np.array(all_acts)


def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    print(f"\nloading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16, device_map="cuda", token=HF_TOKEN)
    model.eval()
    device = next(model.parameters()).device
    print(f"model device: {device}")
    print(f"layers: {model.config.num_hidden_layers + 1}")
    print(f"hidden size: {model.config.hidden_size}")

    print("\n" + "="*80)
    print("COLLECTING ALL ACTIVATIONS UPFRONT")
    print("="*80)

    print("\nloading AdvBench (harmful prompts)...")
    advbench = load_dataset("walledai/AdvBench", split="train")
    harmful_prompts = [x['prompt'] for x in advbench]
    print(f"  → {len(harmful_prompts)} harmful prompts available")

    print("loading Alpaca (harmless prompts)...")
    alpaca = load_dataset("tatsu-lab/alpaca", split="train", streaming=True)
    harmless_prompts = []
    for x in alpaca:
        if len(harmless_prompts) >= 2000:
            break
        if len(x['instruction']) > 10 and len(x['output']) > 10:
            harmless_prompts.append(x['instruction'])
    print(f"  → {len(harmless_prompts)} harmless prompts loaded (subset)")

    all_harmful_acts = get_prompt_activations(harmful_prompts, model, tokenizer, device)
    all_harmless_acts = get_prompt_activations(harmless_prompts, model, tokenizer, device)
    print(f"collected harmful:  {all_harmful_acts.shape}")
    print(f"collected harmless: {all_harmless_acts.shape}")

    print("\nloading HelpSteer...")
    helpsteer = load_dataset("nvidia/HelpSteer", split="train")
    prompt_to_responses = defaultdict(list)
    for row in helpsteer:
        prompt_to_responses[row['prompt']].append({'response': row['response'], 'helpfulness': row['helpfulness']})

    pairs = []
    for prompt, responses in prompt_to_responses.items():
        if len(responses) < 2:
            continue
        sorted_r = sorted(responses, key=lambda x: x['helpfulness'])
        low, high = sorted_r[0], sorted_r[-1]
        delta = high['helpfulness'] - low['helpfulness']
        if delta == 0:
            continue
        pairs.append({'prompt': prompt, 'response_high': high['response'], 'response_low': low['response'], 'helpfulness_high': high['helpfulness'], 'helpfulness_low': low['helpfulness'], 'delta': delta})

    print(f"total usable helpsteer pairs: {len(pairs)}")
    pairs_delta1 = [p for p in pairs if p['delta'] == 1]
    pairs_delta2 = [p for p in pairs if p['delta'] == 2]
    pairs_delta3 = [p for p in pairs if p['delta'] == 3]
    pairs_delta4 = [p for p in pairs if p['delta'] == 4]
    print(f"delta=1 pairs: {len(pairs_delta1)}")
    print(f"delta=2 pairs: {len(pairs_delta2)}")
    print(f"delta=3 pairs: {len(pairs_delta3)}")
    print(f"delta=4 pairs: {len(pairs_delta4)}")

    all_high_acts_delta1 = get_response_activations(pairs_delta1, "high", model, tokenizer, device)
    all_low_acts_delta1 = get_response_activations(pairs_delta1, "low", model, tokenizer, device)
    all_high_acts_delta2 = get_response_activations(pairs_delta2, "high", model, tokenizer, device)
    all_low_acts_delta2 = get_response_activations(pairs_delta2, "low", model, tokenizer, device)
    all_high_acts_delta3 = get_response_activations(pairs_delta3, "high", model, tokenizer, device)
    all_low_acts_delta3 = get_response_activations(pairs_delta3, "low", model, tokenizer, device)
    all_high_acts_delta4 = get_response_activations(pairs_delta4, "high", model, tokenizer, device)
    all_low_acts_delta4 = get_response_activations(pairs_delta4, "low", model, tokenizer, device)

    print(f"collected delta=1 high/low: {all_high_acts_delta1.shape} / {all_low_acts_delta1.shape}")
    print(f"collected delta=2 high/low: {all_high_acts_delta2.shape} / {all_low_acts_delta2.shape}")
    print(f"collected delta=3 high/low: {all_high_acts_delta3.shape} / {all_low_acts_delta3.shape}")
    print(f"collected delta=4 high/low: {all_high_acts_delta4.shape} / {all_low_acts_delta4.shape}")

    print("\n" + "="*80)
    print("EDA TASK 7: Layer Selection via Safety/Utility Divergence")
    print("="*80)

    N_TASK7 = 100  # use all available harmful (520) and harmless (2000), sample 100 from helpsteer
    harmful_sample = all_harmful_acts[np.random.choice(len(all_harmful_acts), min(N_TASK7, len(all_harmful_acts)), replace=False)]
    harmless_sample = all_harmless_acts[np.random.choice(len(all_harmless_acts), min(N_TASK7, len(all_harmless_acts)), replace=False)]
    
    # for utility — use ONLY delta>=3 here for the cleanest signal
    all_high_combined = np.vstack([all_high_acts_delta1, all_high_acts_delta2, all_high_acts_delta3, all_high_acts_delta4])
    all_low_combined = np.vstack([all_low_acts_delta1, all_low_acts_delta2, all_low_acts_delta3, all_low_acts_delta4])
    all_high_delta3plus = np.vstack([all_high_acts_delta3, all_high_acts_delta4])
    all_low_delta3plus = np.vstack([all_low_acts_delta3, all_low_acts_delta4])
    high_sample = all_high_delta3plus[np.random.choice(len(all_high_delta3plus), min(N_TASK7, len(all_high_delta3plus)), replace=False)]
    low_sample = all_low_delta3plus[np.random.choice(len(all_low_delta3plus), min(N_TASK7, len(all_low_delta3plus)), replace=False)]

    n_layers = all_harmful_acts.shape[1]
    r_s_norm, r_u_norm, cos_su, divergence_score = np.zeros(n_layers), np.zeros(n_layers), np.zeros(n_layers), np.zeros(n_layers)

    for l in range(n_layers):
        r_s = harmful_sample[:, l, :].mean(axis=0) - harmless_sample[:, l, :].mean(axis=0)
        r_u = high_sample[:, l, :].mean(axis=0) - low_sample[:, l, :].mean(axis=0)
        r_s_norm[l] = np.linalg.norm(r_s)
        r_u_norm[l] = np.linalg.norm(r_u)
        cos_su[l] = cosine_sim(r_s, r_u)
        divergence_score[l] = (r_s_norm[l] * r_u_norm[l]) / (abs(cos_su[l]) + 1e-8)

    max_div_idx = np.argmax(divergence_score)
    EXTRACTION_LAYER = max_div_idx
    print(f"\nRECOMMENDED EXTRACTION LAYER: {EXTRACTION_LAYER}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    layers = np.arange(n_layers)
    r_s_norm_scaled = r_s_norm / np.max(r_s_norm)
    r_u_norm_scaled = r_u_norm / np.max(r_u_norm)
    abs_cos_su = np.abs(cos_su)

    axes[0].plot(layers, r_s_norm_scaled, marker='o', linewidth=2, label='safety norm (normalized)', color='red')
    axes[0].plot(layers, r_u_norm_scaled, marker='s', linewidth=2, label='utility norm (normalized)', color='blue')
    axes[0].plot(layers, abs_cos_su, marker='^', linewidth=2, linestyle='--', label='|cos(r_s, r_u)|', color='purple')
    axes[0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[0].set_xlabel("layer")
    axes[0].set_ylabel("normalized value / |cosine|")
    axes[0].set_title("Safety vs Utility Direction Properties Across Layers")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    min_cos_idx = np.argmin(abs_cos_su)
    axes[1].fill_between(layers, cos_su, 0, where=(cos_su > 0), alpha=0.3, color='red', label='positive (entangled)')
    axes[1].fill_between(layers, cos_su, 0, where=(cos_su <= 0), alpha=0.3, color='blue', label='negative (orthogonal)')
    axes[1].plot(layers, cos_su, linewidth=2, color='black')
    axes[1].axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    axes[1].axhline(y=1, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    axes[1].axhline(y=-1, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    axes[1].scatter([min_cos_idx], [cos_su[min_cos_idx]], marker='*', s=500, color='gold', edgecolors='black', linewidth=2, zorder=5)
    axes[1].text(min_cos_idx + 0.5, cos_su[min_cos_idx], f"L={min_cos_idx}", fontsize=10, fontweight='bold')
    axes[1].set_xlabel("layer")
    axes[1].set_ylabel("cos(r_s, r_u)")
    axes[1].set_title(f"cos(r_s, r_u) Across Layers\n(mean |cos| = {np.mean(abs_cos_su):.3f})")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    top_3_idx = np.argsort(divergence_score)[-3:]
    axes[2].plot(layers, divergence_score, linewidth=2, color='black', marker='o')
    axes[2].fill_between(layers[top_3_idx], divergence_score[top_3_idx], alpha=0.3, color='green')
    axes[2].scatter([max_div_idx], [divergence_score[max_div_idx]], marker='*', s=500, color='gold', edgecolors='black', linewidth=2, zorder=5)
    axes[2].text(max_div_idx + 0.5, divergence_score[max_div_idx], f"L={max_div_idx}", fontsize=10, fontweight='bold')
    axes[2].set_xlabel("layer")
    axes[2].set_ylabel("divergence score")
    axes[2].set_title(f"Safety-Utility Divergence Score Per Layer\n(recommended extraction layer: L={max_div_idx})")
    axes[2].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("eda_layer_selection.png", dpi=150, bbox_inches='tight')
    print("saved: eda_layer_selection.png")

    print("\n--- layer selection analysis ---")
    print(f"layer with min |cos(r_s, r_u)|: {min_cos_idx}  (value: {abs_cos_su[min_cos_idx]:.4f})")
    print(f"layer with max safety norm:     {np.argmax(r_s_norm)}  (value: {np.max(r_s_norm):.4f})")
    print(f"layer with max utility norm:    {np.argmax(r_u_norm)}  (value: {np.max(r_u_norm):.4f})")
    print(f"layer with max divergence score: {max_div_idx} (value: {divergence_score[max_div_idx]:.4f})  <- recommended")
    print(f"mean |cos(r_s, r_u)| across all layers: {np.mean(abs_cos_su):.4f}")

    print("\n" + "="*80)
    print("EDA TASK 3: HelpSteer Delta Threshold Comparison")
    print("="*80)

    N_TASK3 = min(50, len(pairs_delta4))  # use however many delta=4 pairs are available, cap at 50
    print(f"using N_TASK3 = {N_TASK3} per group")
    delta_groups = {1: (all_high_acts_delta1, all_low_acts_delta1), 2: (all_high_acts_delta2, all_low_acts_delta2), 3: (all_high_acts_delta3, all_low_acts_delta3), 4: (all_high_acts_delta4, all_low_acts_delta4)}
    task3_data = {}
    for d in [1, 2, 3, 4]:
        high_acts, low_acts = delta_groups[d]
        if len(high_acts) < N_TASK3 or len(low_acts) < N_TASK3:
            print(f"WARNING: delta={d} has insufficient pairs ({len(high_acts)} high, {len(low_acts)} low), skipping")
            continue
        high_sampled = high_acts[np.random.choice(len(high_acts), N_TASK3, replace=False)]
        low_sampled = low_acts[np.random.choice(len(low_acts), N_TASK3, replace=False)]
        task3_data[d] = {'high': high_sampled, 'low': low_sampled}

    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    all_vectors_task3, colors_task3, group_means = [], [], {d: {'high': None, 'low': None} for d in task3_data.keys()}

    for d in task3_data.keys():
        all_vectors_task3.extend(task3_data[d]['high'][:, EXTRACTION_LAYER, :])
        colors_task3.extend(['#ADD8E6'] * N_TASK3)
        all_vectors_task3.extend(task3_data[d]['low'][:, EXTRACTION_LAYER, :])
        colors_task3.extend(['#FFA07A'] * N_TASK3)
        group_means[d]['high'] = task3_data[d]['high'][:, EXTRACTION_LAYER, :].mean(axis=0)
        group_means[d]['low'] = task3_data[d]['low'][:, EXTRACTION_LAYER, :].mean(axis=0)

    all_vectors_task3, colors_task3 = np.array(all_vectors_task3), np.array(colors_task3)
    pca = PCA(n_components=2)
    vecs_2d = pca.fit_transform(all_vectors_task3)
    for i, (vec, color) in enumerate(zip(vecs_2d, colors_task3)):
        axes[0].scatter(vec[0], vec[1], c=color, s=50, alpha=0.6, edgecolors='black', linewidth=0.5)

    for d in task3_data.keys():
        mean_high_2d = pca.transform([group_means[d]['high']])[0]
        mean_low_2d = pca.transform([group_means[d]['low']])[0]
        axes[0].scatter(mean_high_2d[0], mean_high_2d[1], marker='X', s=300, c='#00008B', edgecolors='black', linewidth=1.5)
        axes[0].scatter(mean_low_2d[0], mean_low_2d[1], marker='X', s=300, c='#8B0000', edgecolors='black', linewidth=1.5)

    axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    axes[0].set_title(f"PCA of High vs Low Utility Activations at Layer {EXTRACTION_LAYER}")
    axes[0].legend([plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#ADD8E6', markersize=8, markeredgecolor='black'),
                    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#FFA07A', markersize=8, markeredgecolor='black'),
                    plt.Line2D([0], [0], marker='X', color='w', markerfacecolor='#00008B', markersize=10, markeredgecolor='black'),
                    plt.Line2D([0], [0], marker='X', color='w', markerfacecolor='#8B0000', markersize=10, markeredgecolor='black')],
                   ['High', 'Low', 'High Mean', 'Low Mean'], loc='best')

    deltas = sorted(task3_data.keys())
    mean_cosines, std_cosines = [], []
    for d in deltas:
        cosines = [cosine_sim(task3_data[d]['high'][i, EXTRACTION_LAYER, :], task3_data[d]['low'][i, EXTRACTION_LAYER, :]) for i in range(N_TASK3)]
        mean_cosines.append(np.mean(cosines))
        std_cosines.append(np.std(cosines))

    axes[1].bar(range(len(deltas)), mean_cosines, yerr=std_cosines, capsize=5, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'])
    axes[1].set_xticks(range(len(deltas)))
    axes[1].set_xticklabels([f"delta={d}" for d in deltas])
    axes[1].set_ylabel("Mean Cosine Similarity")
    axes[1].set_title("Mean Cosine Similarity: High vs Low\n(lower = more separated = better)")
    axes[1].grid(True, alpha=0.3, axis='y')

    direction_norms = [np.linalg.norm(group_means[d]['high'] - group_means[d]['low']) for d in deltas]
    axes[2].bar(range(len(deltas)), direction_norms, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'])
    axes[2].set_xticks(range(len(deltas)))
    axes[2].set_xticklabels([f"delta={d}" for d in deltas])
    axes[2].set_ylabel("Direction Norm ||r_u||")
    axes[2].set_title("Utility Direction Norm\n(higher = stronger signal)")
    axes[2].grid(True, alpha=0.3, axis='y')

    # panel 3: delta=4 only PCA
    if 4 in task3_data:
        high4 = task3_data[4]['high'][:, EXTRACTION_LAYER, :]
        low4 = task3_data[4]['low'][:, EXTRACTION_LAYER, :]
        vecs4 = np.vstack([high4, low4])
        labels4 = ['high'] * len(high4) + ['low'] * len(low4)

        pca4 = PCA(n_components=2)
        vecs4_2d = pca4.fit_transform(vecs4)

        axes[3].scatter(vecs4_2d[:len(high4), 0], vecs4_2d[:len(high4), 1],
                        c='#ADD8E6', s=60, alpha=0.7, edgecolors='black', linewidth=0.5, label='high (δ=4)')
        axes[3].scatter(vecs4_2d[len(high4):, 0], vecs4_2d[len(high4):, 1],
                        c='#FFA07A', s=60, alpha=0.7, edgecolors='black', linewidth=0.5, label='low (δ=4)')

        mean_high4_2d = pca4.transform([high4.mean(axis=0)])[0]
        mean_low4_2d = pca4.transform([low4.mean(axis=0)])[0]
        axes[3].scatter(*mean_high4_2d, marker='X', s=300, c='#00008B', edgecolors='black', linewidth=1.5)
        axes[3].scatter(*mean_low4_2d, marker='X', s=300, c='#8B0000', edgecolors='black', linewidth=1.5)

        axes[3].set_xlabel(f"PC1 ({pca4.explained_variance_ratio_[0]:.1%})")
        axes[3].set_ylabel(f"PC2 ({pca4.explained_variance_ratio_[1]:.1%})")
        axes[3].set_title(f"Delta=4 Only PCA at Layer {EXTRACTION_LAYER}\n(isolated — no noise from other delta groups)")
        axes[3].legend(loc='best')
        axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("eda_helpsteer_delta.png", dpi=150, bbox_inches='tight')
    print("saved: eda_helpsteer_delta.png")

    print(f"\n--- delta threshold analysis (layer {EXTRACTION_LAYER}) ---")
    for i, d in enumerate(deltas):
        print(f"delta={d}: N available = {len(task3_data[d]['high'])}, mean cosine sim = {mean_cosines[i]:.4f} ± {std_cosines[i]:.4f}, direction norm = {direction_norms[i]:.4f}")

    print("\n" + "="*80)
    print("EDA TASK 8: Direction Stability vs Sample Size")
    print("="*80)

    sample_sizes = [32, 64, 128, 256, 400]
    n_bootstrap = 10

    # three utility pools to compare
    all_high_geq3 = np.vstack([all_high_acts_delta3, all_high_acts_delta4])
    all_low_geq3 = np.vstack([all_low_acts_delta3, all_low_acts_delta4])
    all_high_eq4 = all_high_acts_delta4
    all_low_eq4 = all_low_acts_delta4

    print(f"utility pools: combined={len(all_high_combined)}, delta>=3={len(all_high_geq3)}, delta=4={len(all_high_eq4)}")

    safety_stability = {}
    utility_stab_all = {}
    utility_stab_geq3 = {}
    utility_stab_eq4 = {}

    for N in sample_sizes:
        safety_dirs = []
        udirs_all, udirs_geq3, udirs_eq4 = [], [], []

        for trial in range(n_bootstrap):
            # safety direction
            idx_h = np.random.choice(len(all_harmful_acts), N, replace=False)
            idx_hl = np.random.choice(len(all_harmless_acts), N, replace=False)
            r_s = all_harmful_acts[idx_h, EXTRACTION_LAYER, :].mean(axis=0) - all_harmless_acts[idx_hl, EXTRACTION_LAYER, :].mean(axis=0)
            safety_dirs.append(r_s / (np.linalg.norm(r_s) + 1e-8))

            # utility — all deltas combined
            n_util_all = min(N, len(all_high_combined))
            idx_hi = np.random.choice(len(all_high_combined), n_util_all, replace=False)
            idx_lo = np.random.choice(len(all_low_combined), n_util_all, replace=False)
            r_u = all_high_combined[idx_hi, EXTRACTION_LAYER, :].mean(axis=0) - all_low_combined[idx_lo, EXTRACTION_LAYER, :].mean(axis=0)
            udirs_all.append(r_u / (np.linalg.norm(r_u) + 1e-8))

            # utility — delta >= 3 only
            n_util_geq3 = min(N, len(all_high_geq3))
            idx_hi3 = np.random.choice(len(all_high_geq3), n_util_geq3, replace=False)
            idx_lo3 = np.random.choice(len(all_low_geq3), n_util_geq3, replace=False)
            r_u3 = all_high_geq3[idx_hi3, EXTRACTION_LAYER, :].mean(axis=0) - all_low_geq3[idx_lo3, EXTRACTION_LAYER, :].mean(axis=0)
            udirs_geq3.append(r_u3 / (np.linalg.norm(r_u3) + 1e-8))

            # utility — delta = 4 only (may not have enough for large N, sample with replace if needed)
            n_util_eq4 = min(N, len(all_high_eq4))
            replace_eq4 = n_util_eq4 < N  # flag if we had to use replacement
            idx_hi4 = np.random.choice(len(all_high_eq4), n_util_eq4, replace=replace_eq4)
            idx_lo4 = np.random.choice(len(all_low_eq4), n_util_eq4, replace=replace_eq4)
            r_u4 = all_high_eq4[idx_hi4, EXTRACTION_LAYER, :].mean(axis=0) - all_low_eq4[idx_lo4, EXTRACTION_LAYER, :].mean(axis=0)
            udirs_eq4.append(r_u4 / (np.linalg.norm(r_u4) + 1e-8))

        # pairwise cosine similarity across the 10 bootstrap trials
        def pairwise_cos_stats(dirs):
            sims = [cosine_sim(dirs[i], dirs[j]) for i in range(n_bootstrap) for j in range(i+1, n_bootstrap)]
            return np.mean(sims), np.std(sims)

        safety_stability[N] = pairwise_cos_stats(safety_dirs)
        utility_stab_all[N] = pairwise_cos_stats(udirs_all)
        utility_stab_geq3[N] = pairwise_cos_stats(udirs_geq3)
        utility_stab_eq4[N] = pairwise_cos_stats(udirs_eq4)

    fig, axes = plt.subplots(1, 3, figsize=(21, 5))
    ns = sorted(sample_sizes)

    # panel 0: stability curves for safety + 3 utility variants
    def plot_stability_line(ax, stab_dict, ns, color, label, marker):
        means = [stab_dict[n][0] for n in ns]
        stds = [stab_dict[n][1] for n in ns]
        ax.errorbar(ns, means, yerr=stds, marker=marker, linewidth=2, markersize=8, label=label, color=color)
        return means

    safety_means = plot_stability_line(axes[0], safety_stability, ns, 'red', 'safety', 'o')
    util_all_means = plot_stability_line(axes[0], utility_stab_all, ns, 'blue', 'utility (all δ)', 's')
    util_3_means = plot_stability_line(axes[0], utility_stab_geq3, ns, 'green', 'utility (δ≥3)', '^')
    util_4_means = plot_stability_line(axes[0], utility_stab_eq4, ns, 'orange', 'utility (δ=4)', 'D')

    axes[0].axhline(y=0.95, color='gray', linestyle='--', linewidth=1.5, label='stable threshold (0.95)')
    axes[0].set_xlabel("sample size N")
    axes[0].set_ylabel("mean pairwise cosine similarity")
    axes[0].set_ylim([0.0, 1.01])
    axes[0].set_xticks(ns)
    axes[0].set_title("Direction Stability vs Sample Size\n(safety vs utility — all delta groupings)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # panel 1: bootstrap PCA for N=32 vs N=400 — shows spread shrinking
    def bootstrap_dirs(acts_h, acts_hl, N, layer, n_trials=10):
        dirs = []
        for _ in range(n_trials):
            n_actual = min(N, len(acts_h), len(acts_hl))
            ih = np.random.choice(len(acts_h), n_actual, replace=False)
            ihl = np.random.choice(len(acts_hl), n_actual, replace=False)
            r = acts_h[ih, layer, :].mean(axis=0) - acts_hl[ihl, layer, :].mean(axis=0)
            dirs.append(r / (np.linalg.norm(r) + 1e-8))
        return np.array(dirs)

    dirs_32 = bootstrap_dirs(all_harmful_acts, all_harmless_acts, 32, EXTRACTION_LAYER)
    dirs_400 = bootstrap_dirs(all_harmful_acts, all_harmless_acts, 400, EXTRACTION_LAYER)

    # fit PCA on all 20 directions together so both are in the same space
    all_dirs = np.vstack([dirs_32, dirs_400])
    pca_stab = PCA(n_components=2)
    all_2d = pca_stab.fit_transform(all_dirs)
    dirs_32_2d = all_2d[:10]
    dirs_400_2d = all_2d[10:]

    axes[1].scatter(dirs_32_2d[:, 0], dirs_32_2d[:, 1], s=120, color='tomato', edgecolors='black', linewidth=1.5, alpha=0.8, label='N=32  (unstable)')
    axes[1].scatter(dirs_400_2d[:, 0], dirs_400_2d[:, 1], s=120, color='steelblue', edgecolors='black', linewidth=1.5, alpha=0.8, label='N=400 (stable)')
    axes[1].set_xlabel(f"PC1 ({pca_stab.explained_variance_ratio_[0]:.1%})")
    axes[1].set_ylabel(f"PC2 ({pca_stab.explained_variance_ratio_[1]:.1%})")
    axes[1].set_title(f"Safety Direction Variance: N=32 vs N=400\n(same PCA space — tighter = more stable, layer {EXTRACTION_LAYER})")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # panel 2: same thing but for utility at delta>=3
    dirs_u_32 = bootstrap_dirs(all_high_geq3, all_low_geq3, 32, EXTRACTION_LAYER)
    dirs_u_400 = bootstrap_dirs(all_high_geq3, all_low_geq3, min(400, len(all_high_geq3)), EXTRACTION_LAYER)

    all_udirs = np.vstack([dirs_u_32, dirs_u_400])
    pca_ustab = PCA(n_components=2)
    all_u2d = pca_ustab.fit_transform(all_udirs)
    udirs_32_2d = all_u2d[:10]
    udirs_400_2d = all_u2d[10:]

    axes[2].scatter(udirs_32_2d[:, 0], udirs_32_2d[:, 1], s=120, color='tomato', edgecolors='black', linewidth=1.5, alpha=0.8, label='N=32')
    axes[2].scatter(udirs_400_2d[:, 0], udirs_400_2d[:, 1], s=120, color='steelblue', edgecolors='black', linewidth=1.5, alpha=0.8, label=f'N={min(400, len(all_high_geq3))}')
    axes[2].set_xlabel(f"PC1 ({pca_ustab.explained_variance_ratio_[0]:.1%})")
    axes[2].set_ylabel(f"PC2 ({pca_ustab.explained_variance_ratio_[1]:.1%})")
    axes[2].set_title(f"Utility Direction Variance: N=32 vs N=max (δ≥3)\n(same PCA space — tighter = more stable, layer {EXTRACTION_LAYER})")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("eda_stability.png", dpi=150, bbox_inches='tight')
    print("saved: eda_stability.png")

    print(f"\n=== direction stability analysis (layer {EXTRACTION_LAYER}) ===")
    print(f"{'sample size':>12} | {'safety':^20} | {'util (all δ)':^20} | {'util (δ≥3)':^20} | {'util (δ=4)':^20}")
    print("-" * 97)
    for n in ns:
        s_mean, s_std = safety_stability[n]
        ua_mean, ua_std = utility_stab_all[n]
        u3_mean, u3_std = utility_stab_geq3[n]
        u4_mean, u4_std = utility_stab_eq4[n]
        print(f"{n:>12} | {s_mean:.4f} ± {s_std:.4f}    | {ua_mean:.4f} ± {ua_std:.4f}    | {u3_mean:.4f} ± {u3_std:.4f}    | {u4_mean:.4f} ± {u4_std:.4f}")

    safety_cross = next((n for n in ns if safety_stability[n][0] > 0.95), None)
    util_all_cross = next((n for n in ns if utility_stab_all[n][0] > 0.95), None)
    util_geq3_cross = next((n for n in ns if utility_stab_geq3[n][0] > 0.95), None)
    util_eq4_cross = next((n for n in ns if utility_stab_eq4[n][0] > 0.95), None)

    if safety_cross:
        print(f"\nsafety direction stabilizes at N = {safety_cross}   (stability > 0.95)")
    if util_all_cross:
        print(f"utility (all δ) stabilizes at N = {util_all_cross}")
    if util_geq3_cross:
        print(f"utility (δ≥3) stabilizes at N = {util_geq3_cross}")
    if util_eq4_cross:
        print(f"utility (δ=4) stabilizes at N = {util_eq4_cross}")

    print(f"\nrecommendation: use N=400")
    print("\n" + "="*80)
    print("ALL TASKS COMPLETE")
    print("="*80)
