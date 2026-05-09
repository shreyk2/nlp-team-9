# shared utilities: model loading, activation extraction, dataset loaders
import os
from collections import defaultdict

import numpy as np
import torch
from datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")


# lock all RNGs for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# load a causal LM and return model, tokenizer, device info
def load_model(model_id, dtype=torch.float16):
    print(f"loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto",
        token=HF_TOKEN,
    )
    model.eval()
    device = next(model.parameters()).device
    num_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size
    print(f"  device={device} | layers={num_layers} | hidden={hidden_size}")
    return model, tokenizer, device


# cosine sim between two vectors
def cosine_similarity(vector_a, vector_b):
    dot_product = np.dot(vector_a, vector_b)
    norm_product = np.linalg.norm(vector_a) * np.linalg.norm(vector_b) + 1e-8
    return dot_product / norm_product


# extract last token hidden states for a list of prompts
def get_prompt_activations(prompts, model, tokenizer, device, max_length=512, layer=None):
    activations = []
    for prompt in tqdm(prompts, desc="prompt activations"):
        formatted_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        tokens = tokenizer(formatted_text, return_tensors="pt")
        if tokens.input_ids.shape[1] > max_length:
            continue
        inputs = tokens.to(device)
        with torch.no_grad():
            output = model(**inputs, output_hidden_states=True)
        # grab last token activation at specified layer (or all layers)
        if layer is None:
            per_layer_activations = []
            for hidden_state in output.hidden_states:
                per_layer_activations.append(hidden_state[0, -1, :])
            activation = torch.stack(per_layer_activations)
        else:
            activation = output.hidden_states[layer][0, -1, :]
        activations.append(activation.cpu().float().numpy())
    return np.array(activations)


# mean-pool response tokens excluding the prompt for paired data
def get_response_activations(pairs, key, model, tokenizer, device, max_length=512, layer=None):
    activations = []
    for pair in tqdm(pairs, desc=f"response activations ({key})"):
        prompt_text = pair["prompt"]
        response_text = pair[f"response_{key}"]
        prompt_only_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            tokenize=False,
            add_generation_prompt=True,
        )
        # figure out where the prompt ends so we only pool response tokens
        prompt_token_count = tokenizer(prompt_only_text, return_tensors="pt").input_ids.shape[1]
        full_text = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": prompt_text},
                {"role": "assistant", "content": response_text},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )
        tokens = tokenizer(full_text, return_tensors="pt")
        if tokens.input_ids.shape[1] > max_length:
            continue
        inputs = tokens.to(device)
        with torch.no_grad():
            output = model(**inputs, output_hidden_states=True)
        if layer is None:
            per_layer_activations = []
            for hidden_state in output.hidden_states:
                response_mean = hidden_state[0, prompt_token_count:, :].mean(dim=0)
                per_layer_activations.append(response_mean)
            activation = torch.stack(per_layer_activations)
        else:
            activation = output.hidden_states[layer][0, prompt_token_count:, :].mean(dim=0)
        activations.append(activation.cpu().float().numpy())
    return np.array(activations)


# load harmful prompts from AdvBench
def load_advbench_prompts():
    dataset = load_dataset("walledai/AdvBench", split="train")
    prompts = []
    for row in dataset:
        prompts.append(row["prompt"])
    return prompts


# load benign instruction prompts from Alpaca
def load_alpaca_prompts(max_count=2000):
    dataset = load_dataset("tatsu-lab/alpaca", split="train", streaming=True)
    prompts = []
    for row in dataset:
        if len(prompts) >= max_count:
            break
        if len(row["instruction"]) > 10 and len(row["output"]) > 10:
            prompts.append(row["instruction"])
    return prompts


# build high/low helpfulness pairs from HelpSteer for utility direction
def load_helpsteer_pairs(min_helpfulness_delta=1):
    dataset = load_dataset("nvidia/HelpSteer", split="train")
    grouped_by_prompt = defaultdict(list)
    for row in dataset:
        grouped_by_prompt[row["prompt"]].append({
            "response": row["response"],
            "helpfulness": row["helpfulness"],
        })
    pairs = []
    for prompt_text, responses in grouped_by_prompt.items():
        if len(responses) < 2:
            continue
        responses_sorted = sorted(responses, key=lambda x: x["helpfulness"])
        lowest = responses_sorted[0]
        highest = responses_sorted[-1]
        delta = highest["helpfulness"] - lowest["helpfulness"]
        if delta < min_helpfulness_delta:
            continue
        pairs.append({
            "prompt": prompt_text,
            "response_high": highest["response"],
            "response_low": lowest["response"],
            "helpfulness_high": highest["helpfulness"],
            "helpfulness_low": lowest["helpfulness"],
            "delta": delta,
        })
    return pairs
