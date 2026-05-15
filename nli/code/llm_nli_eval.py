import json
import os
import re
import requests
from tqdm import tqdm

def extract_label(text):
    t = text.strip().lower()
    if re.search(r'\bcontradiction\b|\bcontradicts\b|\bcontradictory\b', t):
        return 'contradiction'
    if re.search(r'\bentailment\b|\bentails\b|\bimplied\b', t):
        return 'entailment'
    if re.search(r'\bneutral\b|\bneither\b|\bunrelated\b', t):
        return 'neutral'
    return 'unknown'

def make_prompt(premise, hypothesis):
    return f"""You are an NLI classifier. Output EXACTLY one word: entailment, neutral, or contradiction. Nothing else.

Examples:
Premise: A man is playing guitar on stage.
Hypothesis: A musician is performing live.
Label: entailment

Premise: A woman is cooking dinner.
Hypothesis: The woman is eating at a restaurant.
Label: contradiction

Premise: Two children are playing outside.
Hypothesis: The children are siblings.
Label: neutral

Now classify:
Premise: {premise}
Hypothesis: {hypothesis}
Label:"""

def ask_llm(prompt):
    r = requests.post("http://localhost:11434/api/generate", json={
        "model": "llama3.2",
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0}
    })
    return r.json()["response"]

def load_snli(path):
    examples = []
    with open(path) as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 7:
                label, premise, hypothesis = parts[0], parts[5], parts[6]
                if label in ('entailment', 'neutral', 'contradiction'):
                    examples.append((premise, hypothesis, label))
    return examples

CACHE_FILE = "results/llm_preds.json"
DATA_PATH  = "data/snli_test/snli_1.0_dev.txt"

os.makedirs("results", exist_ok=True)
cache = json.load(open(CACHE_FILE)) if os.path.exists(CACHE_FILE) else {}

examples = load_snli(DATA_PATH)
print(f"Loaded {len(examples)} examples")

correct = 0
unknown = 0

for i, (premise, hypothesis, gold) in enumerate(tqdm(examples)):
    if str(i) not in cache:
        prompt = make_prompt(premise, hypothesis)
        raw = ask_llm(prompt)
        cache[str(i)] = {"pred": extract_label(raw), "gold": gold}
        if i % 100 == 0:
            json.dump(cache, open(CACHE_FILE, "w"))

    pred = cache[str(i)]["pred"]
    if pred == gold:
        correct += 1
    if pred == "unknown":
        unknown += 1

json.dump(cache, open(CACHE_FILE, "w"))

acc = correct / len(examples)
print(f"\nLlama 3.2 accuracy (few-shot): {acc:.1%}")
print(f"Unknown rate: {unknown/len(examples):.1%}")
print(f"Total examples: {len(examples)}")

with open("results/llm_accuracy.txt", "w") as f:
    f.write(f"Llama 3.2 few-shot accuracy (SNLI test): {acc:.1%}\n")
    f.write(f"LSTM baseline: 79.2%\n")