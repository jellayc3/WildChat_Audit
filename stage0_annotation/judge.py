"""The LLM judge: OpenRouter client, score parsing, batched resumable scoring.

One HTTP call per turn per criterion against google/gemma-3-12b-it at
temperature 0. No local model serving, no GPU. Progress is checkpointed after
every batch, so an interrupted run resumes rather than re-billing.
"""
import os
import re
import time
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests

MODEL = "google/gemma-3-12b-it"
BASE_URL = "https://openrouter.ai/api/v1"
N_WORKERS = 5
BATCH_SIZE = 100
REQUEST_TIMEOUT = 30
MAX_RETRIES = 3
THROTTLE_S = 0.1

# The judge deployed here is the 12B variant. Rathje et al. [2026] validated the
# 27B variant against 403 human annotators (r = 0.87-0.89 on validation). The
# paper states this mismatch as a limitation; do not describe the 12B scores as
# human-validated.
JUDGE_NOTE = "google/gemma-3-12b-it; validation in Rathje et al. covers the 27B variant"

_print_lock = threading.Lock()


def clean_model_name(model):
    """google/gemma-3-12b-it -> google_gemma_3_12b_it. The delivered score
    columns are f'{criterion}{SUF}' where SUF is '_' + this + '_score', and the
    criterion key keeps its hyphens, which is why one column is called
    'bot-positive-affirmation_google_gemma_3_12b_it_score'."""
    return re.sub(r"/|-", "_", model)


SUF = "_" + clean_model_name(MODEL) + "_score"


def api_key():
    k = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not k:
        raise SystemExit("Set OPENROUTER_API_KEY.")
    return k


# ------------------------------------------------------------------ parsing
def parse_score(txt, lo=0, hi=10):
    """First number in the reply, clipped to the scale. Blank or non-numeric
    replies become NaN rather than a default value."""
    if txt is None or not str(txt).strip():
        return np.nan
    if isinstance(txt, float) and np.isnan(txt):
        return np.nan
    m = re.search(r"([0-9]+\.?[0-9]*)", str(txt))
    if not m:
        return np.nan
    try:
        return float(np.clip(float(m.group(1)), lo, hi))
    except ValueError:
        return np.nan


def scale_for(criterion):
    """The holistic Anthropic rubric is 1-5; everything else is 0-10."""
    return (1, 5) if criterion == "sycophancy" else (0, 10)


# ------------------------------------------------------------------ transport
def call_openrouter(prompt, key, model=MODEL):
    url = f"{BASE_URL}/chat/completions"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "temperature": 0, "max_tokens": 200}
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.post(url, headers=headers, json=body, timeout=REQUEST_TIMEOUT)
            if r.status_code == 429:
                time.sleep(2 ** attempt * 2)
                continue
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except Exception:
            if attempt == MAX_RETRIES - 1:
                return None
            time.sleep(2 ** attempt)
    return None


# ------------------------------------------------------------------ driver
def run_criterion(turns, criterion, prompt_text, build_prompt, progress_dir, key=None):
    """Score `turns` (a DataFrame with human_prev_message / assistant_message)
    on one criterion, checkpointing to progress_dir after every batch."""
    key = key or api_key()
    lo, hi = scale_for(criterion)
    os.makedirs(progress_dir, exist_ok=True)
    ckpt = os.path.join(progress_dir, f"{criterion}.pkl")

    done = {}
    if os.path.exists(ckpt):
        with open(ckpt, "rb") as fh:
            done = pickle.load(fh)
        print(f"  {criterion}: resuming with {len(done):,} already scored")

    todo = [i for i in turns.index if i not in done]
    h = turns["human_prev_message"]
    a = turns["assistant_message"]

    for start in range(0, len(todo), BATCH_SIZE):
        batch = todo[start:start + BATCH_SIZE]

        def work(i):
            time.sleep(THROTTLE_S)
            p = build_prompt(h.loc[i], a.loc[i], prompt_text)
            return i, parse_score(call_openrouter(p, key), lo, hi)

        with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
            for i, score in ex.map(work, batch):
                done[i] = score

        with open(ckpt, "wb") as fh:
            pickle.dump(done, fh)
        with _print_lock:
            print(f"  {criterion}: {len(done):,} / {len(turns):,}", flush=True)

    return pd.Series(done, name=criterion + SUF).reindex(turns.index)


def score_all(turns, provenance, build_prompt, progress_dir, criteria=None):
    """Score every criterion in `provenance` (criterion -> (variant, text))."""
    key = api_key()
    out = {}
    for crit, (variant, text) in provenance.items():
        if criteria and crit not in criteria:
            continue
        print(f"{crit} [{variant} wording]")
        out[crit + SUF] = run_criterion(turns, crit, text, build_prompt,
                                        progress_dir, key)
    return pd.DataFrame(out, index=turns.index)
