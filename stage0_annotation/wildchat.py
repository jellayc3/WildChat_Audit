"""Loading WildChat and segmenting conversations into scoreable turns.

Shared by both samplers so that the trend and event samples are segmented
identically; the only thing that differs between them is which conversations
get drawn.
"""
import os
import random

import numpy as np
import pandas as pd

DATASET_NAME = "allenai/WildChat-4.8M-Full"
DATASET_SPLIT = "train"


def set_seeds(seed):
    """Pin the Python and NumPy RNGs. The HuggingFace shuffle is seeded
    explicitly at the call site rather than relying on these."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_wildchat(seed):
    """Full non-streaming load, then a seeded shuffle. Both samplers start from
    exactly this ordering, and the event sampler skips conversation hashes the
    trend sampler already took, so the two samples cannot overlap."""
    from datasets import load_dataset
    set_seeds(seed)
    ds = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
    return ds.shuffle(seed=seed)


def has_assistant_turn(conv):
    for turn in conv.get("conversation") or []:
        if (turn.get("role") or "").lower() == "assistant":
            if str(turn.get("content") or "").strip():
                return True
    return False


def parse_ts(conv):
    """WildChat timestamps, normalised to UTC. Used for weekly bucketing."""
    return pd.to_datetime(conv.get("timestamp"), utc=True, errors="coerce")


def to_paired_turns(conversations):
    """One row per non-empty assistant turn, carrying the preceding user message
    as context. Nothing is truncated.

    row_id is WildChat's own `turn_identifier`, an integer, NOT a composite of
    conversation hash and index. Any later re-join of message text must key on
    row_id: `conversation_hash` is content-derived and collides across senders
    who open with an identical message.
    """
    rows = []
    for conv in conversations:
        conv_id = conv["conversation_hash"]
        last_user = ""
        idx = 0
        for turn in conv.get("conversation") or []:
            role = (turn.get("role") or "").lower()
            content = turn.get("content") or ""
            if role == "user":
                last_user = content
            elif role == "assistant" and str(content).strip():
                idx += 1
                rows.append({
                    "row_id": turn.get("turn_identifier") or f"{conv_id}_{idx}",
                    "ResponseId": conv_id,
                    "assistant_turn_index": idx,
                    "human_prev_message": last_user,
                    "assistant_message": content,
                    "wildchat_model": conv.get("model"),
                    "wildchat_language": conv.get("language"),
                    "wildchat_timestamp": conv.get("timestamp"),
                    "wildchat_toxic": conv.get("toxic"),
                    "wildchat_turn_count": conv.get("turn"),
                    "hashed_ip": conv.get("hashed_ip"),
                    "conversation_hash": conv_id,
                })
    return pd.DataFrame(rows)


def seen_hashes(*csv_paths):
    """Conversation hashes already drawn, so a later sample can exclude them."""
    seen = set()
    for p in csv_paths:
        if p and os.path.exists(p):
            seen |= set(pd.read_csv(p, usecols=["ResponseId"]).ResponseId.astype(str))
    return seen
