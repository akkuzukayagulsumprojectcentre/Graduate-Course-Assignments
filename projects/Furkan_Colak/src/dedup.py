"""Near-duplicate texts by word-shingle containment, shared by the build and leakage stages.

CISA posts some advisories under two addresses (an advisory code and a descriptive slug) or as an updated copy, so
two source ids can hold the same text. Splitting by source id alone would then put one text on both sides.
"""
import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

WORD = re.compile(r"\w+")


def shingles(text: str, n: int) -> set[str]:
    w = WORD.findall(text.lower())
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def index(ref: list[set]) -> dict[str, list[int]]:
    out = defaultdict(list)
    for j, s in enumerate(ref):
        for g in s:
            out[g].append(j)
    return out


def shared(s: set, idx: dict) -> Counter:
    return Counter(j for g in s for j in idx.get(g, ()))


def containment(query: list[set], ref: list[set]):
    """For each query text, the largest share of its shingles found in a single reference text, and that text."""
    idx = index(ref)
    best, arg = np.zeros(len(query)), np.full(len(query), -1)
    for i, s in enumerate(query):
        if s and (hits := shared(s, idx)):
            arg[i], k = hits.most_common(1)[0]
            best[i] = k / len(s)
    return best, arg


def dup_groups(human: pd.DataFrame, n: int, thr: float) -> dict[str, str]:
    """Reports joined when an excerpt of one covers at least `thr` of an excerpt of the other (union-find)."""
    sh = [shingles(t, n) for t in human.text]
    src = human.source_id.to_numpy()
    parent = {s: s for s in src}

    def root(s: str) -> str:
        while parent[s] != s:
            parent[s] = parent[parent[s]]
            s = parent[s]
        return s

    idx = index(sh)
    for i, s in enumerate(sh):
        for j, k in shared(s, idx).items():
            if src[j] != src[i] and s and k / len(s) >= thr:
                parent[root(src[i])] = root(src[j])
    return {s: root(s) for s in parent}


def purge(splits: dict, groups: dict[str, str]) -> tuple[dict, list[dict]]:
    """Keep the test splits as they are and remove their near-duplicates from training and validation.

    Test splits stay fixed because the test-only generators D and E were written for these excerpts only. A group
    split between training and validation alone is moved to training. Returns the new splits and a log of changes.
    """
    members = defaultdict(set)
    for s, g in groups.items():
        members[g].add(s)
    out, log = {}, []
    for seed, s in splits.items():
        test, val, train = set(s["test"]), set(s["val"]), set(s["train"])
        drop = {x for m in members.values() if m & test for x in m} - test
        to_train = {x for x in val - drop if (members[groups[x]] - {x}) & train}
        out[seed] = {"test": sorted(test), "val": sorted(val - drop - to_train), "train": sorted((train | to_train) - drop)}
        log += [{"seed": seed, "source_id": x, "from": "train" if x in train else "val", "action": "removed"}
                for x in sorted(drop)]
        log += [{"seed": seed, "source_id": x, "from": "val", "action": "moved to train"} for x in sorted(to_train)]
    return out, log

