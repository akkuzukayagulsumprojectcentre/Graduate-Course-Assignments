"""Near-duplicate texts by word-shingle containment (used by the leakage stage).

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

