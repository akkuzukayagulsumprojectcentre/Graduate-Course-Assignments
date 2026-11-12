"""Detection agents, the coordinator and the LLM judge."""
import asyncio
import re

import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix, hstack
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
from transformers import set_seed as hf_set_seed
from xgboost import XGBClassifier

from src.features import ALL, GROUPS
from src.llm import make_client
from src.settings import DATA

JUDGE = DATA / "processed" / "judge.parquet"


# ---- detection agents: fit(train) -> predict(texts/features) -> P(LLM) ----

def tfidf(p: dict) -> FeatureUnion:
    return FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=tuple(p["word_ngrams"]), max_features=p["max_features"],
                                 sublinear_tf=True, min_df=2)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=tuple(p["char_ngrams"]),
                                 max_features=p["max_features"], sublinear_tf=True, min_df=2)),
    ])


class LexicalLR:
    """Baseline 1: TF-IDF + logistic regression."""

    def __init__(self, p: dict, seed: int, text_col: str = "text"):
        self.vec, self.col = tfidf(p), text_col
        self.clf = LogisticRegression(C=p["C"], max_iter=2000, random_state=seed)

    def fit(self, d: pd.DataFrame):
        self.clf.fit(self.vec.fit_transform(d[self.col]), d.label)
        return self

    def predict(self, d: pd.DataFrame) -> np.ndarray:
        return self.clf.predict_proba(self.vec.transform(d[self.col]))[:, 1]


class LexicalSVM(LexicalLR):
    """Baseline 2: TF-IDF + linear SVM, Platt-calibrated so it outputs probabilities."""

    def __init__(self, p: dict, seed: int, text_col: str = "text"):
        self.vec, self.col = tfidf(p), text_col
        self.clf = CalibratedClassifierCV(LinearSVC(C=p["C"], random_state=seed), cv=3)


class Stylometry:
    """Hand-crafted stylometric features + XGBoost."""

    def __init__(self, p: dict, seed: int, cols: list[str] = ALL):
        self.cols = cols
        self.clf = XGBClassifier(n_estimators=p["n_estimators"], max_depth=p["max_depth"],
                                 learning_rate=p["learning_rate"], random_state=seed, n_jobs=-1)

    def fit(self, d: pd.DataFrame):
        self.clf.fit(d[self.cols], d.label)
        return self

    def predict(self, d: pd.DataFrame) -> np.ndarray:
        return self.clf.predict_proba(d[self.cols])[:, 1]


class Combined(LexicalLR):
    """Ablation: TF-IDF and scaled stylometric features in one logistic regression."""

    def __init__(self, p: dict, seed: int):
        super().__init__(p, seed)
        self.scaler = StandardScaler()

    def _x(self, d, fit=False):
        f = d[ALL].fillna(0).to_numpy()
        f = self.scaler.fit_transform(f) if fit else self.scaler.transform(f)
        t = self.vec.fit_transform(d.text) if fit else self.vec.transform(d.text)
        return hstack([t, csr_matrix(f)]).tocsr()

    def fit(self, d):
        self.clf.fit(self._x(d, fit=True), d.label)
        return self

    def predict(self, d):
        return self.clf.predict_proba(self._x(d))[:, 1]


class Transformer:
    """Model 3: fine-tuned RoBERTa."""

    def __init__(self, p: dict, seed: int, text_col: str = "text"):
        self.p, self.seed, self.col = p, seed, text_col
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(p["model"])

    def _batches(self, texts, labels=None, shuffle=False):
        idx = np.random.default_rng(self.seed).permutation(len(texts)) if shuffle else np.arange(len(texts))
        for i in range(0, len(idx), self.p["batch_size"]):
            j = idx[i:i + self.p["batch_size"]]
            enc = self.tok([texts[k] for k in j], truncation=True, max_length=self.p["max_len"],
                           padding=True, return_tensors="pt").to(self.dev)
            if labels is not None:
                enc["labels"] = torch.tensor([labels[k] for k in j], device=self.dev)
            yield enc

    def fit(self, d: pd.DataFrame):
        hf_set_seed(self.seed)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.p["model"], num_labels=2).to(self.dev)
        texts, labels = d[self.col].tolist(), d.label.tolist()
        steps = self.p["epochs"] * int(np.ceil(len(texts) / self.p["batch_size"]))
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.p["lr"], weight_decay=0.01)
        sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
        scaler = torch.amp.GradScaler(enabled=self.p["fp16"])
        self.model.train()
        for _ in range(self.p["epochs"]):
            for batch in self._batches(texts, labels, shuffle=True):
                with torch.autocast(self.dev, dtype=torch.float16, enabled=self.p["fp16"]):
                    loss = self.model(**batch).loss
                opt.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                sched.step()
        return self

    @torch.no_grad()
    def predict(self, d: pd.DataFrame) -> np.ndarray:
        self.model.eval()
        out = []
        for batch in self._batches(d[self.col].tolist()):
            with torch.autocast(self.dev, dtype=torch.float16, enabled=self.p["fp16"]):
                logits = self.model(**batch).logits
            out.append(torch.softmax(logits.float(), -1)[:, 1].cpu().numpy())
        return np.concatenate(out)

    def free(self):
        del self.model
        torch.cuda.empty_cache()


class Coordinator:
    """Stacks member probabilities with a logistic regression fitted on the validation split only."""

    def __init__(self, members: list[str], seed: int):
        self.members = members
        self.clf = LogisticRegression(max_iter=1000, random_state=seed)

    def fit(self, val: pd.DataFrame):
        self.clf.fit(self._x(val), val.label)
        return self

    def _x(self, d):
        p = d[self.members].clip(1e-4, 1 - 1e-4).to_numpy()
        return np.log(p / (1 - p))

    def predict(self, d):
        return self.clf.predict_proba(self._x(d))[:, 1]


# ---- LLM judge: zero-shot, scores every text once, cached ----

JUDGE_PROMPT = (
    "You check cyber threat intelligence texts. Decide whether the text below was written by a human analyst "
    "or generated by an AI language model. Reply with one integer from 0 to 100: the probability that the "
    "text is AI-generated. Reply with the number only.\n\nText:\n{text}"
)


async def _judge(df: pd.DataFrame, spec: dict) -> None:
    done = set(pd.read_parquet(JUDGE).id) if JUDGE.exists() else set()
    todo = df[~df.id.isin(done)]
    print(f"[judge] {spec['model']}: {len(todo)} texts left")
    client = make_client({**spec, "concurrency": spec.get("concurrency", 3)})
    rows = []

    async def one(i, text):
        try:
            out = await client.chat("You are a careful analyst.", JUDGE_PROMPT.format(text=text), 0.0, 8)
            m = re.search(r"\d{1,3}", out)
            rows.append({"id": i, "llm_judge": min(int(m.group()), 100) / 100 if m else np.nan, "raw": out})
        except Exception as e:
            print(f"[judge] {i}: {e}")

    for start in range(0, len(todo), 200):  # flush to disk every 200 texts
        part = todo.iloc[start:start + 200]
        await asyncio.gather(*(one(i, t) for i, t in zip(part.id, part.text)))
        old = pd.read_parquet(JUDGE) if JUDGE.exists() else pd.DataFrame()
        pd.concat([old, pd.DataFrame(rows)]).to_parquet(JUDGE, index=False)
        rows.clear()
    await client.close()
