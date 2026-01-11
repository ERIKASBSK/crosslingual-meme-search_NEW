import os
import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer

MODEL = "intfloat/multilingual-e5-base"
CSV_PATH = "data/jpmemes.csv"


@st.cache_resource
def load_model():
    return SentenceTransformer(MODEL)


def _norm_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


@st.cache_data
def load_data_and_emb(csv_mtime: float):
    df = pd.read_csv(CSV_PATH)
    df["jp_text"] = df["jp_text"].fillna("").astype(str)
    df["source"] = df.get("source", pd.Series([""] * len(df))).fillna("").astype(str)

    texts = df["jp_text"].tolist()

    m = load_model()
    emb = m.encode([f"passage: {t}" for t in texts], batch_size=64, show_progress_bar=False)
    emb = _norm_rows(emb)

    return {
        "emb": emb,
        "jp_text": df["jp_text"].to_numpy(),
        "source": df["source"].to_numpy(),
    }


def qvec(m, q: str) -> np.ndarray:
    v = m.encode([f"query: {q}"])
    v = _norm_rows(v)
    return v[0]


def topk(qv: np.ndarray, emb: np.ndarray, k: int):
    s = emb @ qv
    k = max(1, min(int(k), len(s)))
    idx = np.argpartition(-s, kth=k - 1)[:k]
    idx = idx[np.argsort(-s[idx])]
    return idx, s[idx]


def junk(q: str) -> bool:
    if len(q) < 3:
        return True
    alpha_num = sum(ch.isalnum() for ch in q)
    return alpha_num < max(2, int(len(q) * 0.3))


st.set_page_config(page_title="EN → JP Meme Search", layout="wide")
st.title("EN → JP meme text search")
st.caption("English in, Japanese meme-ish text out. Simple.")

if not os.path.exists(CSV_PATH):
    st.error("Missing data/jpmemes.csv")
    st.stop()

mtime = os.path.getmtime(CSV_PATH)
data = load_data_and_emb(mtime)

with st.sidebar:
    k = st.slider("Results", 1, 20, 7)
    min_score = st.slider("Match bar", 0.0, 1.0, 0.72, 0.01)
    strict = st.checkbox("Ignore junk input", value=True)
    st.caption("Match bar: higher = stricter")

q = st.text_input("English", placeholder="cringe / no cap / that's so real / touch grass / delulu ...")
go = st.button("Search", type="primary")

if go:
    q = (q or "").strip()

    if strict and junk(q):
        st.warning("Try a clearer phrase.")
        st.stop()

    m = load_model()
    qv = qvec(m, q)
    idx, sc = topk(qv, data["emb"], k)

    hits = [(int(i), float(s)) for i, s in zip(idx, sc) if float(s) >= float(min_score)]
    if not hits:
        st.info("No good matches. Lower Match bar or try another query.")
        st.stop()

    for r, (i, s) in enumerate(hits, start=1):
        jp = data["jp_text"][i]
        tag = data["source"][i]

        st.markdown(f"### #{r} — {s * 100:.0f}% match")
        st.write(jp)
        if tag:
            st.caption(f"tag: {tag}")
        st.divider()
