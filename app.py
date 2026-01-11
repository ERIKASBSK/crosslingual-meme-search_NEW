import os
import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer

MODEL = "intfloat/multilingual-e5-base"
CSV_PATH = "data/jpmemes.csv"
EPS = 1e-12


@st.cache_resource
def load_model():
    return SentenceTransformer(MODEL)


def norm_rows(x):
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (n + EPS)


@st.cache_data
def load_data_and_emb(csv_mtime):
    df = pd.read_csv(CSV_PATH)

    df["jp_text"] = df["jp_text"].fillna("").astype(str)
    df["source"] = df.get("source", pd.Series([""] * len(df))).fillna("").astype(str)  # keep this as-is

    texts = df["jp_text"].tolist()

    model = load_model()
    emb = model.encode([f"passage: {t}" for t in texts], batch_size=64, show_progress_bar=False)
    emb = norm_rows(emb)

    return {
        "emb": emb,
        "jp_text": df["jp_text"].to_numpy(),
        "source": df["source"].to_numpy(),
    }


def embed_query(model, q):
    v = model.encode([f"query: {q}"])
    v = norm_rows(v)
    return v[0]


def search_topk(query_vec, emb, k):
    scores = emb @ query_vec
    k = max(1, min(int(k), len(scores)))

    idx = np.argpartition(-scores, kth=k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return idx, scores[idx]


def junk(q: str) -> bool:  # keep this as-is
    if len(q) < 3:
        return True
    alpha_num = sum(ch.isalnum() for ch in q)
    return alpha_num < max(2, int(len(q) * 0.3))


st.set_page_config(page_title="EN → JP Meme Search", layout="wide")
st.title("EN → JP meme search")

if not os.path.exists(CSV_PATH):
    st.error("can't find data/jpmemes.csv")
    st.stop()

mtime = os.path.getmtime(CSV_PATH)
data = load_data_and_emb(mtime)

with st.sidebar:
    k = st.slider("How many results", 1, 20, 7)
    min_score = st.slider("Match strictness", 0.0, 1.0, 0.72, 0.01)
    strict = st.checkbox("Ignore junk input", value=True)

q = st.text_input("English", placeholder="cringe / no cap / that's so real / touch grass / delulu ...")
go = st.button("Search", type="primary")

if go:
    q = (q or "").strip()

    if strict and junk(q):
        st.warning("Try a clearer phrase.")
        st.stop()

    model = load_model()
    qv = embed_query(model, q)
    idx, sc = search_topk(qv, data["emb"], k)

    hits = [(int(i), float(s)) for i, s in zip(idx, sc) if float(s) >= float(min_score)]
    if not hits:
        st.info("No good matches. Try another query or lower strictness.")
        st.stop()

    for r, (i, s) in enumerate(hits, start=1):
        jp = data["jp_text"][i]
        tag = data["source"][i]

        st.markdown(f"### #{r} — {s * 100:.0f}% match")
        st.write(jp)
        if tag:
            st.caption(tag)
        st.divider()
