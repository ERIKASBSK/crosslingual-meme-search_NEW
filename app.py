import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer

MODEL = "intfloat/multilingual-e5-base"
CSV_PATH = "data/jpmemes.csv"


@st.cache_resource
def load_model():
    return SentenceTransformer(MODEL)


def normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def clean_url(x) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if not s or s.lower() == "nan":
        return ""
    if s.startswith("http://") or s.startswith("https://"):
        return s
    return ""


@st.cache_data
def load_data_and_emb():
    df = pd.read_csv(CSV_PATH)

    # req column
    if "jp_text" not in df.columns:
        raise ValueError("CSV must contain column: jp_text")

    # opt columns
    if "source" not in df.columns:
        df["source"] = ""
    if "image_url" not in df.columns:
        df["image_url"] = ""

    texts = df["jp_text"].fillna("").astype(str).tolist()

    m = load_model()
    passages = [f"passage: {t}" for t in texts]
    emb = m.encode(passages, batch_size=64, show_progress_bar=False)
    emb = normalize_rows(emb)

    return {
        "emb": emb,
        "jp_text": df["jp_text"].fillna("").astype(str).to_numpy(),
        "source": df["source"].fillna("").astype(str).to_numpy(),
        "image_url": df["image_url"].fillna("").astype(str).to_numpy(),
    }


def qvec(m, q: str) -> np.ndarray:
    v = m.encode([f"query: {q}"])
    v = normalize_rows(v)
    return v[0]


def topk(qv: np.ndarray, emb: np.ndarray, k: int):
    s = emb @ qv  # cosine sim (both normalized)
    k = max(1, min(int(k), len(s)))
    idx = np.argpartition(-s, kth=k - 1)[:k]
    idx = idx[np.argsort(-s[idx])]
    return idx, s[idx]


def is_garbage_query(q: str) -> bool:
    q = q.strip()
    if len(q) < 3:
        return True
    alpha_num = sum(ch.isalnum() for ch in q)
    return alpha_num < max(2, int(len(q) * 0.3))


st.set_page_config(page_title="EN → JP Meme Search", layout="wide")
st.title("EN → JP meme text search")

with st.sidebar:
    k = st.slider("Top-K", 1, 20, 5)
    min_score = st.slider("Min score", 0.0, 1.0, 0.70, 0.01)
    show_img = st.checkbox("show images", value=False)
    strict_mode = st.checkbox("strict (avoid nonsense queries)", value=True)

q = st.text_input(
    "English query",
    placeholder="lol / cringe / delulu / awkward / that's so real / I'm dead 💀 / bro what",
)
go = st.button("Search", type="primary")

if go:
    q = q.strip()

    if strict_mode and is_garbage_query(q):
        st.warning("Try a clearer phrase, or turn off strict mode.")
        st.stop()

    if not q:
        st.warning("Type something.")
        st.stop()

    data = load_data_and_emb()
    m = load_model()
    qv = qvec(m, q)

    idx, sc = topk(qv, data["emb"], k)

    results = [(int(i), float(s)) for i, s in zip(idx, sc) if float(s) >= float(min_score)]

    if not results:
        st.info("No good matches. Try a clearer phrase or lower Min score.")
        st.stop()

    for r, (i, s) in enumerate(results, start=1):
        jp = data["jp_text"][i]
        src = data["source"][i]
        img = clean_url(data["image_url"][i])

        st.markdown(f"### #{r}  ({s:.4f})")
        st.write(jp)

        if src and src.lower() != "nan":
            st.caption(src)

        if show_img:
            if img:
                st.image(img)
            else:
                st.caption("No image URL for this entry.")

        st.divider()
