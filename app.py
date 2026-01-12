import os
import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer

MODEL = "intfloat/multilingual-e5-base"
CSV_PATH = "data/jpmemes.csv"
# epsはほぼ保険のようなもの、０だけじゃ分からないので、norm = 0するとプログラム自体爆発しちゃう
# 詳しい説明はgoogle drive のノートでメモしましたから自分で復習しなー＞：０
EPS = 1e-12

# よく使うネットスラングリスト - 短いけどOK
ALLOW_SHORT = {
    "lol", "lmao", "rofl", "omg", "wtf", "idk", "ikr", "ngl", "fr", "tbh",
    "sus", "cap", "bet", "bruh", "gg", "w", "l", "nsfw", "jk", "pls", "thx", "wtf"
}

@st.cache_resource
def load_model():
    # 一回だけロードすればstreamlitがキャッシュしてくれる
    return SentenceTransformer(MODEL)

def norm_rows(x):
    """正規化"""
    #we could try float64 when i buy a new pc tho....ram is not good enough
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    # たまにゼロベクトルあるから一応epsにおきかえるー
    norms = np.where(norms > 0, norms, EPS)
    return x / norms

# stackoverflowの説明参考しながら書いたですけど。。将来的に参考になれる
# https://stackoverflow.com/questions/78879594/st-cache-data-is-deprecated-in-streamlit

@st.cache_data
def load_data_and_emb(csv_mtime):
    """CSV読んでembedding計算。mtimeが変わったら再計算"""
    df = pd.read_csv(CSV_PATH)

    # NaN処理、jp_textの存在は必要
    if "jp_text" not in df.columns:
        st.error("No 'jp_text' column in CSV...")
        st.stop()

    df["jp_text"] = df["jp_text"].fillna("").astype(str)

    # source―Excelのcolumn
    if "source" in df.columns:
        df["source"] = df["source"].fillna("").astype(str)
    else:
        df["source"] = ""

    # cｓｖの処理
    passages = []
    for idx, row in df.iterrows():
        txt = row["jp_text"].strip()
        src = str(row["source"]).strip() if row["source"] else ""

        if src:
            # タグありのやつ
            passages.append(f"passage: {txt} | tag: {src}")
        else:
            passages.append(f"passage: {txt}")

    model = load_model()
    emb = model.encode(passages, batch_size=64, show_progress_bar=False)
    emb = norm_rows(emb)

    return {
        "emb": emb,
        "jp_text": df["jp_text"].to_numpy(),
        "source": df["source"].to_numpy(),
    }

def embed_query(model, q):
    """クエリをベクトル化"""
    q = q.strip()

    # 2パターン試してみる
    v1 = model.encode([f"query: {q}"], show_progress_bar=False)
    v2 = model.encode([q], show_progress_bar=False)

    # 平均取ったら精度上がった気がする（要検証）
    combined = (v1 + v2) / 2.0
    combined = norm_rows(combined)

    return combined[0]
# cosine similarity, lambda
def search_topk(query_vec, emb, k):
    """コサイン類似度でソート"""
    scores = emb.dot(query_vec)

    # kが変な値来ないように
    k = min(k, len(scores))
    if k <= 0:
        return np.array([]), np.array([])

    # 上位k個のインデックス
    top_idx = np.argsort(scores)[::-1][:k]

    return top_idx, scores[top_idx]

def junk(q: str) -> bool:
    """短すぎるとか記号ばっかのやつ弾く"""
    q = q.strip()
    if len(q) < 3:
        return True

    # 英数字が少なすぎたらNG
    alnum_count = sum(1 for c in q if c.isalnum())
    threshold = max(2, int(len(q) * 0.3))

    return alnum_count < threshold

# ========== UI =================================================================

st.set_page_config(page_title="EN → JP Meme Search", layout="wide")
st.title("EN → JP meme search")

# ファイルチェック
if not os.path.exists(CSV_PATH):
    st.error(f"CSV file not found: {CSV_PATH}")
    st.stop()

# データロード
mtime = os.path.getmtime(CSV_PATH)
data = load_data_and_emb(mtime)

# サイドバー設定
with st.sidebar:
    st.header("Settings")
    k = st.slider("How many results", 1, 50, 12)
    min_score = st.slider("Match strictness", 0.0, 1.0, 0.55, 0.01)
    strict = st.checkbox("Ignore weird input", value=True)

# 検索ボックス
q = st.text_input(
    "Type English meme/slang",
    placeholder="cringe / no cap / that's so real / touch grass / delulu ..."
)

if st.button("Search", type="primary"):
    q = q.strip()

    if not q:
        st.warning("Type something pls")
        st.stop()

    # ゴミ入力チェック
    q_lower = q.lower()
    if strict:
        if q_lower not in ALLOW_SHORT and junk(q):
            st.warning("Maybe type a bit longer phrase...")
            st.stop()

    # 検索実行
    model = load_model()
    qv = embed_query(model, q)
    idx, scores = search_topk(qv, data["emb"], k)

    # スコアでフィルター
    results = []
    for i, score in zip(idx, scores):
        if score >= min_score:
            results.append((int(i), float(score)))

    # 結果表示
    if not results:
        st.info("No match. Try other words or lower the strictness.")
        st.stop()

    st.success(f"Found {len(results)} results")

    for rank, (i, score) in enumerate(results, 1):
        jp_text = data["jp_text"][i]
        source_tag = data["source"][i]

        # カード風に表示（好きなEMOJI入れる:)）
        with st.container():
            #decimal point
            st.markdown(f"### Rank {rank} — match {score*100:.3f}%  (raw {score:.6f})")
            st.write(jp_text)

            if source_tag:
                st.caption(f"🏷️ {source_tag}")

            st.divider()
