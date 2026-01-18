import os
import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer

MODEL = "intfloat/multilingual-e5-base"
CSV_PATH = "data/jpmemes.csv"
# epsはほぼ保険のようなもの、０だけじゃ分からないので、norm = 0するとプログラム自体爆発しちゃう
# 詳しい説明はgoogle drive のノートでメモしました
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

    # source―Excelのcolumn+ img
    if "source" in df.columns:
        df["source"] = df["source"].fillna("").astype(str)
    else:
        df["source"] = ""

    if "img" in df.columns:
        df["img"] = df["img"].fillna("").astype(str)
    else:
        df["img"] = ""

    # cｓｖの処理
    passages = []
    for idx, row in df.iterrows():
        txt = row["jp_text"].strip()
        passages.append(f"passage: {txt}")

    model = load_model()
    emb = model.encode(passages, batch_size=64, show_progress_bar=False)
    emb = norm_rows(emb)

    return {
        "emb": emb,
        "jp_text": df["jp_text"].to_numpy(),
        "source": df["source"].to_numpy(),
        "img": df["img"].to_numpy(),
    }

def embed_query(model, q):
    """E5安定版for the accuracy"""
    q = q.strip()
    v = model.encode([f"query: {q}"], show_progress_bar=False)
    v = norm_rows(v)
    return v[0]

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

def mmr_select(query_vec, emb, candidate_idx, k, lam=0.75):
    """
    λ（ラムダ）について
    λが大きい -> 精度重視
    λが小さい -> 多様性重視 
    """
    candidate_idx = list(map(int, candidate_idx))
    if not candidate_idx or k <= 0:
        return []

    # relevance: cosine(query, doc) ≈ dot(query_vec, doc_vec)
    rel = emb[candidate_idx].dot(query_vec)

    selected = []

    while len(selected) < k and candidate_idx:
        if not selected:
            best_pos = int(np.argmax(rel))
        else:
            cand_emb = emb[candidate_idx]     
            sel_emb = emb[selected]           
  
            sim_to_selected = cand_emb.dot(sel_emb.T)   
            max_sim = sim_to_selected.max(axis=1)       

            mmr_scores = lam * rel - (1 - lam) * max_sim
            best_pos = int(np.argmax(mmr_scores))

        best_idx = candidate_idx[best_pos]
        selected.append(best_idx)

        
        candidate_idx.pop(best_pos)
        rel = np.delete(rel, best_pos)

    return selected

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
    use_mmr = st.checkbox("Use MMR(Under testing. Please refrain from use for now.)", value=False)
    mmr_lam = st.slider("MMR lambda", 0.50, 0.95, 0.75, 0.01)


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

    # ====== 検索実行するよ ======
    model = load_model()
    qv = embed_query(model, q)

    # MMR使うなら候補は多めに取っとく（選べる余地が大事なの）
    candidate_pool = max(80, k * 10)
    idx, scores = search_topk(qv, data["emb"], candidate_pool)

    # スコア低すぎるのは先に落とすよ
    cand = [(int(i), float(s)) for i, s in zip(idx, scores) if s >= min_score]

    if not cand:
        st.info("No match. Try other words or lower the strictness.")
        st.stop()

    cand_idx = np.array([i for i, _ in cand], dtype=int)

    # MMR同じのばっか出る問題を回避するやつ
    if use_mmr:
        final_idx = mmr_select(qv, data["emb"], cand_idx, k=k, lam=mmr_lam)
    else:
        # MMRなしなら普通に上からk個ね
        final_idx = cand_idx[:k].tolist()

    if not final_idx:
        st.info("No match after filtering. Try lower strictness.")
        st.stop()

    st.success(f"Found {len(final_idx)} results")

    # ====== 結果表示だよ〜 ======
    for rank, i in enumerate(final_idx, 1):
        jp_text = data["jp_text"][i]
        source_tag = data["source"][i]
        img = str(data["img"][i]).strip()

        # 表示用にスコアをもう一回計算（正規化してるからdotでOK + image no setting
        score = float(data["emb"][i].dot(qv))

        with st.container():
            st.markdown(f"### Rank {rank} — match {score*100:.1f}%")

            if img:
                try:
                    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
                    st.image(img, width=320)
                    st.markdown("</div>", unsafe_allow_html=True)
                except Exception:
                    st.caption("🖼️ image link broken / blocked")

            st.write(jp_text)

            if source_tag:
                st.caption(f"🏷️ {source_tag}")

            st.divider()


