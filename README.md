---
title: crosslingual-meme-search
colorFrom: purple
colorTo: pink
sdk: streamlit
app_file: app.py
pinned: false
---

# crosslingual-meme-search
A tiny cross-lingual meme finder: type English, get Japanese meme text back.

I built this after chatting with a friend in Switzerland. We kept running into the same problem: even when the *meaning* is clear in English, the “right” Japanese meme phrasing or slang doesn’t translate cleanly. 
My friend is also learning Japanese, so I made this little tool so we can explore Japanese internet language together learn Japanese, pick up slang, and discover meme expressions without getting stuck on literal translation.

# EN → JP Meme Search (Cross-lingual Retrieval Demo)

A small Streamlit app that matches **English meme/slang queries** to **Japanese meme examples** using a multilingual sentence embedding model (**intfloat/multilingual-e5-base**) and cosine similarity.

**Demo:** https://huggingface.co/spaces/Erikasbsk/crosslingual-meme-search-new  

**Learning notes (NLP + Ds theory):** [https://github.com/ERIKASBSK/Learning-Portfolio/tree/main](https://github.com/ERIKASBSK/Learning-Note/tree/main)

**MEMO(implementation ideas):** https://github.com/ERIKASBSK/Learning-Note/blob/main/Memo%20of%20meme%20research.md

---

## What it does

- Takes an **English query** (e.g., *“touch grass”*, *“no cap”*)
- Embeds the query with **E5** (`query:` prefix)
- Embeds Japanese examples from a CSV with **E5** (`passage:` prefix)
- Ranks results by **cosine similarity**
- Optionally applies **MMR** to reduce near-duplicate outputs (diversity-aware ranking)

---

## Features

- Cross-lingual semantic retrieval (EN → JP)
- Cosine similarity ranking (normalized embeddings)
- Streamlit UI + sidebar controls
- Cached model + cached embeddings (recomputed when CSV changes)
- Optional “Ignore weird input” filter for cleaner demos
- Optional MMR reranking (diversity vs. relevance trade-off) --*under testing

---

## Notes / Known limitations

Embedding-based retrieval can show length / structure bias across languages.  
For example, very short English queries sometimes retrieve short Japanese outputs that look plausible but are not the best semantic match.
This project is a demo to explore those failure modes and iteration ideas.
