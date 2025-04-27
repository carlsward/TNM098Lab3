#!/usr/bin/env python3
"""
VAST Challenge 2011 – MC3 quick-and-dirty text analysis
=======================================================

• Author:  <your name>
• Created: 2025-04-27
• Requires: pandas, matplotlib, scikit-learn, nltk (1st run will download NLTK stop-word list)
• Usage:   python vast_mc3_analysis.py
"""

# ── imports ───────────────────────────────────────────────────────────────
from pathlib import Path
import re
import warnings

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer

# uncomment first run only – downloads stop-word list once
# import nltk; nltk.download("stopwords")
from nltk.corpus import stopwords

warnings.filterwarnings("ignore")              # keep console clean

# ── CONFIG – tweak these if you want ─────────────────────────────────────
CSV_FILE   = Path("TNM098-MC3-2011.csv")       # put the CSV in the same folder
N_TOPICS   = 5                                 # LDA topics
KEYWORDS   = [
    "threat", "terror", "terrorism", "attack", "explosion",
    "bomb", "dirty", "weapon", "warnings", "rifle"
]
STOP_WORDS = set(stopwords.words("english"))
MIN_TOKEN_LEN = 2                              # drop tokens shorter than this
FIG_DPI = 110

# ── helpers ───────────────────────────────────────────────────────────────
def preprocess(txt: str) -> str:
    """
    Very light cleanup:
    • lower-case
    • strip non-letters
    • remove stop-words & short tokens
    """
    txt = txt.lower()
    txt = re.sub(r"[^a-z\s]", " ", txt)
    tokens = [
        tok for tok in txt.split()
        if tok not in STOP_WORDS and len(tok) > MIN_TOKEN_LEN
    ]
    return " ".join(tokens)


def bar(ax, series, title, color):
    """Draws a bar chart with nicer defaults."""
    series.plot.bar(ax=ax, color=color, width=0.9)
    ax.set_title(title, weight="bold")
    ax.set_xlabel("date")
    ax.set_ylabel("# reports")
    for label in ax.get_xticklabels():
        label.set_rotation(90)
        label.set_fontsize(8)
    plt.tight_layout()


# ── 1. Load & basic cleaning ─────────────────────────────────────────────
if not CSV_FILE.exists():
    raise FileNotFoundError(
        f"❌  Could not find {CSV_FILE}. Place the CSV in this folder or update CSV_FILE."
    )

df = pd.read_csv(CSV_FILE, sep=';')

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.dropna(subset=["Date"])        # drop malformed dates

# combine title + content, then clean
df["text"] = (
    df["Title"].fillna("") + " " + df["Content"].fillna("")
).apply(preprocess)

# ── 2. overall timeline ──────────────────────────────────────────────────
counts_all = (
    df.groupby(df["Date"].dt.date)
      .size()
      .sort_index()
)

# ── 3. simple keyword filter ─────────────────────────────────────────────
regex = r"\b(" + "|".join(KEYWORDS) + r")\b"
mask  = df["text"].str.contains(regex, regex=True)
df_f  = df[mask]

counts_f = (
    df_f.groupby(df_f["Date"].dt.date)
        .size()
        .sort_index()
)

# ── 4. plot unfiltered vs filtered timelines side-by-side ───────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), dpi=FIG_DPI, sharex=True)
bar(ax1, counts_all, "All news items per day", "tab:grey")
bar(ax2, counts_f,   "Threat-related items per day", "tab:red")
fig.suptitle("Temporal distribution of news reports", weight="bold", y=0.98)
plt.show()

# ── 5. LDA topic modelling on filtered set ───────────────────────────────
vectorizer = TfidfVectorizer(max_df=0.95, min_df=2)   # ignore very common & very rare
tfidf = vectorizer.fit_transform(df_f["text"])

lda = LatentDirichletAllocation(
    n_components=N_TOPICS,
    learning_method="batch",
    random_state=0
)
lda.fit(tfidf)

terms = vectorizer.get_feature_names_out()

print("\n──────────  TOP TERMS PER TOPIC  ──────────")
for t, comp in enumerate(lda.components_):
    top_ids = comp.argsort()[:-11:-1]                 # 10 strongest words
    print(f"Topic {t+1}: " + ", ".join(terms[i] for i in top_ids))

# assign the most-probable topic id to every filtered article
df_f["TopicID"] = lda.transform(tfidf).argmax(axis=1)

# ── 6. per-topic timelines ───────────────────────────────────────────────
for topic in range(N_TOPICS):
    topic_mask   = df_f["TopicID"] == topic
    topic_counts = (
        df_f.loc[topic_mask]
            .groupby(df_f["Date"].dt.date)
            .size()
            .sort_index()
    )
    if topic_counts.empty:
        continue

    plt.figure(figsize=(10, 3), dpi=FIG_DPI)
    bar(plt.gca(), topic_counts,
        f"Timeline for topic {topic+1}", "tab:blue")
    plt.show()

print("\nDone!  ✨  Use the plots & printed keywords to decide which topic(s) capture the real threat story.")
