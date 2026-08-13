"""
Hybrid retriever combining dense (bge-m3 embedding) and sparse (BM25) retrieval.
Fuses both rankings via weighted score combination (or reciprocal rank fusion).
"""
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import numpy as np
import json
import re
import jieba

class HybridRetriever:
    def __init__(self, corpus_path, model_name="BAAI/bge-m3", dense_weight=0.5):
        self.model = SentenceTransformer(model_name)
        self.corpus = self._load_jsonl(corpus_path)
        self.corpus_texts = [c["text"] for c in self.corpus]
        self.corpus_ids = [c["id"] for c in self.corpus]
        self.dense_weight = dense_weight  # 0.5 = equal weight; tune as needed

        print(f"Encoding {len(self.corpus)} chunks with {model_name} (dense) ...")
        self.corpus_emb = self.model.encode(
            self.corpus_texts, normalize_embeddings=True, show_progress_bar=True,
            batch_size=8,
        )

        print("Building BM25 index (sparse) ...")
        # Simple tokenizer: works reasonably for both English and Chinese
        # (Chinese gets character-level tokens, English gets word-level)
        self.tokenized_corpus = [self._tokenize(t) for t in self.corpus_texts]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    @staticmethod
    def _load_jsonl(path):
        with open(path, encoding="utf-8-sig") as f:
            return [json.loads(line) for line in f if line.strip()]

    @staticmethod
    def _tokenize(text):
        """
        Mixed-language tokenizer: uses jieba for Chinese word segmentation
        (word-level, not character-level) and simple regex splitting for
        English/numeric tokens. This fixes the earlier issue where CJK
        characters were split individually, which badly hurt BM25 recall
        on Chinese-language queries.
        """
        tokens = []
        # jieba.cut handles the whole string; for mixed EN/CN text it
        # will yield Chinese words as segmented units and leave English
        # substrings largely intact as their own tokens too.
        for word in jieba.cut(text):
            word = word.strip().lower()
            if not word:
                continue
            # Keep only tokens that contain at least one alphanumeric or CJK char
            # (filters out pure punctuation/whitespace tokens)
            if re.search(r"[A-Za-z0-9\u4e00-\u9fff]", word):
                tokens.append(word)
        return tokens

    @staticmethod
    def _is_likely_chinese(text):
        """Rough heuristic: if >30% of non-whitespace chars are CJK, treat as Chinese."""
        chars = [c for c in text if not c.isspace()]
        if not chars:
            return False
        cjk_count = sum(1 for c in chars if '\u4e00' <= c <= '\u9fff')
        return (cjk_count / len(chars)) > 0.3

    @staticmethod
    def _normalize_scores(scores):
        """Min-max normalize to [0, 1] so dense and sparse scores are comparable."""
        scores = np.array(scores, dtype=float)
        if scores.max() == scores.min():
            return np.zeros_like(scores)
        return (scores - scores.min()) / (scores.max() - scores.min())

    def retrieve(self, query, top_k=5, dense_weight=None):
        if dense_weight is not None:
            # Explicit override always wins (used by grid-search style testing)
            w = dense_weight
        else:
            # Query routing: cross-lingual query/document mismatches are a
            # known structural weakness of BM25 (sparse token matching can't
            # bridge languages). If the query itself is NOT predominantly
            # Chinese, lean much more heavily on dense (semantic) retrieval
            # to avoid BM25 dragging down cross-lingual matches. If the query
            # IS Chinese, keep the base weight to preserve BM25's benefit on
            # same-language keyword/concept matching.
            query_is_chinese = self._is_likely_chinese(query)
            w = self.dense_weight if query_is_chinese else 0.9

        # Dense (semantic) scores
        q_emb = self.model.encode([query], normalize_embeddings=True)
        dense_scores = (self.corpus_emb @ q_emb.T).flatten()

        # Sparse (BM25 keyword) scores
        tokenized_query = self._tokenize(query)
        sparse_scores = np.array(self.bm25.get_scores(tokenized_query))

        # Normalize both to [0,1] before combining (raw scales differ a lot)
        dense_norm = self._normalize_scores(dense_scores)
        sparse_norm = self._normalize_scores(sparse_scores)

        combined = w * dense_norm + (1 - w) * sparse_norm

        ranked_idx = np.argsort(-combined)[:top_k]
        results = []
        for idx in ranked_idx:
            results.append({
                "id": self.corpus_ids[idx],
                "text": self.corpus_texts[idx],
                "source": self.corpus[idx]["source"],
                "url": self.corpus[idx].get("url", ""),
                "combined_score": float(combined[idx]),
                "dense_score": float(dense_scores[idx]),
                "sparse_score": float(sparse_scores[idx]),
            })
        return results