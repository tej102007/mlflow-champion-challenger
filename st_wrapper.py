"""Semantic-search helpers + custom MLflow pyfunc wrapper (Lab 9 style).
The wrapper ships the sentence-transformer, the SciFact corpus and its pre-computed embeddings,
so a model loaded from MLflow can answer a query without any Hugging Face code in the caller.
Student: Eggonu Venkata Tejal | USN: 1MS25CI040"""
import mlflow.pyfunc
import numpy as np
import pandas as pd


def add_prefix(texts, prefix):
    """Some models (e.g. BGE) expect an instruction prefix on the query side only."""
    return [prefix + t for t in texts] if prefix else list(texts)


def encode(model, texts, batch_size=32, prefix=""):
    """Return L2-normalised embeddings, so a dot product equals cosine similarity."""
    return model.encode(add_prefix(texts, prefix), batch_size=batch_size, convert_to_numpy=True,
                        normalize_embeddings=True, show_progress_bar=False)


def search(query_emb, corpus_emb, top_k=10):
    """Return (indices, scores) of the top_k most similar corpus rows for each query embedding."""
    scores = query_emb @ corpus_emb.T
    idx = np.argsort(-scores, axis=1)[:, :top_k]
    return idx, np.take_along_axis(scores, idx, axis=1)


class SemanticSearchModel(mlflow.pyfunc.PythonModel):
    """Input: DataFrame with a `query` column.
    Output: DataFrame with one row per (query, rank): query, rank, doc_id, title, score."""

    def load_context(self, context):
        from sentence_transformers import SentenceTransformer
        cfg = context.model_config or {}
        self.top_k = int(cfg.get("output_top_k", 3))
        self.query_prefix = cfg.get("query_instruction", "")
        self.model = SentenceTransformer(context.artifacts["st_model_dir"], device="cpu")
        self.model.max_seq_length = int(cfg.get("max_length", 256))
        self.corpus = pd.read_parquet(context.artifacts["corpus"])
        self.corpus_emb = np.load(context.artifacts["corpus_embeddings"])

    def predict(self, context, model_input, params=None):
        queries = model_input["query"].astype(str).tolist()
        q_emb = encode(self.model, queries, prefix=self.query_prefix)
        idx, sc = search(q_emb, self.corpus_emb, self.top_k)
        rows = []
        for q, ii, ss in zip(queries, idx, sc):
            for r, (i, s) in enumerate(zip(ii, ss), start=1):
                rows.append({"query": q, "rank": r, "doc_id": str(self.corpus.doc_id.iloc[i]),
                             "title": self.corpus.title.iloc[i], "score": round(float(s), 4)})
        return pd.DataFrame(rows)
