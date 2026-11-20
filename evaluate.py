"""Run Model A (Champion) and Model B (Challenger) on the same SciFact queries and log everything to MLflow.
Task: semantic search (retrieve the abstract that supports / refutes a scientific claim).
Student: Eggonu Venkata Tejal | USN: 1MS25CI040"""
import json, os, time, shutil
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow, torch, transformers, sentence_transformers
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer
from st_wrapper import SemanticSearchModel, encode, search

USN = "1MS25CI040"
SEED, N_QUERIES, MAX_LENGTH, BATCH_SIZE, TOP_K = 40, 100, 128, 32, 10
EXPERIMENT = f"{USN}-hf-semantic-search-champion-challenger"
# Champion rule (written in README BEFORE the final run)
PRIMARY, MARGIN, LATENCY_LIMIT = "ndcg_at_10", 2.0, 3.0
MODELS = {
    "A": {"short": "minilm-l6", "id": "sentence-transformers/all-MiniLM-L6-v2", "role": "champion",
          "query_prefix": ""},
    "B": {"short": "bge-small", "id": "BAAI/bge-small-en-v1.5", "role": "challenger",
          "query_prefix": "Represent this sentence for searching relevant passages: "},
}
METRICS = ["recall_at_1", "recall_at_5", "recall_at_10", "mrr_at_10", "ndcg_at_10",
           "avg_latency_ms", "total_inference_s", "corpus_encode_s"]
HIGHER_IS_BETTER = {"recall_at_1": True, "recall_at_5": True, "recall_at_10": True, "mrr_at_10": True,
                    "ndcg_at_10": True, "avg_latency_ms": False, "total_inference_s": False, "corpus_encode_s": False}
for d in ("data", "outputs", "hf_models"):
    os.makedirs(d, exist_ok=True)
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
mlflow.set_experiment(EXPERIMENT)
np.random.seed(SEED); torch.manual_seed(SEED); torch.set_num_threads(max(1, os.cpu_count() or 1))

# ---- 1. data: BEIR SciFact corpus (5,183 abstracts) + 100 test claims (seed 40) ----
corpus = pd.read_parquet(hf_hub_download("BeIR/scifact", "corpus/corpus-00000-of-00001.parquet", repo_type="dataset"))
queries = pd.read_parquet(hf_hub_download("BeIR/scifact", "queries/queries-00000-of-00001.parquet", repo_type="dataset"))
qrels = pd.read_csv(hf_hub_download("BeIR/scifact-qrels", "test.tsv", repo_type="dataset"), sep="\t")
corpus = pd.DataFrame({"doc_id": corpus["_id"].astype(str), "title": corpus.title.fillna("").astype(str),
                       "text": corpus.text.fillna("").astype(str)})
corpus.to_parquet("hf_models/corpus.parquet", index=False)
doc_texts = (corpus.title + ". " + corpus.text).tolist()

rel = qrels[qrels.score > 0].astype(str).groupby("query-id")["corpus-id"].apply(lambda s: ";".join(sorted(s)))
qids = sorted(rel.index, key=int)
chosen = pd.Series(qids).sample(N_QUERIES, random_state=SEED).sort_values(key=lambda s: s.astype(int)).tolist()
qtext = queries.assign(_id=queries["_id"].astype(str)).set_index("_id").text
ev = pd.DataFrame({"sample_id": [f"q_{i:03d}" for i in range(1, N_QUERIES + 1)], "query_id": chosen,
                   "query": [qtext[q].strip() for q in chosen], "relevant_doc_ids": [rel[q] for q in chosen]})
ev.to_csv("data/eval_inputs.csv", index=False)
relevant = [set(r.split(";")) for r in ev.relevant_doc_ids]
print(f"Evaluation set: {len(ev)} SciFact test claims, corpus of {len(corpus)} abstracts -> data/eval_inputs.csv")


def score_rankings(ranked_ids):
    """ranked_ids: list (per query) of top-10 doc ids. Returns per-query table + mean metrics (in %)."""
    rows = []
    for ids, rel_set in zip(ranked_ids, relevant):
        hits = [1 if d in rel_set else 0 for d in ids]
        first = next((i + 1 for i, h in enumerate(hits) if h), None)
        dcg = sum(h / np.log2(i + 2) for i, h in enumerate(hits))
        idcg = sum(1 / np.log2(i + 2) for i in range(min(len(rel_set), TOP_K)))
        rows.append({"recall_at_1": sum(hits[:1]) / len(rel_set), "recall_at_5": sum(hits[:5]) / len(rel_set),
                     "recall_at_10": sum(hits[:10]) / len(rel_set), "mrr_at_10": 1 / first if first else 0.0,
                     "ndcg_at_10": dcg / idcg, "first_relevant_rank": first or 0,
                     "recall_curve": [sum(hits[:k]) / len(rel_set) for k in range(1, TOP_K + 1)]})
    per = pd.DataFrame(rows)
    means = {m: round(100 * per[m].mean(), 4) for m in ["recall_at_1", "recall_at_5", "recall_at_10", "mrr_at_10", "ndcg_at_10"]}
    curve = (100 * np.array(per.recall_curve.tolist()).mean(axis=0)).round(3).tolist()
    return per, means, curve


tags = {"student_usn": USN, "dataset_name": "BeIR/scifact (test claims) + BeIR/scifact-qrels",
        "transformers_version": transformers.__version__, "torch_version": torch.__version__,
        "sentence_transformers_version": sentence_transformers.__version__, "mlflow_version": mlflow.__version__}
results, curves, per_query = {}, {}, {}

with mlflow.start_run(run_name="compare_semantic_search") as parent:
    mlflow.set_tags(tags)
    for key, m in MODELS.items():
        run_name = f"{key}_semantic-search_{m['short']}"
        print(f"\n=== {run_name}  ({m['id']}) ===")
        st = SentenceTransformer(m["id"], device="cpu")
        st.max_seq_length = MAX_LENGTH

        t0 = time.perf_counter()
        corpus_emb = encode(st, doc_texts, BATCH_SIZE)               # passages: no prefix
        corpus_s = time.perf_counter() - t0
        print(f"Corpus encoded: {corpus_emb.shape} in {corpus_s:.1f}s")

        encode(st, ["warm-up query"], prefix=m["query_prefix"])    # warm-up, not timed
        ranked, top_scores, lat = [], [], []
        for q in ev["query"]:                                        # one query at a time = realistic latency
            t = time.perf_counter()
            idx, sc = search(encode(st, [q], prefix=m["query_prefix"]), corpus_emb, TOP_K)
            lat.append(time.perf_counter() - t)
            ranked.append(corpus.doc_id.iloc[idx[0]].tolist()); top_scores.append(float(sc[0][0]))
        per, means, curve = score_rankings(ranked)
        metrics = {**means, "avg_latency_ms": round(1000 * float(np.mean(lat)), 4),
                   "total_inference_s": round(float(np.sum(lat)), 4), "corpus_encode_s": round(corpus_s, 4)}
        results[key], curves[key], per_query[key] = metrics, curve, per

        # predictions file: input, reference, prediction and per-query scores side by side
        title_of = dict(zip(corpus.doc_id, corpus.title))
        preds = ev.copy()
        preds["top1_doc_id"] = [r[0] for r in ranked]
        preds["top1_title"] = [title_of[r[0]] for r in ranked]
        preds["top1_score"] = [round(s, 4) for s in top_scores]
        preds["top10_doc_ids"] = [";".join(r) for r in ranked]
        preds["first_relevant_rank"] = per.first_relevant_rank          # 0 = not in top 10
        preds["hit_at_1"] = (per.first_relevant_rank == 1).astype(int)
        preds["recall_at_10"] = per.recall_at_10.round(4)
        preds["ndcg_at_10"] = per.ndcg_at_10.round(4)
        pred_path = f"outputs/predictions_{m['short']}.csv"
        preds.to_csv(pred_path, index=False)
        ms_path = f"outputs/metrics_summary_{m['short']}.json"
        json.dump({"model": m["id"], **metrics, "recall_at_k_curve": curve}, open(ms_path, "w"), indent=2)

        # save the HF model + corpus embeddings so the pyfunc can be loaded without the Hub
        model_dir = f"hf_models/{m['short']}"
        shutil.rmtree(model_dir, ignore_errors=True); st.save(model_dir)
        emb_path = f"hf_models/corpus_emb_{m['short']}.npy"; np.save(emb_path, corpus_emb.astype(np.float32))

        with mlflow.start_run(run_name=run_name, nested=True) as child:
            mlflow.set_tags({**tags, "role": m["role"]})
            mlflow.log_params({"hf_model_id": m["id"], "task": "semantic-search", "role": f"{m['role']} candidate",
                               "max_length": MAX_LENGTH, "batch_size": BATCH_SIZE, "top_k": TOP_K,
                               "query_prefix": m["query_prefix"] or "(none)", "similarity": "cosine (normalised dot)",
                               "eval_samples": N_QUERIES, "corpus_size": len(corpus), "seed": SEED, "device": "cpu"})
            mlflow.log_metrics(metrics)
            for k, v in enumerate(curve, start=1):
                mlflow.log_metric("recall_at_k_curve", v, step=k)
            for p in (pred_path, ms_path, "data/eval_inputs.csv"):
                mlflow.log_artifact(p)
            fig, ax = plt.subplots(figsize=(6, 4))
            ranks = per.first_relevant_rank.replace(0, TOP_K + 1)
            ax.hist(ranks, bins=np.arange(0.5, TOP_K + 2.5, 1), color="#4C72B0" if key == "A" else "#DD8452")
            ax.set_xticks(range(1, TOP_K + 2)); ax.set_xticklabels([*map(str, range(1, TOP_K + 1)), ">10"])
            ax.set_xlabel("Rank of first relevant abstract"); ax.set_ylabel("Number of queries")
            ax.set_title(f"{key}: {m['short']} – first relevant rank"); fig.tight_layout()
            fig.savefig(f"outputs/first_relevant_rank_{m['short']}.png", dpi=120)
            mlflow.log_figure(fig, f"first_relevant_rank_{m['short']}.png"); plt.close(fig)
            info = mlflow.pyfunc.log_model(
                name="model", python_model=SemanticSearchModel(), code_paths=["st_wrapper.py"],
                artifacts={"st_model_dir": model_dir, "corpus": "hf_models/corpus.parquet", "corpus_embeddings": emb_path},
                model_config={"output_top_k": 3, "max_length": MAX_LENGTH, "query_instruction": m["query_prefix"]},
                input_example=pd.DataFrame({"query": ["Vitamin D deficiency is linked to depression."]}),
                pip_requirements=["mlflow", "sentence-transformers", "transformers", "torch", "pandas", "numpy", "pyarrow"])
            m["run_id"], m["model_uri"] = child.info.run_id, info.model_uri
        print(json.dumps(metrics, indent=2))
        del st

    # ---- 2. Champion–Challenger decision (rule fixed before the run) ----
    a, b = results["A"], results["B"]
    delta = round(b[PRIMARY] - a[PRIMARY], 4)
    ratio = round(b["avg_latency_ms"] / a["avg_latency_ms"], 3)
    if ratio > LATENCY_LIMIT:
        winner, reason = "A", f"B latency {ratio}x > {LATENCY_LIMIT}x guardrail, Champion kept"
    elif delta >= MARGIN:
        winner, reason = "B", f"delta nDCG@10 {delta:+.2f} >= {MARGIN} margin and latency {ratio}x within guardrail"
    elif abs(delta) < MARGIN:
        faster = "A" if a["avg_latency_ms"] <= b["avg_latency_ms"] else "B"
        winner, reason = faster, f"|delta nDCG@10| {abs(delta):.2f} < {MARGIN}, faster model kept"
    else:
        winner, reason = "A", f"B is worse by {abs(delta):.2f} nDCG@10 points, Champion kept"
    print(f"\nDECISION: winner = {winner} ({MODELS[winner]['id']})\nReason: {reason}")

    mlflow.log_params({"primary_metric": PRIMARY, "margin": MARGIN, "latency_guardrail_x": LATENCY_LIMIT,
                       "tie_breaker": "faster model (avg_latency_ms)", "model_A": MODELS["A"]["id"], "model_B": MODELS["B"]["id"]})
    mlflow.set_tags({"winner": f"{winner}_{MODELS[winner]['short']}", "decision_reason": reason})
    mlflow.log_metrics({"delta_primary_metric": delta, "delta_ndcg_at_10": delta, "latency_ratio_B_over_A": ratio,
                        **{f"A_{k}": v for k, v in a.items()}, **{f"B_{k}": v for k, v in b.items()}})

    # comparison table + summary json
    comp = pd.DataFrame({"metric": METRICS, "A_minilm-l6": [a[k] for k in METRICS], "B_bge-small": [b[k] for k in METRICS],
                         "B_minus_A": [round(b[k] - a[k], 4) for k in METRICS],
                         "higher_is_better": [HIGHER_IS_BETTER[k] for k in METRICS]})
    comp.to_csv("outputs/model_comparison.csv", index=False)
    summary = {"decision": {"winner": winner, "reason": reason, "primary_metric": PRIMARY, "margin": MARGIN,
                            "delta_ndcg_at_10": delta, "latency_ratio": ratio},
               **{k: {**results[k], "hf_model_id": MODELS[k]["id"], "run_name": f"{k}_semantic-search_{MODELS[k]['short']}",
                      "run_id": MODELS[k]["run_id"], "model_uri": MODELS[k]["model_uri"]} for k in ("A", "B")}}
    json.dump(summary, open("outputs/metrics_summary.json", "w"), indent=2)

    # plot 1: bar chart of every quality metric
    qm = ["recall_at_1", "recall_at_5", "recall_at_10", "mrr_at_10", "ndcg_at_10"]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [3, 1]})
    x = np.arange(len(qm)); w = 0.38
    for off, k, c in ((-w / 2, "A", "#4C72B0"), (w / 2, "B", "#DD8452")):
        bars = axs[0].bar(x + off, [results[k][q] for q in qm], w, label=f"{k}: {MODELS[k]['short']}", color=c)
        axs[0].bar_label(bars, fmt="%.1f", fontsize=8)
    axs[0].set_xticks(x); axs[0].set_xticklabels(["Recall@1", "Recall@5", "Recall@10", "MRR@10", "nDCG@10"])
    axs[0].set_ylabel("% (higher is better)"); axs[0].set_ylim(0, 105); axs[0].legend(loc="upper left")
    axs[0].set_title("Retrieval quality on 100 SciFact claims")
    bars = axs[1].bar(["A", "B"], [a["avg_latency_ms"], b["avg_latency_ms"]], color=["#4C72B0", "#DD8452"])
    axs[1].bar_label(bars, fmt="%.1f", fontsize=8); axs[1].set_title("Avg latency, ms (lower = better)", fontsize=10)
    fig.tight_layout(); fig.savefig("outputs/metric_comparison.png", dpi=120)
    mlflow.log_figure(fig, "metric_comparison.png"); plt.close(fig)

    # plot 2: Recall@k curve
    fig, ax = plt.subplots(figsize=(6, 4))
    for k, c in (("A", "#4C72B0"), ("B", "#DD8452")):
        ax.plot(range(1, TOP_K + 1), curves[k], marker="o", color=c, label=f"{k}: {MODELS[k]['short']}")
    ax.set_xlabel("k"); ax.set_ylabel("Recall@k (%)"); ax.set_xticks(range(1, TOP_K + 1)); ax.grid(alpha=.3)
    ax.set_title("Recall@k curve (SciFact, 100 claims)"); ax.legend(); fig.tight_layout()
    fig.savefig("outputs/recall_at_k_curve.png", dpi=120)
    mlflow.log_figure(fig, "recall_at_k_curve.png"); plt.close(fig)

    # disagreements: queries where the two models return a different top-1 abstract
    pa, pb = pd.read_csv("outputs/predictions_minilm-l6.csv", dtype=str), pd.read_csv("outputs/predictions_bge-small.csv", dtype=str)
    dis = pd.DataFrame({"sample_id": pa.sample_id, "query": pa["query"], "relevant_doc_ids": pa.relevant_doc_ids,
                        "A_top1_doc_id": pa.top1_doc_id, "A_top1_title": pa.top1_title, "A_first_relevant_rank": pa.first_relevant_rank,
                        "B_top1_doc_id": pb.top1_doc_id, "B_top1_title": pb.top1_title, "B_first_relevant_rank": pb.first_relevant_rank})
    dis = dis[dis.A_top1_doc_id != dis.B_top1_doc_id]
    dis.to_csv("outputs/disagreements.csv", index=False)
    a_right = (dis.A_first_relevant_rank == "1").sum(); b_right = (dis.B_first_relevant_rank == "1").sum()
    print(f"Top-1 disagreements: {len(dis)} / {N_QUERIES}  (A correct on {a_right}, B correct on {b_right})")
    mlflow.log_metrics({"top1_disagreements": len(dis), "disagree_A_correct": int(a_right), "disagree_B_correct": int(b_right)})

    for p in ("outputs/model_comparison.csv", "outputs/metrics_summary.json", "outputs/disagreements.csv",
              "data/eval_inputs.csv", "README.md", "outputs/qualitative_examples.csv"):
        if os.path.exists(p):
            mlflow.log_artifact(p)

json.dump({k: {"run_id": MODELS[k]["run_id"], "hf_model_id": MODELS[k]["id"]} for k in ("A", "B")} | {"winner": winner},
          open("outputs/model_runs.json", "w"), indent=2)
print(f"\nParent run: {parent.info.run_id}  ->  {mlflow.get_tracking_uri()}")
