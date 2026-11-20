# 1MS25CI040 – Champion–Challenger Model Comparison with Hugging Face and MLflow
**Student:** Eggonu Venkata Tejal · **USN:** 1MS25CI040 · **Section:** CSE (AI&ML) A · Skill Lab (MLOps with MLflow)

## 1. Task and models
| Item | Value |
|---|---|
| Task | Semantic search: given a scientific claim, retrieve the abstract that supports or refutes it |
| Dataset | BEIR SciFact (`BeIR/scifact` corpus + queries, `BeIR/scifact-qrels` test labels) |
| Subset | 100 test claims (seed 40) searched against the full corpus of 5,183 abstracts → `data/eval_inputs.csv` |
| Model A – current Champion | `sentence-transformers/all-MiniLM-L6-v2` (~22M, table row 21) |
| Model B – Challenger | `BAAI/bge-small-en-v1.5` (~33M, table row 22) |
| Inference settings | CPU, max_length 128, corpus batch_size 32, one query at a time, cosine similarity on normalised embeddings, top_k 10 |

Pairing "21 vs 22" is one of the suggested semantic-search pairings in the brief. BGE expects the instruction prefix `Represent this sentence for searching relevant passages: ` on queries only (not on passages), so `st_wrapper.py` adds it for Model B; MiniLM uses no prefix. The same corpus text (`title. abstract`) and the same 100 claims, in the same order, are used for both models.

## 2. How to run
```bash
pip install -r requirements.txt
python -m mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db
# in a second terminal:
python evaluate.py                       # runs A and B on the same 100 claims, logs everything, makes the decision
python register_champion.py              # registers both versions, sets @champion / @challenger aliases
python predict_from_mlflow.py            # loads models:/1MS25CI040-semantic-search-model@champion and searches data/new_data.csv
python predict_from_mlflow.py challenger # stretch goal: same script, other alias, no code change
```
MLflow experiment: `1MS25CI040-hf-semantic-search-champion-challenger` · Registered model: `1MS25CI040-semantic-search-model`
(To use another tracking server or store, set `MLFLOW_TRACKING_URI`. No run IDs are hard-coded.)
The first run downloads both models (~90 MB + ~130 MB) and the SciFact files. Encoding the 5,183 abstracts takes a few minutes per model on a CPU.

## 3. Champion rule (written before the final run)
- **Primary metric:** nDCG@10 (%) on the shared 100 claims. **Higher is better.** nDCG rewards putting the relevant abstract near the top and handles claims with more than one relevant abstract.
- **Margin:** B must beat A by **at least 2.0 nDCG@10 points**. With only 100 claims, one claim moving from rank 1 to "not found" changes nDCG@10 by up to 1 point, so a gain under 2 points could come from one or two lucky claims.
- **Guardrail:** B is rejected if its average query latency is **more than 3×** A's. **Lower latency is better.**
- **Tie-breaker:** if |Δ nDCG@10| < 2.0, keep the **faster** model.
- Secondary metrics (reported, not used for the decision): Recall@1/5/10 ↑, MRR@10 ↑, total query time ↓, corpus encoding time ↓.

## 4. Results
| Metric | A: MiniLM-L6 | B: BGE-small | B − A |
|---|---:|---:|---:|
| Recall@1 ↑ | 50.20 | 53.70 | +3.50 |
| Recall@5 ↑ | 67.70 | 78.10 | +10.40 |
| Recall@10 ↑ | 72.90 | 82.30 | +9.40 |
| MRR@10 ↑ | 58.57 | 64.77 | +6.20 |
| **nDCG@10 ↑ (primary)** | **61.57** | **68.73** | **+7.16** |
| Avg query latency (ms) ↓ | 43.60 | 79.43 | 1.82× |
| Total query time, 100 claims (s) ↓ | 4.36 | 7.94 | +3.58 |
| Corpus encoding, 5,183 abstracts (s) ↓ | 137.6 | 278.4 | +140.8 |

The two models return a different top-1 abstract for 40 of the 100 claims (`outputs/disagreements.csv`); in those, B's top-1 is correct 11 times and A's 7 times. Five are commented in `outputs/qualitative_examples.csv`.
Plots: `outputs/metric_comparison.png`, `outputs/recall_at_k_curve.png`, `outputs/first_relevant_rank_*.png`.

## 5. Decision – Challenger wins: Model B (`BAAI/bge-small-en-v1.5`) becomes Champion
Yes, the Challenger clearly improves on the Champion. Its nDCG@10 is 68.73 against 61.57, a gain of +7.16 points, more than three times the 2.0-point margin fixed before the run. Its average query latency is 1.82× MiniLM's (79 ms vs 44 ms), which is inside the 3× guardrail, so the rule promotes B. `register_champion.py` therefore gives version 2 (BGE-small) the `@champion` alias and version 1 (MiniLM) the `@challenger` alias.

Is the difference meaningful? Probably yes. The gain is not one lucky claim: B is ahead on every quality metric, and the gap grows further down the list (Recall@5 +10.4, Recall@10 +9.4), meaning BGE finds the right abstract for many claims where MiniLM misses it completely. Still, 100 claims from one biomedical dataset is a small sample, and the Recall@1 gain (+3.5) is small enough to be noise.

Trade-offs: BGE-small has about 1.5× the parameters (33M vs 22M), takes twice as long to encode the corpus and to answer a query, and needs the query instruction prefix. On my 8 hand-written claims the new Champion put the expected abstract first in 5/8 and MiniLM in 6/8 (both 8/8 in the top 3), so on easy, plain-language queries the two are close. 8 claims are far too few to change the decision, and the rule was fixed on the evaluation set.

## 6. Limitations and declaration
- Only 100 SciFact claims (biomedical), so a few points either way is within noise. Results may differ on other domains such as news or product search.
- Abstracts were truncated to 128 tokens for both models to keep CPU time reasonable; longer inputs might help BGE (trained up to 512) more than MiniLM.
- SciFact labels some claims as refuted, so the "relevant" abstract can contradict the claim. Both models only match topics; neither reasons about whether the claim is true (see q_034 in the qualitative examples).
- Latency was measured on a CPU, one query at a time, with a warm-up call excluded. It varies between runs; the ratio is more reliable than the absolute ms.
- `predict_from_mlflow.py` never calls Hugging Face directly. It loads the model through the alias URI, and the pyfunc wrapper (`st_wrapper.py`) calls sentence-transformers inside MLflow. The version and HF model ID are read from the registry tags. The corpus and its embeddings are stored as model artifacts, so the loaded model can search on its own.
- New data (`data/new_data.csv`, 8 claims I wrote in my own words, each with the abstract I expect) is not from the SciFact test claims, and none of the expected abstracts is a relevant document for any test claim.
- AI assistant used: Hark (AI assistant) helped write and debug the scripts and README. I can explain every line.

## 7. Repository structure
```
data/        eval_inputs.csv, new_data.csv
outputs/     predictions_*.csv, metrics_summary*.json, model_comparison.csv, disagreements.csv, qualitative_examples.csv,
             metric_comparison.png, recall_at_k_curve.png, first_relevant_rank_*.png,
             new_data_predictions*.csv, model_runs.json, *_terminal.txt
screenshots/ MLflow runs, compare view, summary run, registry aliases, predict_from_mlflow.py and register terminal output
evaluate.py, st_wrapper.py, register_champion.py, predict_from_mlflow.py, requirements.txt, .gitignore
```
