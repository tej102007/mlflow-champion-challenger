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

## 3. Champion rule (written before the final run)
- **Primary metric:** nDCG@10 (%) on the shared 100 claims. **Higher is better.**
- **Margin:** B must beat A by **at least 2.0 nDCG@10 points**.
- **Guardrail:** B is rejected if its average query latency is **more than 3×** A's. **Lower latency is better.**
- **Tie-breaker:** if |Δ nDCG@10| < 2.0, keep the **faster** model.
