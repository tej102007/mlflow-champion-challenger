"""Load the Champion through its MLflow registry alias and search new claims (no Hugging Face code here).
Usage: python predict_from_mlflow.py            -> uses @champion
       python predict_from_mlflow.py challenger -> uses @challenger
Student: Eggonu Venkata Tejal | USN: 1MS25CI040"""
import os, sys, mlflow, pandas as pd
from mlflow.tracking import MlflowClient

REGISTERED = "1MS25CI040-semantic-search-model"
ALIAS = sys.argv[1] if len(sys.argv) > 1 else "champion"
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))

mv = MlflowClient().get_model_version_by_alias(REGISTERED, ALIAS)
uri = f"models:/{REGISTERED}@{ALIAS}"
print(f"Loading {uri}")
print(f"  -> version {mv.version} | hf_model_id = {mv.tags.get('hf_model_id')} | run = {mv.tags.get('run_name')}")
model = mlflow.pyfunc.load_model(uri)

new = pd.read_csv("data/new_data.csv", dtype=str)
out = model.predict(new[["query"]])                       # top-3 abstracts per query
out = out.merge(new.rename(columns={"query": "query"}), on="query")
out["is_expected"] = (out.doc_id == out.expected_doc_id).astype(int)
out["model_version"] = mv.version; out["hf_model_id"] = mv.tags.get("hf_model_id")
out = out[["sample_id", "query", "expected_doc_id", "rank", "doc_id", "title", "score", "is_expected",
           "model_version", "hf_model_id"]]
os.makedirs("outputs", exist_ok=True)
fname = f"outputs/new_data_predictions{'' if ALIAS == 'champion' else '_' + ALIAS}.csv"
out.to_csv(fname, index=False)

top1 = out[out["rank"] == 1].copy()
top1["title"] = top1.title.str.slice(0, 55)
top1["query"] = top1["query"].str.slice(0, 50)
hit3 = out.groupby("sample_id").is_expected.max()
pd.set_option("display.width", 200)
print(top1[["sample_id", "query", "doc_id", "title", "score", "is_expected"]].to_string(index=False))
print(f"\nExpected abstract at rank 1: {top1.is_expected.sum()} / {len(top1)}   |   in top 3: {hit3.sum()} / {len(hit3)}")
print(f"Saved {fname}")
