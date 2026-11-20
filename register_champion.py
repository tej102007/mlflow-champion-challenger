"""Register Model A and Model B and set the @champion / @challenger aliases from the decision logged by evaluate.py.
No run IDs are hard-coded: the latest comparison run is looked up by name.
Student: Eggonu Venkata Tejal | USN: 1MS25CI040"""
import os, mlflow
from mlflow.tracking import MlflowClient

USN = "1MS25CI040"
EXPERIMENT = f"{USN}-hf-semantic-search-champion-challenger"
REGISTERED = f"{USN}-semantic-search-model"
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
client = MlflowClient()
exp = client.get_experiment_by_name(EXPERIMENT)

parent = client.search_runs([exp.experiment_id], "attributes.run_name = 'compare_semantic_search'",
                            order_by=["attributes.start_time DESC"], max_results=1)[0]
winner = parent.data.tags["winner"].split("_")[0]          # "A" or "B"
print(f"Comparison run : {parent.info.run_id}")
print(f"Winner tag     : {parent.data.tags['winner']}")
print(f"Reason         : {parent.data.tags['decision_reason']}\n")

versions = {}
for key in ("A", "B"):
    child = client.search_runs([exp.experiment_id],
                               f"tags.mlflow.parentRunId = '{parent.info.run_id}' and attributes.run_name LIKE '{key}_%'",
                               max_results=1)[0]
    mv = mlflow.register_model(f"runs:/{child.info.run_id}/model", REGISTERED)
    hf_id = child.data.params["hf_model_id"]
    for t, v in {"hf_model_id": hf_id, "run_name": child.info.run_name,
                 "ndcg_at_10": f"{child.data.metrics['ndcg_at_10']:.2f}",
                 "avg_latency_ms": f"{child.data.metrics['avg_latency_ms']:.2f}"}.items():
        client.set_model_version_tag(REGISTERED, mv.version, t, v)
    versions[key] = (mv.version, hf_id)
    print(f"Registered {child.info.run_name:32s} -> {REGISTERED} v{mv.version}")

loser = "B" if winner == "A" else "A"
client.set_registered_model_alias(REGISTERED, "champion", versions[winner][0])
client.set_registered_model_alias(REGISTERED, "challenger", versions[loser][0])
print(f"\n@champion   -> v{versions[winner][0]}  {versions[winner][1]}")
print(f"@challenger -> v{versions[loser][0]}  {versions[loser][1]}")
