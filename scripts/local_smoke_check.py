"""Local smoke check across all shipped model artifacts (uses the real artifacts/ and in-memory DB unless MONGO_URI is set)."""
from backend.app import create_app

app = create_app()
c = app.test_client()
h = c.get("/api/health").get_json()
print("health", h["ok"], h["data"]["database"], h["data"]["models_loaded"])

for d in ["cvd", "diabetes", "breast_cancer", "ckd", "tb"]:
    r = c.get(f"/api/models/{d}/schema")
    if r.status_code != 200:
        print(d, "schema", r.status_code, r.get_json()["error"]["code"])
        continue
    f = r.get_json()["data"]["fields"]
    inp = {x["name"]: (x["median"] if x["type"] == "number" else x["mode"] if x["type"] == "category" else x["options"][0]) for x in f}
    p = c.post(f"/api/predict/{d}", json={"inputs": inp, "lime": True})
    j = p.get_json()
    if not j["ok"]:
        print(d, "predict FAIL", p.status_code, j["error"])
        continue
    data = j["data"]
    sh = data["explanations"].get("shap", {})
    li = data["explanations"].get("lime", {})
    print(d, "prob", round(data["probability"], 4), "flag", data["flagged"],
          "shap", len(sh.get("contributions", [])), sh.get("error"),
          "lime", len(li.get("items", [])), li.get("error"), "pid", bool(data["prediction_id"]))
    fa = c.get(f"/api/fairness/{d}").get_json()["data"]
    print("   fairness", fa["status"], [a["attribute"] for a in fa.get("attributes", [])])

print("history", c.get("/api/predictions").get_json()["data"]["count"])
print("summary trained", c.get("/api/dashboard/summary").get_json()["data"]["trained"])
