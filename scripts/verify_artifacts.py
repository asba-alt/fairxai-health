"""Validate every shipped model artifact end-to-end WITHOUT retraining (the same checks the
deployed service relies on). Run from the repo root:

    python scripts/verify_artifacts.py

Exits non-zero if any artifact cannot be loaded or produce a prediction + explanations.
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from backend.ml import inference as inf  # noqa: E402
from backend.ml.diseases import DISEASES  # noqa: E402


def defaults(schema):
    out = {}
    for f in schema:
        if f["type"] == "number":
            out[f["name"]] = f.get("median", f.get("min_seen", 0))
        elif f["type"] == "category":
            out[f["name"]] = f.get("mode") or f["options"][0]
        else:
            out[f["name"]] = f["options"][0]
    return out


def main():
    failures, checked = [], 0
    for d in DISEASES:
        version = inf.latest_version(d)
        if not version:
            print(f"[SKIP] {d}: no trained artifact")
            continue
        checked += 1
        try:
            meta = inf.load_meta(d, version)
            res = inf.predict(d, defaults(meta["feature_schema"]), version=version, explain=True, lime=True)
            assert 0.0 <= res["probability"] <= 1.0
            assert res["explanations"]["shap"].get("contributions"), "SHAP missing"
            assert res["explanations"]["lime"].get("items"), "LIME missing"
            assert meta["test_metrics"]["roc_auc"] is not None
            print(f"[PASS] {d:14s} {meta['algorithm']:32s} p={res['probability']:.4f} "
                  f"thr={meta['threshold']:.4f} shap={len(res['explanations']['shap']['contributions'])} "
                  f"lime={len(res['explanations']['lime']['items'])}")
        except Exception as e:  # noqa: BLE001
            failures.append(d)
            print(f"[FAIL] {d}: {type(e).__name__}: {e}")
    if checked == 0:
        print("No trained artifacts found.")
        return 1
    print(f"\n{checked - len(failures)}/{checked} shipped models verified.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
