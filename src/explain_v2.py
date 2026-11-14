"""
Explain the v2 KSI model: which conditions push an accident towards
"killed or seriously injured"?

Uses SHAP when it supports the model; otherwise falls back to permutation
importance (how much ROC AUC drops when a feature is shuffled).

Usage:
  python src/explain_v2.py
"""
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "v2"


def main():
    bundle = joblib.load(ROOT / "models" / "ksi_model_v2.joblib")
    model, features = bundle["model"], bundle["features"]
    sample = pd.read_parquet(OUT / "test_sample.parquet")
    X, y = sample[features], sample["KSI"]
    for c in bundle["categorical"]:
        X[c] = pd.Categorical(X[c].astype(str), categories=bundle["categories"][c])

    method = "permutation"
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        sv = explainer(X.iloc[:5000])
        shap.plots.beeswarm(sv, max_display=15, show=False)
        plt.tight_layout(); plt.savefig(OUT / "shap_beeswarm.png", dpi=150); plt.close()
        imp = pd.Series(np.abs(sv.values).mean(0), index=features)
        method = "shap"
    except Exception as e:  # SHAP may not support native categorical splits
        print(f"SHAP not available for this model ({type(e).__name__}: {e}); using permutation importance")
        r = permutation_importance(model, X, y, scoring="roc_auc", n_repeats=5,
                                   random_state=42, n_jobs=-1)
        imp = pd.Series(r.importances_mean, index=features)

    imp = imp.sort_values(ascending=False)
    top = imp.head(15)[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(top.index, top.values, color="#2b6cb0")
    ax.set_title("What drives KSI risk (top 15)" + (" - mean |SHAP|" if method == "shap" else " - drop in ROC AUC"))
    fig.tight_layout(); fig.savefig(OUT / "feature_importance.png", dpi=150); plt.close(fig)

    # Plain-English effect of key conditions: KSI rate in the test sample by group.
    effects = {}
    for col in ["Speed_limit", "Urban_or_Rural_Area", "Light_Conditions", "1st_Road_Class",
                "has_motorcycle", "has_pedal_cycle", "young_driver", "overturned"]:
        if col in sample.columns:
            g = sample.groupby(sample[col].astype(str), observed=True)["KSI"].agg(["mean", "size"])
            effects[col] = {k: {"ksi_rate": round(float(r["mean"]), 4), "n": int(r["size"])}
                            for k, r in g[g["size"] >= 200].iterrows()}

    (OUT / "explain.json").write_text(json.dumps({
        "method": method, "importance": imp.round(5).to_dict(), "ksi_rate_by_group": effects}, indent=2))
    print(f"Method: {method}")
    print(imp.head(15).round(4).to_string())


if __name__ == "__main__":
    main()
