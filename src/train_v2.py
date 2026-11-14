"""
Road Accident Severity v2 (2026 rebuild)

Question: given the conditions of a road accident, how likely is it that
someone was killed or seriously injured (KSI)? KSI is the UK Department for
Transport's own headline road-safety measure.

What changed from the 2024 MSc version:
- Target: KSI (Fatal + Serious) vs Slight, instead of 3 heavily imbalanced classes.
- Data: the whole of Great Britain (2005-2017), not only London.
- Split by time: train 2005-2014, tune on 2015, test on 2016-2017. The model is
  judged on years it has never seen, the way it would be used in practice.
- Model: gradient-boosted trees (scikit-learn HistGradientBoosting, the same
  family as LightGBM) with native categorical handling and class weighting.
- Decision threshold tuned on the validation year, not left at 0.5.
- Metrics that suit a rare outcome: ROC AUC, PR AUC against the base rate,
  recall and precision at the chosen threshold, and lift in the top 10%.
- Leakage check: "Did a police officer attend" is dropped, because police
  attend more often *because* an accident is serious.

Usage:
  python src/train_v2.py data/Accident_Information.csv [data/Vehicle_Information.csv]
"""
import json
import sys
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "v2"
MODEL_DIR = ROOT / "models"
SEED = 42

ACC_CAT = [
    "1st_Road_Class", "Carriageway_Hazards", "Day_of_Week", "Junction_Control",
    "Junction_Detail", "Light_Conditions", "Pedestrian_Crossing-Human_Control",
    "Pedestrian_Crossing-Physical_Facilities", "Road_Surface_Conditions",
    "Road_Type", "Special_Conditions_at_Site", "Urban_or_Rural_Area",
    "Weather_Conditions", "Police_Force",
]
ACC_NUM = ["Speed_limit", "Number_of_Vehicles", "Hour", "Month", "Latitude", "Longitude"]


def load_accidents(path):
    usecols = ACC_CAT + ["Accident_Index", "Accident_Severity", "Speed_limit",
                         "Number_of_Vehicles", "Time", "Date", "Year",
                         "Latitude", "Longitude"]
    df = pd.read_csv(path, usecols=usecols, low_memory=False, encoding="latin-1")
    df["KSI"] = df["Accident_Severity"].isin(["Fatal", "Serious"]).astype(int)
    df["Hour"] = pd.to_datetime(df["Time"], format="%H:%M", errors="coerce").dt.hour
    df["Month"] = pd.to_datetime(df["Date"], errors="coerce").dt.month
    df = df.dropna(subset=["Speed_limit", "Year"])
    df = df[df["Number_of_Vehicles"] <= 9]
    return df


def add_vehicles(df, path):
    """Summarise the vehicles in each accident: what hit what, and who drove."""
    v = pd.read_csv(path, low_memory=False, encoding="latin-1",
                    usecols=["Accident_Index", "Vehicle_Type", "Age_Band_of_Driver",
                             "Sex_of_Driver", "Vehicle_Manoeuvre", "Skidding_and_Overturning",
                             "Hit_Object_off_Carriageway", "Hit_Object_in_Carriageway",
                             "Vehicle_Leaving_Carriageway", "Age_of_Vehicle", "Engine_Capacity_.CC."])
    vt = v["Vehicle_Type"].fillna("").str.lower()
    flags = pd.DataFrame({
        "Accident_Index": v["Accident_Index"],
        "has_motorcycle": vt.str.contains("motorcycle").astype(int),
        "has_pedal_cycle": vt.str.contains("pedal cycle").astype(int),
        "has_hgv": vt.str.contains("goods 7.5|goods over").astype(int),
        "has_bus": vt.str.contains("bus|coach").astype(int),
        "young_driver": v["Age_Band_of_Driver"].isin(["16 - 20", "21 - 25"]).astype(int),
        "older_driver": v["Age_Band_of_Driver"].isin(["66 - 75", "Over 75"]).astype(int),
        "overturned": v["Skidding_and_Overturning"].fillna("").str.contains("Overturn").astype(int),
        "left_carriageway": (~v["Vehicle_Leaving_Carriageway"].fillna("Did not leave carriageway")
                             .isin(["Did not leave carriageway", "Data missing or out of range"])).astype(int),
        "hit_object_off_road": (~v["Hit_Object_off_Carriageway"].fillna("None")
                                .isin(["None", "Data missing or out of range"])).astype(int),
        "head_on_risk": v["Vehicle_Manoeuvre"].fillna("").str.contains("Overtaking|Going ahead other").astype(int),
        "vehicle_age": pd.to_numeric(v["Age_of_Vehicle"], errors="coerce"),
        "engine_cc": pd.to_numeric(v["Engine_Capacity_.CC."], errors="coerce"),
    })
    agg = flags.groupby("Accident_Index").agg({
        **{c: "max" for c in flags.columns if c not in ("Accident_Index", "vehicle_age", "engine_cc")},
        "vehicle_age": "mean", "engine_cc": "max",
    }).reset_index()
    out = df.merge(agg, on="Accident_Index", how="left")
    print(f"Vehicle summary joined to {out['has_motorcycle'].notna().mean():.1%} of accidents")
    return out, [c for c in agg.columns if c != "Accident_Index"]


def lift_at(y, p, top=0.10):
    n = int(len(p) * top)
    idx = np.argsort(-p)[:n]
    return y[idx].mean() / y.mean()


def evaluate(name, y, p, thr):
    pred = (p >= thr).astype(int)
    return {
        "set": name, "rows": int(len(y)), "ksi_rate": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "threshold": float(thr),
        "recall": float(recall_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred)),
        "lift_top10": float(lift_at(y, p)),
    }


def main(acc_path, veh_path=None):
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)

    df = load_accidents(acc_path)
    num = list(ACC_NUM)
    if veh_path:
        df, veh_cols = add_vehicles(df, veh_path)
        num += veh_cols
    print(f"Accidents: {len(df):,} | KSI rate {df['KSI'].mean():.1%} | {time.time()-t0:.0f}s")

    features = ACC_CAT + num
    for c in ACC_CAT:
        # Some columns mix numeric codes and text, so turn everything into text first.
        # Blanks (e.g. "None" for no hazard, which pandas reads as missing) stay missing;
        # the model treats missing as its own branch.
        df[c] = df[c].map(lambda v: v if pd.isna(v) else str(v)).astype(object).astype("category")

    train = df[df["Year"] <= 2014]
    val = df[df["Year"] == 2015]
    test = df[df["Year"] >= 2016]
    print(f"Train {len(train):,} (2005-14) | Validate {len(val):,} (2015) | Test {len(test):,} (2016-17)")

    model = HistGradientBoostingClassifier(
        categorical_features="from_dtype", class_weight="balanced",
        learning_rate=0.06, max_iter=600, max_leaf_nodes=63, min_samples_leaf=200,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=30, random_state=SEED)
    model.fit(train[features], train["KSI"])
    print(f"Trained {model.n_iter_} trees in {time.time()-t0:.0f}s")

    p_val = model.predict_proba(val[features])[:, 1]
    prec, rec, thr = precision_recall_curve(val["KSI"], p_val)
    f1s = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-9, None)
    best_thr = float(thr[np.argmax(f1s)])

    p_test = model.predict_proba(test[features])[:, 1]
    res = [evaluate("validation_2015", val["KSI"].to_numpy(), p_val, best_thr),
           evaluate("test_2016_2017", test["KSI"].to_numpy(), p_test, best_thr)]

    # Charts
    yt = test["KSI"].to_numpy()
    fpr, tpr, _ = roc_curve(yt, p_test)
    pr, rc, _ = precision_recall_curve(yt, p_test)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(fpr, tpr); ax[0].plot([0, 1], [0, 1], "--", color="grey")
    ax[0].set(title=f"ROC (test 2016-17), AUC {res[1]['roc_auc']:.3f}", xlabel="False positive rate", ylabel="True positive rate")
    ax[1].plot(rc, pr); ax[1].axhline(yt.mean(), ls="--", color="grey", label=f"Base rate {yt.mean():.1%}")
    ax[1].set(title=f"Precision-recall, PR AUC {res[1]['pr_auc']:.3f}", xlabel="Recall", ylabel="Precision"); ax[1].legend()
    fig.tight_layout(); fig.savefig(OUT / "roc_pr_test.png", dpi=150); plt.close(fig)

    joblib.dump({"model": model, "features": features, "categorical": ACC_CAT,
                 "numeric": num, "threshold": best_thr,
                 "categories": {c: list(df[c].cat.categories) for c in ACC_CAT}},
                MODEL_DIR / "ksi_model_v2.joblib")
    test.sample(20000, random_state=SEED)[features + ["KSI"]].to_parquet(OUT / "test_sample.parquet")

    (OUT / "metrics.json").write_text(json.dumps({
        "with_vehicle_features": bool(veh_path), "features": features,
        "trees": int(model.n_iter_), "results": res}, indent=2))
    lines = ["| Set | Rows | KSI rate | ROC AUC | PR AUC | Recall | Precision | Lift top 10% |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in res:
        lines.append(f"| {r['set']} | {r['rows']:,} | {r['ksi_rate']:.1%} | {r['roc_auc']:.3f} | "
                     f"{r['pr_auc']:.3f} | {r['recall']:.1%} | {r['precision']:.1%} | {r['lift_top10']:.2f}x |")
    (OUT / "metrics.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"Done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        sys.exit("Usage: python src/train_v2.py data/Accident_Information.csv [data/Vehicle_Information.csv]")
    main(*sys.argv[1:])
