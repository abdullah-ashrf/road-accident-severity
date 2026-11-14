"""
Export tidy tables for the Power BI dashboard (powerbi/ folder).

Usage:
  python src/export_powerbi.py data/Accident_Information.csv
"""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "powerbi" / "data"

CONDITIONS = ["Speed_limit", "Urban_or_Rural_Area", "Light_Conditions", "Weather_Conditions",
              "Road_Surface_Conditions", "Road_Type", "1st_Road_Class", "Junction_Detail",
              "Day_of_Week", "Hour"]


def main(path):
    OUT.mkdir(parents=True, exist_ok=True)
    cols = ["Accident_Index", "Accident_Severity", "Year", "Date", "Time", "Police_Force",
            "Latitude", "Longitude", "Number_of_Casualties"] + [c for c in CONDITIONS if c != "Hour"]
    df = pd.read_csv(path, usecols=cols, low_memory=False, encoding="latin-1")
    df["KSI"] = df["Accident_Severity"].isin(["Fatal", "Serious"]).astype(int)
    df["Hour"] = pd.to_datetime(df["Time"], format="%H:%M", errors="coerce").dt.hour

    # 1. Trend: accidents by year and severity
    trend = df.groupby(["Year", "Accident_Severity"]).size().rename("Accidents").reset_index()
    trend.to_csv(OUT / "trend_by_year_severity.csv", index=False)

    # 2. KSI rate by condition (long format: one slicer for condition type)
    rows = []
    for c in CONDITIONS:
        g = df.groupby(df[c].astype(str))["KSI"].agg(["size", "mean"]).reset_index()
        g.columns = ["Value", "Accidents", "KSI_Rate"]
        g.insert(0, "Condition", c.replace("_", " "))
        rows.append(g[g["Accidents"] >= 500])
    pd.concat(rows).to_csv(OUT / "ksi_rate_by_condition.csv", index=False)

    # 3. Map: by police force
    force = df.groupby("Police_Force").agg(
        Accidents=("KSI", "size"), KSI_Rate=("KSI", "mean"),
        Latitude=("Latitude", "median"), Longitude=("Longitude", "median")).reset_index()
    force.to_csv(OUT / "ksi_by_police_force.csv", index=False)

    # 4. Model results (v1 rebuild and v2)
    v1 = json.loads((ROOT / "results" / "metrics.json").read_text())["results"]
    v2 = json.loads((ROOT / "results" / "v2" / "metrics.json").read_text())["results"]
    m = [{"Stage": "Part 1 (London, 3 classes)", "Model": r["model"], "Accuracy": r["accuracy"],
          "Balanced_Accuracy": r["balanced_accuracy"], "Recall_Fatal": r["recall_fatal"],
          "Recall_Serious": r["recall_serious"]} for r in v1]
    pd.DataFrame(m).to_csv(OUT / "model_comparison_part1.csv", index=False)
    pd.DataFrame(v2).to_csv(OUT / "model_results_v2.csv", index=False)

    # 5. Feature importance
    imp = json.loads((ROOT / "results" / "v2" / "explain.json").read_text())["importance"]
    pd.Series(imp, name="Importance").rename_axis("Feature").reset_index().to_csv(
        OUT / "feature_importance.csv", index=False)

    # 6. Scored test sample (for a "risk score" page)
    bundle = joblib.load(ROOT / "models" / "ksi_model_v2.joblib")
    s = pd.read_parquet(ROOT / "results" / "v2" / "test_sample.parquet")
    s["Risk_Score"] = bundle["model"].predict_proba(s[bundle["features"]])[:, 1]
    s["Risk_Band"] = pd.qcut(s["Risk_Score"], 5, labels=["Very low", "Low", "Medium", "High", "Very high"])
    s.to_csv(OUT / "scored_test_sample.csv", index=False)

    for f in sorted(OUT.glob("*.csv")):
        print(f"{f.name}: {sum(1 for _ in open(f, encoding='utf-8')) - 1:,} rows")


if __name__ == "__main__":
    main(sys.argv[1])
