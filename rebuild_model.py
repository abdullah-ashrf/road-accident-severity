"""
Road Accident Severity: model rebuild (v2)

Why this exists:
The original MSc model (2024) reported 88.34% accuracy, but it predicted
"Slight" for every accident. 88% of London accidents in the data are slight,
so it scored 88% without learning anything (ROC AUC 0.51, recall on fatal
and serious accidents = 0).

This script keeps the same data and cleaning steps as the dissertation and
fixes the modelling:
  1. One-hot encodes categories (the original turned them into numbers 0..n,
     which tells the model that "Rain" is bigger than "Fine").
  2. Uses a stratified train/test split, so every class appears in both sets.
  3. Weights the rare classes (fatal, serious) so the model can't ignore them.
  4. Reports the metrics that matter for imbalanced data: recall per class,
     balanced accuracy and macro F1, next to plain accuracy.

Usage:
  python rebuild_model.py path/to/Accident_Information.csv
Outputs go to ./results/
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

RANDOM_STATE = 42
CLASSES = ["Fatal", "Serious", "Slight"]
OUT = Path(__file__).resolve().parent / "results"

LONDON = [
    "Kensington and Chelsea", "Westminster", "Hammersmith and Fulham",
    "City of London", "Tower Hamlets", "Southwark", "Hackney", "Islington",
    "Camden", "Lambeth", "Brent", "Haringey", "Barnet", "Ealing",
    "Richmond upon Thames", "Waltham Forest", "Newham",
    "London Airport (Heathrow)", "Hillingdon", "Harrow", "Hounslow", "Enfield",
    "Redbridge", "Barking and Dagenham", "Havering", "Croydon", "Wandsworth",
    "Sutton", "Lewisham", "Bromley", "Greenwich", "Bexley",
    "Kingston upon Thames", "Merton",
]

# Same columns the dissertation dropped before modelling.
DROP = [
    "Date", "Time", "Latitude", "Longitude", "Special_Conditions_at_Site",
    "Did_Police_Officer_Attend_Scene_of_Accident", "Carriageway_Hazards",
    "Accident_Index", "2nd_Road_Class", "1st_Road_Number", "2nd_Road_Number",
    "InScotland", "Police_Force", "Local_Authority_(Highway)",
    "Location_Easting_OSGR", "Location_Northing_OSGR",
    "LSOA_of_Accident_Location", "Pedestrian_Crossing-Human_Control",
    "Pedestrian_Crossing-Physical_Facilities", "1st_Road_Class",
]
NUMERIC = ["Number_of_Vehicles", "Number_of_Casualties", "Speed_limit", "Year", "Month"]


def time_frame(t):
    if pd.isna(t):
        return "Unknown"
    h = t.hour
    return "0-6" if h < 6 else "6-12" if h < 12 else "12-18" if h < 18 else "18-24"


def load(path):
    df = pd.read_csv(path, low_memory=False, encoding="latin-1")
    print(f"Loaded {len(df):,} rows, {df.shape[1]} columns")
    df = df[df["Local_Authority_(District)"].isin(LONDON)].copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Month"] = df["Date"].dt.month
    df["Time_Frame"] = pd.to_datetime(df["Time"], format="%H:%M", errors="coerce").apply(time_frame)
    df = df.dropna(subset=["Speed_limit"])
    df = df[(df["Number_of_Vehicles"] <= 9) & (df["Number_of_Casualties"] <= 9)]
    df = df.drop(columns=[c for c in DROP if c in df.columns])
    print(f"London rows after cleaning: {len(df):,}")
    return df


def evaluate(name, model, X_test, y_test):
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)
    rec = recall_score(y_test, pred, labels=CLASSES, average=None, zero_division=0)
    m = {
        "model": name,
        "accuracy": accuracy_score(y_test, pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, pred),
        "macro_f1": f1_score(y_test, pred, average="macro", zero_division=0),
        "roc_auc_macro_ovr": roc_auc_score(y_test, proba, multi_class="ovr",
                                           average="macro", labels=list(model.classes_)),
        "recall_fatal": rec[0], "recall_serious": rec[1], "recall_slight": rec[2],
        "confusion_matrix": confusion_matrix(y_test, pred, labels=CLASSES).tolist(),
    }
    print(f"\n=== {name} ===")
    print(classification_report(y_test, pred, labels=CLASSES, zero_division=0))
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y_test, pred, labels=CLASSES, ax=ax,
                                            cmap="Blues", colorbar=False)
    ax.set_title(name)
    fig.tight_layout()
    fig.savefig(OUT / f"confusion_{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.png", dpi=150)
    plt.close(fig)
    return m


def main(path):
    OUT.mkdir(exist_ok=True)
    df = load(path)
    y = df["Accident_Severity"].astype(str)
    X = df.drop(columns=["Accident_Severity"])
    categorical = [c for c in X.columns if c not in NUMERIC]
    X[categorical] = X[categorical].astype(str)
    print("Class share:\n", y.value_counts(normalize=True).round(4))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    onehot = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), categorical),
        ("num", StandardScaler(), NUMERIC),
    ])
    ordinal = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), categorical),
        ("num", "passthrough", NUMERIC),
    ])

    models = {
        "Baseline (always Slight)": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression (original, unweighted)": Pipeline([
            ("prep", onehot), ("clf", LogisticRegression(max_iter=2000))]),
        "Logistic Regression (balanced)": Pipeline([
            ("prep", onehot), ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))]),
        "Gradient Boosting (balanced)": Pipeline([
            ("prep", ordinal),
            ("clf", HistGradientBoostingClassifier(
                categorical_features=list(range(len(categorical))),
                class_weight="balanced", max_iter=300, learning_rate=0.08,
                early_stopping=True, random_state=RANDOM_STATE))]),
    }

    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        results.append(evaluate(name, model, X_test, y_test))

    (OUT / "metrics.json").write_text(json.dumps({
        "rows_used": int(len(df)), "test_rows": int(len(y_test)),
        "class_share": y.value_counts(normalize=True).round(4).to_dict(),
        "results": results}, indent=2))

    lines = ["| Model | Accuracy | Balanced acc. | Macro F1 | ROC AUC | Recall fatal | Recall serious | Recall slight |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in results:
        lines.append(f"| {r['model']} | {r['accuracy']:.1%} | {r['balanced_accuracy']:.1%} | "
                     f"{r['macro_f1']:.3f} | {r['roc_auc_macro_ovr']:.3f} | {r['recall_fatal']:.1%} | "
                     f"{r['recall_serious']:.1%} | {r['recall_slight']:.1%} |")
    (OUT / "metrics.md").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python rebuild_model.py path/to/Accident_Information.csv")
    main(sys.argv[1])
