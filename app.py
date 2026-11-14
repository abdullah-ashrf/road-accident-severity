"""
Road Accident Risk Explorer (Streamlit)

Pick the conditions of a road accident and see how likely it is that someone
was killed or seriously injured (KSI), compared with the UK average.

Run locally:  streamlit run app.py
"""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Road Accident Risk Explorer", page_icon="🚦", layout="centered")


@st.cache_resource
def load():
    bundle = joblib.load(ROOT / "models" / "ksi_model_v2.joblib")
    sample = pd.read_parquet(ROOT / "results" / "v2" / "test_sample.parquet")
    return bundle, sample


bundle, sample = load()
model, features, cats = bundle["model"], bundle["features"], bundle["categories"]
base_rate = float(sample["KSI"].mean())

# Typical location per police force, so users pick a region instead of coordinates.
loc = sample.groupby(sample["Police_Force"].astype(str), observed=True)[["Latitude", "Longitude"]].median()


def options(col):
    return [c for c in cats[col] if c not in ("Missing", "Data missing or out of range")]


def default(col, preferred):
    opts = options(col)
    return opts.index(preferred) if preferred in opts else 0


st.title("🚦 Road Accident Risk Explorer")
st.caption("How likely is an accident under these conditions to kill or seriously injure someone? "
           "Model trained on 1.6M accidents in Great Britain (2005–2014) and tested on 2016–2017.")

c1, c2 = st.columns(2)
with c1:
    speed = st.select_slider("Speed limit (mph)", options=[20, 30, 40, 50, 60, 70], value=30)
    area = st.radio("Area", options("Urban_or_Rural_Area"), horizontal=True)
    road_type = st.selectbox("Road type", options("Road_Type"), index=default("Road_Type", "Single carriageway"))
    road_class = st.selectbox("Road class", options("1st_Road_Class"), index=default("1st_Road_Class", "A"))
    junction = st.selectbox("Junction", options("Junction_Detail"),
                            index=default("Junction_Detail", "Not at junction or within 20 metres"))
with c2:
    light = st.selectbox("Light", options("Light_Conditions"), index=default("Light_Conditions", "Daylight"))
    weather = st.selectbox("Weather", options("Weather_Conditions"),
                           index=default("Weather_Conditions", "Fine no high winds"))
    surface = st.selectbox("Road surface", options("Road_Surface_Conditions"),
                           index=default("Road_Surface_Conditions", "Dry"))
    vehicles = st.slider("Vehicles involved", 1, 6, 2)
    hour = st.slider("Hour of day", 0, 23, 17)

force = st.selectbox("Region (police force)", sorted(loc.index), index=sorted(loc.index).index("Metropolitan Police")
                     if "Metropolitan Police" in loc.index else 0)

# Values match the data exactly. In the source data, "no hazard" and "no special
# conditions" are blank (missing), and pedestrian crossings are coded 0.0 = none.
row = {f: None for f in bundle["categorical"]}
row.update({
    "1st_Road_Class": road_class, "Carriageway_Hazards": None,
    "Day_of_Week": "Friday", "Junction_Control": "Give way or uncontrolled"
    if junction != "Not at junction or within 20 metres" else "Data missing or out of range",
    "Junction_Detail": junction, "Light_Conditions": light,
    "Pedestrian_Crossing-Human_Control": "0.0",
    "Pedestrian_Crossing-Physical_Facilities": "0.0",
    "Road_Surface_Conditions": surface, "Road_Type": road_type,
    "Special_Conditions_at_Site": None, "Urban_or_Rural_Area": area,
    "Weather_Conditions": weather, "Police_Force": force,
    "Speed_limit": float(speed), "Number_of_Vehicles": vehicles, "Hour": hour, "Month": 6,
    "Latitude": float(loc.loc[force, "Latitude"]), "Longitude": float(loc.loc[force, "Longitude"]),
})
X = pd.DataFrame([row])[features].copy()
for c in bundle["categorical"]:
    X[c] = pd.Categorical(X[c], categories=cats[c])

# The model was trained with class weighting, so its raw score is a risk score,
# not a calibrated probability. Report it relative to an average accident.
score = float(model.predict_proba(X)[0, 1])
avg_score = float(model.predict_proba(sample[features])[:, 1].mean())
relative = score / avg_score

st.divider()
m1, m2 = st.columns(2)
m1.metric("Risk compared with an average accident", f"{relative:.1f}×")
band = "High" if score >= bundle["threshold"] else "Lower"
m2.metric("Risk band", band, help=f"High = above the threshold tuned on 2015 data ({bundle['threshold']:.2f}).")
st.progress(min(relative / 3, 1.0))
st.caption(f"In the test years, {base_rate:.1%} of accidents were serious or fatal. "
           "Scores are relative risk, not exact probabilities.")

with st.expander("How this works and its limits"):
    st.markdown(
        "- **Data:** UK STATS19 police accident records, 2005–2017 (public, Department for Transport).\n"
        "- **Model:** gradient-boosted trees with class weighting, judged on 2016–2017, years it never saw. "
        "Test ROC AUC 0.65; its top 10% riskiest accidents are 1.9× more likely to be serious.\n"
        "- **Limits:** it only knows road, weather, light and time. It doesn't know who was involved "
        "(motorbike, cyclist, lorry, driver age), which matters a lot. Use it to explore patterns, not to "
        "judge a single accident.\n"
        "- Built by Abdullah Ashraf. Original MSc project 2024, rebuilt 2026.")
