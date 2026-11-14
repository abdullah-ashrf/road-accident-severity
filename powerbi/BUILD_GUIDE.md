# Power BI dashboard: build guide (about 1 hour)

All the data is ready in `powerbi/data/`. You only build the visuals.

## 0. Install
Install **Power BI Desktop** (free) from the Microsoft Store. Search "Power BI Desktop".

## 1. Load the data
**Home → Get data → Text/CSV**, and load each file from `powerbi/data/`:

| File | What it holds |
| --- | --- |
| `trend_by_year_severity.csv` | Accidents per year by severity (2005–2017) |
| `ksi_rate_by_condition.csv` | Share of serious or fatal accidents (KSI rate) for each condition: speed, light, weather, road type… |
| `ksi_by_police_force.csv` | KSI rate and location for each police force (for the map) |
| `model_comparison_part1.csv` | The 2024 model vs the rebuilt models |
| `model_results_v2.csv` | v2 results on 2015 and 2016–17 |
| `feature_importance.csv` | What drives KSI risk |
| `scored_test_sample.csv` | 20,000 test accidents with the model's risk score and band |

In Power Query, check that the numeric columns are set as numbers. Then click **Close & Apply**.

## 2. Measures (Modeling → New measure)
```DAX
Total Accidents = SUM(trend_by_year_severity[Accidents])
KSI Accidents = CALCULATE([Total Accidents], trend_by_year_severity[Accident_Severity] IN {"Fatal", "Serious"})
KSI Rate = DIVIDE([KSI Accidents], [Total Accidents])
Actual KSI in Band = AVERAGE(scored_test_sample[KSI])
```
Format `KSI Rate`, `Actual KSI in Band` and the `KSI_Rate` columns as **Percentage**.

## 3. Pages

**Page 1: Overview**
- Cards: Total Accidents (2.05M), KSI Rate (about 15%)
- Stacked column chart: Year × Accidents, legend = Accident_Severity
- Title: *"2 million UK road accidents, 2005–2017"*

**Page 2: What makes a crash serious?**
- Slicer: `ksi_rate_by_condition[Condition]` (single select, default "Light Conditions")
- Bar chart: Value × KSI_Rate, sorted descending, with data labels
- Text box: *"Unlit roads at night: 27.5% of accidents are serious, vs 16.9% in daylight."*

**Page 3: Map**
- Map or Azure Map: Latitude/Longitude, bubble size = Accidents, colour = KSI_Rate
- Table: Police_Force, Accidents, KSI_Rate, sorted by KSI_Rate

**Page 4: The model**
- Table: `model_comparison_part1` (Model, Accuracy, Balanced_Accuracy, Recall_Fatal, Recall_Serious)
- Text box: *"The 2024 model scored 88% by always guessing 'Slight'. The rebuild catches 43% of fatal accidents (was 0%)."*
- Cards from `model_results_v2` (filter set = test_2016_2017): roc_auc 0.648, recall 48.9%, lift_top10 1.93×
- Bar chart: `feature_importance` top 10

**Page 5: Risk bands**
- Column chart: Risk_Band × Actual KSI in Band. It should rise from "Very low" to "Very high", which shows the model ranks risk correctly.

## 4. Finish
- Use the same colours on every page (for example, dark blue for KSI, grey for slight).
- **File → Save as** `powerbi/Road_Accident_Dashboard.pbix`
- **File → Export → Export to PDF**, plus screenshots of pages 2 and 4 saved to `powerbi/` for the README and your LinkedIn Featured section.

To rebuild the data: `python src/export_powerbi.py data/Accident_Information.csv`
