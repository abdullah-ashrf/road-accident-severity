| Model | Accuracy | Balanced acc. | Macro F1 | ROC AUC | Recall fatal | Recall serious | Recall slight |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline (always Slight) | 88.3% | 33.3% | 0.313 | 0.500 | 0.0% | 0.0% | 100.0% |
| Logistic Regression (original, unweighted) | 88.3% | 33.3% | 0.313 | 0.656 | 0.0% | 0.0% | 100.0% |
| Logistic Regression (balanced) | 51.0% | 45.7% | 0.297 | 0.620 | 53.9% | 29.6% | 53.7% |
| Gradient Boosting (balanced) | 56.7% | 46.3% | 0.323 | 0.646 | 43.4% | 36.0% | 59.5% |
