# AIAP Module 6 Mini-Project 2

Predicts whether a hotel booking will be a **no-show** (binary classification). Dataset: place `mini_project_2_data.csv` in `data/` (not tracked by git).

## Structure

```
data/              raw dataset (gitignored)
EDA.ipynb          exploratory data analysis
main.py            pipeline entry point
models/            saved model artifacts (gitignored)
reports/           metrics JSON and figures
src/               pipeline modules + config.yaml
tests/             pytest suite
```

## Setup

```bash
pip install -r requirements-dev.txt
python main.py
pytest
```

## Pipeline (`python main.py`)

Predicts `no_show` (1 = no-show) from the 10 features chosen in `EDA.ipynb` §10.5. All settings live in `src/config.yaml`.

1. **Validate** the config and the raw data; write `reports/data_validation_report.json`.
2. **Clean** (`src/data_preparation.py`, following the EDA):
   - Drop the one empty row (missing target) and `booking_id`.
   - Title-case the categorical columns (`arrival_month` had 88 casing variants of 12 months).
   - `num_adults`: `one` / `two` → 1 / 2.
   - `checkout_day`: take the absolute value (5,696 sign errors).
   - `price`: parse the `SGD$` / `USD$` text into `price_sgd` (USD × 1.37).
   - Add `stay_nights` (assumed year 2000; stays crossing the new year handled) and `lead_time_months` ((arrival − booking month) mod 12).
3. **Split** 80/10/10 into train/validation/test, **stratified** on `no_show` (37% no-shows in every split).
4. **Pipeline** for every model (`imblearn.pipeline.Pipeline`):
   - `RoomPriceImputer`, learned from the training data only: missing `price_sgd` → the median for the same room × branch; missing `room` → the room whose price band (per branch) contains the price.
   - Preprocess: `StandardScaler` for `lead_time_months`, `stay_nights`, `price_sgd`, `num_adults`, `num_children`; one-hot for `country`, `branch`, `first_time`, `arrival_month`, `room`. Unlisted columns are dropped.
   - Sampler (training folds only): none, `RandomUnderSampler` or `RandomOverSampler`.
5. **Train**:
   - Baselines: dummy (most frequent class) and default Logistic Regression.
   - Tuned with `GridSearchCV` (5-fold, ROC-AUC): Logistic Regression, KNN, Decision Tree, Random Forest, Gradient Boosting. Each grid also compares 4 **imbalance strategies**: none, `class_weight='balanced'` (LR / DT / RF only), undersampling, oversampling.
6. **Select** the highest validation ROC-AUC, then **tune the decision threshold** on validation (best no-show F1). Refit on train + val and evaluate once on test, at 0.5 and at the tuned threshold. Compute permutation feature importance.
7. **Save**:
   - `models/best_pipeline.joblib`
   - `reports/*.json`
   - `reports/figures/*.png`

A full run takes about 24 minutes on 16 cores, mostly Random Forest (~5.5 min) and Gradient Boosting (~14.5 min).

## Results and discussion

### Model performance (validation set)

| Model | Imbalance strategy chosen | ROC-AUC | PR-AUC | F1 | Precision | Recall |
|---|---|---|---|---|---|---|
| **Random Forest** (selected) | class_weight | **0.790** | **0.707** | **0.643** | 0.643 | 0.644 |
| Gradient Boosting | none | 0.782 | 0.690 | 0.611 | 0.709 | 0.537 |
| Decision Tree | oversample | 0.773 | 0.697 | 0.629 | 0.605 | 0.656 |
| KNN | none | 0.764 | 0.670 | 0.595 | 0.695 | 0.521 |
| Logistic Regression (tuned) | class_weight | 0.753 | 0.632 | 0.624 | 0.599 | 0.651 |
| Logistic Regression (baseline) | none | 0.753 | 0.635 | 0.592 | 0.670 | 0.530 |
| Dummy (always "show") | none | 0.500 | 0.370 | 0.000 | 0.000 | 0.000 |

F1, precision and recall use the default 0.5 threshold. PR-AUC's chance level is the no-show rate, 0.37.

Selected model: Random Forest, 200 trees, `max_depth=20`, `min_samples_leaf=10`, `class_weight='balanced'`.

### Test set (Random Forest refit on train + validation)

| | Threshold 0.5 | Tuned threshold 0.42 | Dummy |
|---|---|---|---|
| ROC-AUC | 0.795 | 0.795 | 0.500 |
| PR-AUC | 0.717 | 0.717 | 0.370 |
| Precision (no-show) | 0.645 | 0.577 | 0.000 |
| Recall (no-show) | 0.655 | **0.749** | 0.000 |
| F1 (no-show) | 0.650 | 0.652 | 0.000 |
| Accuracy | 0.739 | 0.703 | 0.630 |
| No-shows caught / missed | 2,897 / 1,525 | **3,310 / 1,112** | 0 / 4,422 |
| False alarms | 1,597 | 2,429 | 0 |

Test scores match validation (ROC-AUC 0.795 vs 0.790), so the model is not overfitting.

**Threshold choice.** Lowering the threshold to 0.42 barely changes F1 (0.650 → 0.652), but it changes *which* errors the model makes: it catches 413 more no-shows at the cost of 832 more false alarms. Which threshold is better depends on the hotel's costs:
- If a missed no-show (an empty room) costs more than contacting a guest who would have come, use the lower threshold.
- If false alarms are costly (e.g. overbooking real guests), use 0.5 or higher.

The threshold is a business decision; the model's ranking quality (ROC-AUC 0.795) is the same either way.

### Imbalance handling

Every strategy won for at least one model, and none was clearly best:
- Random Forest and Logistic Regression did best with **class weights**.
- The Decision Tree did best with **oversampling**.
- KNN and Gradient Boosting did best with **no rebalancing**.

At a 63/37 ratio the imbalance is moderate, so rebalancing mainly shifts the predicted probabilities, and a tuned threshold can do the same job. Undersampling never won: it discards about a quarter of the training data.

### Feature importance (permutation, drop in test ROC-AUC)

| Feature | Importance | EDA strength (§9) |
|---|---|---|
| `country` | **0.233** | strongest (V = 0.34) |
| `arrival_month` | 0.040 | moderate (V = 0.07) |
| `branch` | 0.040 | strong (V = 0.14) |
| `lead_time_months` | 0.030 | moderate (ρ = 0.09) |
| `stay_nights` | 0.027 | moderate (ρ = 0.07) |
| `first_time` | 0.015 | moderate (V = 0.09) |
| `room` | 0.013 | moderate (V = 0.07) |
| `price_sgd` | 0.009 | weak; mostly explained by branch × room |
| `num_adults` | 0.000 | very weak |
| `num_children` | 0.000 | very weak |

The ranking agrees with the EDA. `country` dominates: shuffling it alone costs 0.23 ROC-AUC, more than all the other features together. `num_adults` and `num_children` contribute nothing and could be dropped. `price_sgd` adds little beyond branch and room, as §10.3 predicted.

### Possible next steps

- **Drop `num_adults` / `num_children`.** They have zero importance, and dropping them simplifies the model.
- **Pick the threshold from costs.** Ask the hotel to estimate the cost of a missed no-show vs a false alarm, and choose the threshold that minimises total cost.
- **Probability calibration.** If the predicted probabilities themselves will be used (e.g. expected no-shows per night), calibrate them, since class weighting distorts them.
- **Faster runs.** Gradient Boosting takes over half the run time; `HistGradientBoostingClassifier` would be much faster on 95k rows.
