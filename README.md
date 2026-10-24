# cost-adaptive-pdm

Reproducible code and outputs for the manuscript **"Machine-Learning-Based Predictive Maintenance and Cost–Risk Trade-off in Automated Production Systems"** (J. Engineering, Project, and Production Management, manuscript PZM0150).

## Data
AI4I 2020 Predictive Maintenance Dataset (Matzka, 2020), UCI Machine Learning Repository, https://doi.org/10.24432/C5HS5C (CC BY 4.0).
The dataset is **not redistributed** in this repository. On first run, `code/run_analysis.py` downloads the original
`ai4i2020.csv` from UCI into `data/` and verifies its MD5 checksum (`f0adaa9c5e25366c6df1cc2684479871`), so the analysis
always runs on the unmodified official file.

## Protocol (as reported in the paper)
* Raw physical units: **no scaling, standardisation, normalisation or transformation**. Product grade L/M/H → 0/1/2; load–thermal interaction = torque × (process − air temperature). Failure-mode flags (TWF, HDF, PWF, OSF, RNF) are dropped to avoid label leakage.
* Five learners, all with **library default hyperparameters**: Random forest, Gradient boosting (scikit-learn), XGBoost, SVM (RBF), MLP.
* Inverse-frequency class weights w+ = N/(2N+), w− = N/(2N−).
* Stratified 5-fold outer CV (seed 42). In each outer training fold: inner stratified 80/20 split, model on 80 %, Platt calibration on 20 %.
* Class-wise F1 (normal and failure), macro-F1, precision, recall, PR-AUC, accuracy; hard label = calibrated p ≥ 0.5.
* External feature validation: out-of-fold permutation importance (PR-AUC drop, 10 repeats) → ordered sets; Kendall's W and Spearman across folds and learners; ordered-prefix subsets; quintile dose–response with Spearman and tie-corrected Jonckheere–Terpstra test.
* Cost–risk evaluation of five maintenance policies on out-of-fold calibrated probabilities (C_p = 1, C_f = 20, ρ = 0.9), 2,000-sample paired bootstrap, loss-to-cost ratio and probability-noise sensitivity.

## Reproduce
```bash
pip install -r requirements.txt
python code/run_analysis.py   # downloads the data from UCI if needed; ~2 min on a desktop CPU; writes results/
python code/make_figures.py   # writes figures/Fig2–Fig4
```

## Outputs (`results/`)
| File | Content |
|---|---|
| `table4_model_comparison.csv` | Table 4 (mean and SD over outer folds) |
| `permutation_importance_selected_by_fold.csv`, `permutation_importance_by_model.csv` | ordered-set inputs |
| `ordered_prefix_subsets.csv` | nested ordered-prefix evaluation |
| `dose_response_quintiles.csv` | quintile incidence and mean calibrated probability |
| `table5_economic_comparison.csv`, `frontier.csv`, `sensitivity.csv` | Table 5, Fig. 2, Fig. 4 |
| `results.json` | all statistics quoted in the manuscript |
| `oof_calibrated_probabilities_all_models.csv` | out-of-fold calibrated probabilities |

Environment used: Python 3.12.10, scikit-learn 1.9.1, xgboost 3.4.1 (see `requirements.txt`).
