# Home Credit Default Risk: GCI World 2026 Competition

My solution to the final competition of the **GCI World 2026** data science course: predict whether a loan applicant will default (`TARGET = 1`) from a single application table. The metric is ROC AUC.

| | Score |
|---|---|
| Tutorial baseline | ~0.70 |
| **Final model: 5-fold CV AUC** | **0.7730** |
| **Final model: public leaderboard AUC** | **0.7772** |

The final model blends LightGBM (3 seeds) and CatBoost 50/50. It uses only the competition's `train.csv` and `test.csv`: no external data, no hand labelling.

## How the score improved

Every version was measured on the same 5 folds (`StratifiedKFold(5, shuffle=True, random_state=42)`).

| Version | What changed | CV AUC |
|---|---|---|
| v1 | LightGBM, 52 cleaned and ratio features | 0.7613 |
| [v2](notebooks/v2_lgbm_catboost_blend.ipynb) | + CatBoost, blend with 3 LightGBM seeds | 0.7629 |
| [v3](notebooks/v3_optuna_tuned.ipynb) | + Optuna-tuned LightGBM settings | 0.7632 |
| [v4](notebooks/v4_loan_product_encoding.ipynb) | + loan-product encoding | 0.7694 |
| [v5](notebooks/v5_repeat_customers.ipynb) | + repeat-customer features | 0.7720 |
| [**v6**](notebooks/v6_final_ext_source_imputation.ipynb) | **+ predicted EXT_SOURCE scores (final)** | **0.7730** |

Model tuning stopped paying off after v3. Every later gain came from feature engineering, through the three ideas below.

## Key ideas

### 1. Loan-product encoding (v4, +0.006)
Dividing `AMT_CREDIT` by `AMT_ANNUITY` gives the number of payments. Rounded to 1 or 2 decimals, that ratio fingerprints the lender's loan products (term and interest rate). Some products default much more often than others. The model gets:
- The ratio, **target-encoded** inside each fold, with an inner 5-fold split and smoothing (M = 30) so no row ever sees its own label.
- A loan-structure score.
- The ratio, annuity/goods price and annuity as **CatBoost categorical features**.

### 2. Repeat customers (v5, +0.003)
The same person can appear several times under different IDs. Their dates (`DAYS_BIRTH`, `DAYS_EMPLOYED`, `DAYS_REGISTRATION`, `DAYS_ID_PUBLISH`) shift with each application date, but each date *minus* `DAYS_BIRTH` stays fixed. Rows of the same gender where at least 3 of these 4 fixed values match are linked as one customer. The model then gets each customer's number of other applications and, computed per fold from training rows only, how often those applications defaulted.

### 3. Predicting the missing EXT_SOURCE scores (v6, +0.0013)
`EXT_SOURCE_1/2/3` are the strongest features, but `EXT_SOURCE_1` is missing for about 70% of the rows. A LightGBM regressor learns to predict `EXT_SOURCE_1` and `EXT_SOURCE_3` from the rest of the profile, without using `TARGET`. Rows that have the score get out-of-fold predictions, so the model can't simply copy the true value. The new features are:
- the predicted score and the filled-in score
- the residual (actual minus predicted)
- the mean, product and minimum of the filled scores

The regressor's R² is 0.49 for `EXT_SOURCE_1` and 0.14 for `EXT_SOURCE_3`.

### Leakage rules followed throughout
Anything that uses `TARGET` is computed inside each fold, from that fold's training rows only. Test rows use statistics from the full training set. All seeds are fixed, so results reproduce.

## Ideas that did not help
Tested on the same folds, all within noise (±0.0002) or worse:
- Model families: XGBoost, ExtraTrees, MLP, logistic regression, LightGBM linear trees and GOSS, deeper CatBoost, re-tuning.
- Group-relative features (value ÷ group mean), pairwise EXT products, KNN features, frequency encodings, implied interest rates.
- Retirement and loan-maturity features, job-instability features, rank and logit blending.
- Weaker 2-of-4 repeat-customer links.
- Expected-vs-actual residuals for income, loan size, job length and `EXT_SOURCE_2`.
- Target-encoding `EXT_SOURCE_3`.

## Running it
1. Get the competition data (provided to GCI World 2026 participants; **not included in this repo**).
2. Put `train.csv`, `test.csv` and `sample_submission.csv` in an `input/` folder next to the notebook.
3. The notebooks were written for Google Colab. In another environment, remove the Drive-mount and `%cd` cells, and install the packages in `requirements.txt`.
4. Run all cells. The final notebook takes about 50–70 minutes on a free Colab CPU, and writes `output/submission.csv`.

The v6 notebook keeps the outputs from the run that produced the submitted predictions.
