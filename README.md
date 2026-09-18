# IN6227 Assignment 1

Author: Shen Chengang (G2606169D). Variant 1.

This project compares logistic regression and random forest on the supplied classification dataset. Hyperparameters are selected by training-only stratified three-fold cross-validation using average precision. The provided test set is used only for final performance evaluation, with a fixed probability threshold of 0.50.

## Run

Use Python 3.12. Put the supplied `dataset.zip` beside the script; keep its original `dataset/train.csv` and `dataset/test.csv` paths inside the archive.

```sh
python -m pip install -r requirements.txt
python train_models.py --data dataset.zip --out results
```

The run takes roughly two minutes on the environment used for this report; hardware can change runtime. Random seeds and exact package versions are recorded for reproducibility.

## Files

- `train_models.py`: data auditing, fold-local preprocessing, model selection and evaluation.
- `requirements.txt`: versions used in the experiment.
- `results/results.json`: data audit, selected parameters, test metrics and package versions.
- `results/*_cv.csv`: all cross-validation candidate scores.
- `results/*_predictions.csv`: predictions for labelled test records, in original row order after excluding the one missing-label record.
- `IN6227-Assignment-1.pdf`: two-page report.

The dataset is not included in this source package. Obtain it from the course materials.

## Experiment decisions

Missing labels are excluded. Numeric features use median imputation; categorical features use most-frequent imputation and one-hot encoding. Only logistic regression standardizes numeric features. All fitted preprocessing stays inside the cross-validation pipeline. No rows are removed for outliers, no class weighting or oversampling is used, and all 15 predictors are retained.

Logistic regression searches C = 0.1, 1, 10. Random forest searches max_depth = 12 or None and min_samples_leaf = 2 or 10, with 200 trees. Both use seed 42 where relevant. Average precision is the selection metric; accuracy, balanced accuracy, precision, recall, F1, ROC-AUC and average precision are reported on the held-out test set.

## Observed results

| Model | Test AP | Test F1 | Test recall |
|---|---:|---:|---:|
| Logistic regression | 0.7025 | 0.6286 | 0.5713 |
| Random forest | 0.7067 | 0.6080 | 0.5256 |

The forest has slightly higher AP; logistic regression has higher F1 and recall at the fixed threshold. These results do not establish statistical significance or performance on an independent future population.
