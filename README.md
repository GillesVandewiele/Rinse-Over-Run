# Rinse Over Run

Solution to get the second rank on the leaderboard of the Rinse over Run competition

## Install

## Building the features

`python src/features/build_features.py data/raw/train_values.csv data/raw/test_values.csv data/raw/train_labels.csv data/raw/recipe_metadata.csv data/features/`

**NOTE: This requires quite a lot of RAM (>8 GB) and takes a while (~5 hours). It should be noted that this can possibly be reduced by fiddling with the parameters of the `extract_features` function from tsfresh, or by partitioning the input file into multiple chunks and handling each chunk independently.**

## Generating out-of-sample predictions for stacking

## Evaluate models with cross-validation

`python src/models/gradient_boosting.py --cross_validation data/features/ output/`

### Results without stacking

```
+MAPE per model----------------+--------+---------------------+
| Recipe | Process Combination | Weight | MAPE                |
+--------+---------------------+--------+---------------------+
| 3      | 1                   | 0.0219 | 0.38150749103694254 |
| 3      | 2                   | 0.0064 | 0.30265204730067097 |
| 3      | 3                   | 0.1695 | 0.3036406446617803  |
| 9      | 8                   | 0.0411 | 0.261495246538695   |
| 15     | 1                   | 0.0765 | 0.3113903339369102  |
| 15     | 2                   | 0.0013 | 0.28706253186845304 |
| 15     | 3                   | 0.2289 | 0.28177810261072755 |
| 15     | 6                   | 0.0007 | 0.27781473057860173 |
| 15     | 7                   | 0.2258 | 0.2799518406911836  |
| 15     | 14                  | 0.0017 | 0.2539413686237908  |
| 15     | 15                  | 0.2262 | 0.2523300341218434  |
+--------+---------------------+--------+---------------------+
TOTAL MAPE = 0.2821164305690393
```

### Results with stacking

## Create submission

`python src/models/gradient_boosting.py --submission data/features/ output/`

## Project Organization

```
    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── features       <- The extracted features from the timeseries.
    │   ├── predictions    <- Predictions generated through stacking.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── output             <- Generated graphics/plots and submission files.
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and make predictions
    │   │   ├── stacking.py
    └── └── └── gradient_boosting.py
```
