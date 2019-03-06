# Rinse Over Run

Solution to get the second rank on the leaderboard of the Rinse over Run competition

## Install

## Building the features

`python src/features/build_features.py data/raw/train_values.csv data/raw/test_values.csv data/raw/train_labels.csv data/raw/recipe_metadata.csv data/features/`

## Generating out-of-sample predictions for stacking

## Evaluate models with cross-validation

`python src/models/gradient_boosting.py --cross_validation data/features/ output/`

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
