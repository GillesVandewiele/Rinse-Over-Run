# The essentials
import pandas as pd
import numpy as np

# CLI & Logging
import click
import logging

# Gradient Boosting
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

# Plotting
import matplotlib.pyplot as plt

# ML utils
from sklearn.model_selection import KFold

# Some fancy printing
from terminaltables import AsciiTable

# Retrieving files from HD using regexes
import glob

# Python standard library
from collections import defaultdict
import datetime

# Model explanation with Shapley values
import shap

combinations_per_recipe = {
    #3: [1, 2, 3], 
    #9: [8],
    #15: [1, 2, 3, 6, 7, 14, 15]
    3: [3],
    15: [15]
}

weights = {
    (3, 1): 0.0219,
    (3, 2): 0.0064,
    (3, 3): 0.1695,
    (9, 8): 0.0411,
    (15, 1): 0.0765,
    (15, 2): 0.0013,
    (15, 3): 0.2289,
    (15, 6): 0.0007,
    (15, 7): 0.2258,
    (15, 14): 0.0017,
    (15, 15): 0.2262,
}

def custom_mape(approxes, targets):
    """Competition metric"""
    nominator = np.abs(np.subtract(approxes, targets))
    denominator = np.maximum(np.abs(targets), 290000)
    return np.mean(nominator / denominator)


def mape_xgb(yhat, y):
    # y = y.values
    y = y.get_label()
    y = np.exp(y)
    yhat = np.exp(yhat)
    error = np.mean(np.abs(yhat - y)/np.maximum(290000, y))
    return "mape", error


def mape_lgbm(y, yhat):
    y = np.exp(y)
    yhat = np.exp(yhat)
    error = np.mean(np.abs(yhat - y)/np.maximum(290000, y))
    return "mape", error, False


def mapeobj(preds,dtrain):
    """objective function for xgb"""
    gaps = dtrain
    grad = np.sign(preds - gaps)/np.maximum(np.log(290000), gaps)
    hess = 1 / gaps
    grad[(gaps == 0)] = 0
    hess[(gaps == 0)] = 0
    return -grad,hess


class MAPEMetric(object):
    """eval_metric for CatBoost (can only be used when task_type=CPU)"""
    def get_final_error(self, error, weight):
        return error

    def is_max_optimal(self):
        return False

    def evaluate(self, approxes, targets, weight):
        return custom_mape(np.exp(approxes), np.exp(targets)), len(targets)


def get_validation_data(X_train, y_train):
    """Just take 5% of the data at random to serve as validation set"""
    train_idx = np.random.choice(X_train.index, replace=False, 
                                 size=int(0.95 * len(X_train)))
    val_idx = list(set(X_train.index) - set(train_idx))

    X_val = X_train.loc[val_idx, :]
    y_val = y_train.loc[val_idx]
    X_train = X_train.loc[train_idx, :]
    y_train = y_train.loc[train_idx]

    return X_train, y_train, X_val, y_val

# TODO: To be honest, this stuff beneath should probably be wrapped into 
# TDOO: classes, with a general interface (fit, generate_shapley, 
# TODO: generate_prediction, ...) and an implementation for the 3 models.
def generate_shapley(model, X_train, X_val, X_test, output_path=None):
    """Generate a Shapley plot, based on all data."""
    explainer = shap.TreeExplainer(model)
    all_data = pd.concat([X_train, X_val, X_test])
    shap_values = explainer.shap_values(all_data.values)

    plt.figure()
    shap.summary_plot(shap_values, all_data, max_display=30, 
                      auto_size_plot=True, show=False, color_bar=False)
    
    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()


def fit_lgbm(X_train, y_train, X_test, processes=None, output_path=None):
    """Fit a LGBM Gradient Booster, plot shapley value and return predictions
    on test set"""
    X_train, y_train, X_val, y_val = get_validation_data(X_train, y_train)

    lgbm = LGBMRegressor(n_estimators=100000, objective='mape')
    lgbm.fit(X_train.values, y_train.values, 
             eval_set=(X_val.values, y_val.values), 
             early_stopping_rounds=100, verbose=0, eval_metric=mape_lgbm)

    best_nr_trees = lgbm.best_iteration_

    predictions = lgbm.predict(X_test, num_iteration=best_nr_trees)

    generate_shapley(lgbm, X_train, X_val, X_test, output_path='{}_shap.png'.format(output_path))

    return predictions

def fit_cat(X_train, y_train, X_test, processes=None, output_path=None):
    """Fit a CatBoost Gradient Booster, plot shapley value and return predictions
    on test set"""
    X_train, y_train, X_val, y_val = get_validation_data(X_train, y_train)

    cat = CatBoostRegressor(iterations=100000, od_type='Iter', od_wait=100, 
                            learning_rate=0.33,
                            loss_function='MAPE', eval_metric=MAPEMetric(), 
                            task_type='CPU')
    cat.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=0)

    predictions = cat.predict(X_test)

    generate_shapley(cat, X_train, X_val, X_test, output_path='{}_shap.png'.format(output_path))

    return predictions

def fit_xgb(X_train, y_train, X_test, processes=None, output_path=None):
    """Fit a XGBoost Gradient Booster, plot shapley value and return predictions
    on test set"""
    X_train, y_train, X_val, y_val = get_validation_data(X_train, y_train)

    xgb = XGBRegressor(n_estimators=100000, objective=mapeobj)
    xgb.fit(X_train.values, y_train.values, 
            eval_set=[(X_val.values, y_val.values)], 
            early_stopping_rounds=100, verbose=0, eval_metric=mape_xgb)

    best_n_trees = xgb.best_ntree_limit

    predictions = xgb.predict(X_test.values, ntree_limit=best_n_trees)

    generate_shapley(xgb, X_train, X_val, X_test, output_path='{}_shap.png'.format(output_path))
  
    return predictions

def fit_models(models, feature_path, output_path, stack_path, cv=True):
    # Models are tuples of following form (name, fit_method)
    all_predictions = defaultdict(dict)
    for recipe in combinations_per_recipe:
        for process_combination in combinations_per_recipe[recipe]:
            train_features = pd.read_csv('{}/train_features_{}_{}.csv'.format(feature_path, recipe, process_combination), index_col=0)
            X_train = train_features.drop('target', axis=1)
            y_train = np.log(train_features['target'])

            if stack_path:
                for file in glob.glob('{}/train_predictions_stack_*_{}_{}.csv'.format(stack_path, recipe, process_combination)):
                    train_predictions = pd.read_csv(file, index_col=0)
                    X_train = X_train.merge(train_predictions, left_index=True, right_index=True)

            if cv:
                for name, fit_method in models:
                    cv_predictions = []
                    kf = KFold(n_splits=5, random_state=2019, shuffle=True)
                    for fold_nr, (train_idx, test_idx) in enumerate(kf.split(X_train, y_train)):
                        X_cv_train = X_train.iloc[train_idx, :]
                        X_cv_test = X_train.iloc[test_idx, :]

                        y_cv_train = y_train.iloc[train_idx]
                        y_cv_test = y_train.iloc[test_idx]
                        logging.info('Fitting {} with data from recipe {} and present phases combination {}...'.format(name, recipe, process_combination))
                        predictions = np.exp(fit_method(X_cv_train, y_cv_train, X_cv_test, output_path='{}/{}_{}_{}'.format(output_path, name, recipe, process_combination)))
                        mape = custom_mape(predictions, np.exp(y_cv_test))
                        logging.info('[Recipe: {} Present Phases: {} Fold: {}] {} TEST MAPE = {}'.format(recipe, process_combination, fold_nr + 1, name, mape))
                        cv_predictions.append(pd.DataFrame(np.reshape(predictions, (-1, 1)), index=X_cv_test.index, columns=['prediction']))

                    prediction_df = pd.concat(cv_predictions)
                    all_predictions[(recipe, process_combination)][name] = prediction_df

            else:
                X_test = pd.read_csv('{}/test_features_{}_{}.csv'.format(feature_path, recipe, process_combination), index_col=0)
                for name, fit_method in models:
                    logging.info('Fitting {} with data from recipe {} and present phases combination {}...'.format(name, recipe, process_combination))
                    predictions = np.exp(fit_method(X_train, y_train, X_test, output_path='{}/{}_{}_{}'.format(output_path, name, recipe, process_combination)))
                    all_predictions[(recipe, process_combination)][name] = pd.DataFrame(np.reshape(predictions, (-1, 1)), index=X_test.index, columns=['prediction'])

    return all_predictions



def fit_models_cross_validation(feature_path, output_path, stack_path):
    models = [
        ('lgb', fit_lgbm),
        ('xgb', fit_xgb)
    ]

    predictions = fit_models(models, feature_path, output_path, stack_path, cv=True)

    table_data = [['Recipe', 'Process Combination', 'Weight', 'Model', 'MAPE']]
    total_mape = defaultdict(float)
    for recipe in combinations_per_recipe:
        for process_combination in combinations_per_recipe[recipe]:
            train_features = pd.read_csv('{}/train_features_{}_{}.csv'.format(feature_path, recipe, process_combination), index_col=0)
            y = train_features['target']
            agg_predictions = []
            for i, model in enumerate(models):
                preds = predictions[(recipe, process_combination)][model[0]].loc[y.index]
                agg_predictions.append(preds)
                mape = custom_mape(preds['prediction'], y)
                total_mape[model[0]] += weights[(recipe, process_combination)] * mape
                if i == 0:
                    table_data.append([str(recipe), str(process_combination), str(weights[(recipe, process_combination)]), model[0], str(mape)])
                else:
                    table_data.append(['', '', model[0], str(mape), ''])

            agg_predictions = pd.concat(agg_predictions, axis=1).mean(axis=1)
            mape = custom_mape(agg_predictions, y)
            total_mape['agg'] += weights[(recipe, process_combination)] * mape
            table_data.append(['', '', 'agg', str(mape), ''])

    result_table = AsciiTable(table_data, 'MAPE per model')
    print(result_table.table)
    print(total_mape)


def fit_models_submission(feature_path, output_path, stack_path):
    models = [
        ('lgb', fit_lgbm),
        ('xgb', fit_xgb)
    ]

    submission = []
    predictions = fit_models(models, feature_path, output_path, stack_path, cv=False)
    for recipe in combinations_per_recipe:
        for process_combination in combinations_per_recipe[recipe]:
            agg_predictions = []
            for i, model in enumerate(models):
                preds = predictions[(recipe, process_combination)][model[0]]
                agg_predictions.append(preds)

            agg_predictions = pd.concat(agg_predictions, axis=1).mean(axis=1)
            submission.append(agg_predictions)

    today = datetime.datetime.now()
    submission_df = pd.concat(submission).sort_index().reset_index(drop=False)
    submission_df.columns = ['process_id', 'final_rinse_total_turbidity_liter']
    submission_df.to_csv('{}/submission_{:02d}{:02d}.csv'.format(output_path, today.month, today.day), index=False)


@click.command()
@click.option('--cross_validation/--submission', default=False)
@click.argument('feature_path', type=click.Path(exists=True))
@click.argument('output_path', type=click.Path(exists=True))
@click.argument('stack_path', required=False)
def main(cross_validation, feature_path, output_path, stack_path=None):
    if cross_validation:
        fit_models_cross_validation(feature_path, output_path, stack_path)
    else:
        fit_models_submission(feature_path, output_path, stack_path)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
