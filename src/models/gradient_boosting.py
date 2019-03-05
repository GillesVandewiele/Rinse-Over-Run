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


def custom_mape(approxes, targets):
	"""Competition metric"""
	nominator = np.abs(np.subtract(approxes, targets))
	denominator = np.maximum(np.abs(targets), 290000)
    return np.mean(nominator / denominator)


def mape_xgb(yhat, y):
	"""eval function for xgb"""
    y = y.get_label()
    return "mape", custom_mape(np.exp(yhat), np.exp(y))


def mape_lgbm(y, yhat):
	"""eval function for lgbm"""
    y = np.exp(y)
    yhat = np.exp(yhat)
    return "mape", custom_mape(np.exp(yhat), np.exp(y)), False


def mapeobj(preds,dtrain):
	"""objective function for xgb"""
    gaps = dtrain
    # TODO: This is wrong, and should be len(gaps)
    grad = np.sign(pred - gaps)/np.maximum(np.log(290000), gaps)
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
	all_data = pd.concat([X_train, X_val, X_test]).values

	plt.figure()
	shap.summary_plot(shap_values, all_data, max_display=30, 
					  auto_size_plot=True, show=False, color_bar=False)
	
	if output_path:
		pass
	else:
		plt.show()


def fit_lgbm(X_train, y_train, X_test, processes=None):
	"""Fit a LGBM Gradient Booster, plot shapley value and return predictions
	on test set"""
	X_train, y_train, X_val, y_val = get_validation_data(X_train, y_train)

	lgbm = LGBMRegressor(n_estimators=100000, objective='mape')
	lgbm.fit(X_train.values, y_train.values, 
			 eval_set=(X_val.values, y_val.values), 
  			 early_stopping_rounds=100, verbose=50, eval_metric=mape_lgbm)

	best_nr_trees = lgbm.best_iteration_

	predictions = lgbm.predict(X_test, num_iteration=best_nr_trees)

	generate_shapley(lgbm, X_train, X_val, X_test)

	return predictions

def fit_cat(X_train, y_train, X_test, processes=None):
	"""Fit a CatBoost Gradient Booster, plot shapley value and return predictions
	on test set"""
	X_train, y_train, X_val, y_val = get_validation_data(X_train, y_train)

	cat = CatBoostRegressor(iterations=100000, od_type='Iter', od_wait=100, 
	                        learning_rate=0.33,
	                        loss_function='MAPE', eval_metric=MAPEMetric(), 
	                        task_type='CPU')
	cat.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=50)

	predictions = cat.predict(X_test)

	generate_shapley(lgbm, X_train, X_val, X_test)

	return predictions

def fit_xgb(X_train, y_train, X_test, processes=None):
	"""Fit a XGBoost Gradient Booster, plot shapley value and return predictions
	on test set"""
	X_train, y_train, X_val, y_val = get_validation_data(X_train, y_train)

	xgb = XGBRegressor(n_estimators=100000, objective=mapeobj)
	xgb.fit(X_train.values, y_train.values, 
			eval_set=[(X_val.values, y_val.values)], 
			early_stopping_rounds=100, verbose=50, eval_metric=mape_xgb)

	best_n_trees = xgb.best_ntree_limit

	predictions = xgb.predict(X_test.values, ntree_limit=best_n_trees)

	generate_shapley(lgbm, X_train, X_val, X_test)
  
  	return predictions


def fit_models_cross_validation():
	pass

def fit_models_submission():
	pass


@click.command()
@click.argument('mode', type=str)
@click.argument('feature_path', type=click.Path(exists=True))
@click.argument('output_path', type=click.Path(exists=True))
def fit_models():
    combinations_per_recipe = {
        3: [1, 2, 3], 
        9: [8],
        15: [1, 2, 3, 6, 7, 14, 15]
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

	all_mapes = defaultdict(list)
	for recipe in [15, 3]:
	  	for process_combination in combinations_per_recipe[recipe]:
