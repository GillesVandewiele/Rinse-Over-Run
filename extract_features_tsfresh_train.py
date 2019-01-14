import pandas as pd
import numpy as np

import os

import matplotlib.pyplot as plt

from tqdm import tqdm_notebook

from tsfresh import extract_features
from tsfresh.feature_extraction import EfficientFCParameters
from tsfresh.utilities.dataframe_functions import impute

ts_cols = [
    'supply_flow',
    'supply_pressure',
    'return_temperature',
    'return_conductivity',
    'return_turbidity',
    'return_flow',
    'tank_level_pre_rinse',
    'tank_level_caustic',
    'tank_level_acid',
    'tank_level_clean_water',
    'tank_temperature_pre_rinse',
    'tank_temperature_caustic',
    'tank_temperature_acid',
    'tank_concentration_caustic',
    'tank_concentration_acid',
    'supply_pump',
    'supply_pre_rinse',
    'supply_caustic',
    'return_caustic',
    'supply_acid',
    'return_acid',
    'supply_clean_water',
    'return_recovery_water',
    'return_drain',
    'object_low_level'
]

train_df = pd.read_csv('data/train_values.csv', index_col=0, parse_dates=['timestamp'])
train_df['process_phase'] = train_df['process_id'].astype(str) + '-' + train_df['phase']
import warnings
warnings.simplefilter("ignore")
extraction_settings = EfficientFCParameters()
filtered_functions = ['variance_larger_than_standard_deviation', 'has_duplicate_max', 
                      'has_duplicate_min', 'has_duplicate', 'sum_values', 'abs_energy', 
                      'mean_abs_change', 'mean_change', 'mean_second_derivative_central', 
                      'median', 'mean', 'length', 'standard_deviation', 'variance', 
                      'skewness', 'kurtosis', 'absolute_sum_of_changes', 
                      'longest_strike_below_mean', 'longest_strike_above_mean', 
                      'count_above_mean', 'count_below_mean', 'last_location_of_maximum', 
                      'first_location_of_maximum', 'last_location_of_minimum', 
                      'first_location_of_minimum', 
                      'percentage_of_reoccurring_datapoints_to_all_datapoints', 
                      'percentage_of_reoccurring_values_to_all_values', 
                      'sum_of_reoccurring_values', 'sum_of_reoccurring_data_points', 
                      'ratio_value_number_to_time_series_length', 'maximum', 'minimum', 
                      'cid_ce', 'symmetry_looking', 'large_standard_deviation', 'quantile', 
                      'autocorrelation', 'number_peaks', 'binned_entropy', 'index_mass_quantile', 
                      'linear_trend',  'number_crossing_m']
filtered_settings = {}
for function in filtered_functions:
    filtered_settings[function] = extraction_settings[function]
processes = set(train_df['process_id'])
for i, process in enumerate(processes):
    print('process {}/{}'.format(i+1, len(processes)))
    if os.path.isfile('features/train_{}.csv'.format(process)): 
        # We already have this process its features, so skip it
        continue
    subset_df = train_df[train_df['process_id'] == process]
    subset_df[ts_cols] = subset_df[ts_cols].astype(float)
    train_features = extract_features(subset_df[ts_cols + ['process_phase']], 
                                      column_id='process_phase', 
                                      impute_function=impute, 
                                      default_fc_parameters=filtered_settings,
                                      show_warnings=False)
    train_features.to_csv('features/train_{}.csv'.format(process))