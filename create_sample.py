import pandas as pd
import numpy as np

train_df = pd.read_csv('./data/train_values.csv', index_col=0, parse_dates=['timestamp'])
labels_df = pd.read_csv('./data/train_labels.csv', index_col='process_id')

train_df = train_df.merge(labels_df, left_on='process_id', right_index=True, how='left')

all_machines = list(train_df['object_id'].unique())
rand_machines = np.random.choice(all_machines, replace=False, size=int(0.2*len(all_machines)))

train_df[train_df['object_id'].isin(rand_machines)].to_csv('./data/sampled_machines_train.csv')
