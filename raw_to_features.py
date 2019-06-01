"""
This file handles converting 'raw' tabular data to processed data for modeling.

usage:
    >>> define list of seasons ['1213', '1314', '1415', '1516', '1617', '1718', '1819']
    >>> define list of leagues ['EPL', 'LL', 'FL', 'ISA', 'GBL']
"""

import math
import numpy as np
import pandas as pd
import yaml
from functions import preprocess_data

INPUT_PATH = './Data/raw/'
OUTPUT_PATH = './Data/processed/'

list_of_seasons = ['1213', '1314', '1415', '1516', '1617', '1718']
list_of_leagues = ['EPL', 'LL', 'FL', 'ISA', 'GBL']

# Import features:
with open("./config/features_config.yml", 'r') as stream:
    feature_dict = yaml.load(stream)

direct_features = feature_dict['direct']
head_cross_features = feature_dict['head_cross']
cross_features = feature_dict['cross']
head_features = feature_dict['head']
regular_features = feature_dict['regular']

# Concat different leagues:
data_list = []
for season in list_of_seasons:
    for league in list_of_leagues:
        tmp_df = pd.read_csv((INPUT_PATH + "shots_{}{}.csv".format(league, season)), header=None, names=['league', 'season',
                                                                                                         'homeTeam', 'awayTeam', 'date', 'team', 'min', 'sec', 'x', 'y', 'goalYN',
                                                                                                         'state', 'headerYN', 'bigChanceYN', 'fromCornerYN', 'fastBreakYN',
                                                                                                         'penaltyYN', 'directFKYN', 'ownGoalYN', 'chanceX1', 'chanceY1',
                                                                                                         'chanceX2', 'chanceY2', 'crossYN', 'throughballYN', 'indirectFKYN',
                                                                                                         'secondThroughballYN', 'dribbleKeeperYN', 'dribbleBeforeYN', 'reboundYN',
                                                                                                         'errorYN', 'onTarget', 'sixYard', 'penaltyArea', 'outBox'])
        data_list.append(tmp_df)
raw_df = pd.concat(data_list, axis=0)

# Preprocess data:
data_df = preprocess_data(raw_df)

data_df = data_df[(data_df['penaltyYN'] == 0) & (data_df['ownGoalYN'] == 0)]

# split dataframe into subtypes:
direct_df = data_df[(data_df['directFKYN'] == 1) & (
    (data_df['indirectFKYN'] == 0) & (data_df['crossYN'] == 0))]
head_cross_df = data_df[(data_df['headerYN'] == 1) & (data_df['crossYN'] == 1)]
cross_df = data_df[(data_df['headerYN'] == 0)
                   & (data_df['crossYN'] == 1)]
head_df = data_df[(data_df['headerYN'] == 1) & (data_df['crossYN'] == 0)]
regular_df = data_df[(data_df['headerYN'] == 0) & (
    data_df['crossYN'] == 0) & (data_df['directFKYN'] == 0)]

# Subselect featureas to be used:
direct_df = direct_df[direct_features]
head_cross_df = head_cross_df[head_cross_features]
cross_df = cross_df[cross_features]
head_df = head_df[head_features]
regular_df = regular_df[regular_features]

# Save as csv's:
direct_df.to_csv(OUTPUT_PATH + 'direct.csv', index=False)
head_cross_df.to_csv(OUTPUT_PATH + 'head_cross.csv', index=False)
cross_df.to_csv(OUTPUT_PATH + 'cross.csv', index=False)
head_df.to_csv(OUTPUT_PATH + 'head.csv', index=False)
regular_df.to_csv(OUTPUT_PATH + 'regular.csv', index=False)
