import numpy as np
import pandas as pd
import math
import yaml
from joblib import load
from functions import preprocess_data, make_prediction, get_game_xg

league = 'EPL'
season = '1415'

homeTeam = 'Arsenal'
awayTeam = 'Chelsea'


# Import features:
with open("./config/features_config.yml", 'r') as stream:
    feature_dict = yaml.load(stream)

direct_features = feature_dict['direct']
head_cross_features = feature_dict['head_cross']
cross_features = feature_dict['cross']
head_features = feature_dict['head']
regular_features = feature_dict['regular']

try:
    direct_scaler = load('Models/direct_scaler.joblib')
except:
    direct_scaler = None

try:
    head_cross_scaler = load('Models/head_cross_scaler.joblib')
except:
    head_cross_scaler = None

try:
    cross_scaler = load('Models/cross_scaler.joblib')
except:
    cross_scaler = None

try:
    head_scaler = load('Models/head_scaler.joblib')
except:
    head_scaler = None

try:
    regular_scaler = load('Models/regular_scaler.joblib')
except:
    regular_scaler = None

direct_model = load('Models/direct_model.joblib')
head_cross_model = load('Models/head_cross_model.joblib')
cross_model = load('Models/cross_model.joblib')
head_model = load('Models/head_model.joblib')
regular_model = load('Models/regular_model.joblib')

tmp_df = pd.read_csv(('Data/raw/shots_{}{}.csv'.format(league, season)), header=None, names=['league', 'season',
                                                                                             'homeTeam', 'awayTeam', 'date', 'team', 'min', 'sec', 'x', 'y', 'goalYN',
                                                                                             'state', 'headerYN', 'bigChanceYN', 'fromCornerYN', 'fastBreakYN',
                                                                                             'penaltyYN', 'directFKYN', 'ownGoalYN', 'chanceX1', 'chanceY1',
                                                                                             'chanceX2', 'chanceY2', 'crossYN', 'throughballYN', 'indirectFKYN',
                                                                                             'secondThroughballYN', 'dribbleKeeperYN', 'dribbleBeforeYN', 'reboundYN',
                                                                                             'errorYN', 'onTarget', 'sixYard', 'penaltyArea', 'outBox'])

raw_df = preprocess_data(tmp_df)

season = '2014/2015'

df = raw_df[(raw_df['season'] == season) & (
    raw_df['homeTeam'] == homeTeam) & (raw_df['awayTeam'] == awayTeam)]

for index, row in df.iterrows():
    # split into type to predict:
    if row['penaltyYN'] == 1:
        xg = 0.76
        df.loc[index, 'xg'] = xg
    if row['ownGoalYN'] == 1:
        xg = 0
        df.loc[index, 'xg'] = xg
    if row['directFKYN'] == 1 and row['indirectFKYN'] == 0 and row['crossYN'] == 0:
        xg = make_prediction(index, row, direct_features,
                             direct_scaler, direct_model)
        df.loc[index, 'xg'] = xg
    if row['headerYN'] == 1 and row['crossYN'] == 1:
        xg = make_prediction(index, row, head_cross_features,
                             head_cross_scaler, head_cross_model)
        df.loc[index, 'xg'] = xg
    if row['headerYN'] == 0 and row['crossYN'] == 1:
        xg = make_prediction(index, row, cross_features,
                             cross_scaler, cross_model)
        df.loc[index, 'xg'] = xg
    if row['headerYN'] == 1 and row['crossYN'] == 0:
        xg = make_prediction(index, row, head_features,
                             head_scaler, head_model)
        df.loc[index, 'xg'] = xg
    if row['headerYN'] == 0 and row['crossYN'] == 0 and row['directFKYN'] == 0 and row['penaltyYN'] == 0 and row['ownGoalYN'] == 0:
        xg = make_prediction(index, row, regular_features,
                             regular_scaler, regular_model)
        df.loc[index, 'xg'] = xg


get_game_xg(df)
