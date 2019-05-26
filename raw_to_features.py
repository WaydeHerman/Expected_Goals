import math
import numpy as np
import pandas as pd

INPUT_PATH = './Data/raw/'
OUTPUT_PATH = './Data/features/'

list_of_seasons = ['1213', '1314']
list_of_leagues = ['EPL', 'LL']

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

# remove columns which won't be used in model:
data_df = raw_df.drop(['season', 'homeTeam', 'awayTeam', 'date', 'team', 'min', 'sec', 'chanceX1', 'chanceY1', 'chanceX2',
                       'chanceY2', 'onTarget', 'sixYard', 'penaltyArea', 'outBox', 'penaltyYN', 'ownGoalYN'], axis=1)

# one-hot encoding for league feature:
dummy_league = pd.get_dummies(data_df['league'])
data_df.drop(['league'], axis=1, inplace=True)
data_df = pd.concat([data_df, dummy_league], axis=1)

# clip shot 'state' (whether a team is ahead, equal or behind to the other team)
data_df['state'] = data_df['state'].apply(
    lambda x: 1 if x >= 1 else (-1 if x <= -1 else 0))

# Correct x-axis:
data_df['x'] = 100 - data_df['x']

# distance to the centre of the goal posts:
data_df['distance'] = np.sqrt(data_df['x']**2 + (50 - data_df['y'])**2)

# angle of goals seen:

for index, row in data_df[['x', 'y', 'distance']].iterrows():
    if row['y'] <= 44.3:
        if row['x'] != 0:
            b = 44.3 - row['y']
            c = 55.7 - row['y']
            angle = math.degrees(math.atan(c/row['x']) - math.atan(b/row['x']))
        else:
            angle = 0
    elif row['y'] >= 55.7:
        if row['x'] != 0:
            b = row['y'] - 55.7
            c = row['y'] - 44.3
            angle = math.degrees(math.atan(c/row['x']) - math.atan(b/row['x']))
        else:
            angle = 0
    else:
        if row['x'] != 0:
            d = row['y'] - 44.3
            e = 55.7 - row['y']
            angle = math.degrees(math.atan(d/row['x']) + math.atan(e/row['x']))
        else:
            angle = 180
    data_df.loc[index, 'angle'] = angle

# split dataframe into subtypes:
direct_df = data_df[data_df['directFKYN'] == 1]
head_cross_df = data_df[(data_df['headerYN'] == 1) & (data_df['crossYN'] == 1)]
cross_df = data_df[(data_df['headerYN'] == 0) & (data_df['crossYN'] == 1)]
head_df = data_df[(data_df['headerYN'] == 1) & (data_df['crossYN'] == 0)]
regular_df = data_df[(data_df['headerYN'] == 0) & (
    data_df['crossYN'] == 0) & (data_df['directFKYN'] == 0)]

# save as csv's:
direct_df.to_csv(OUTPUT_PATH + 'direct_df.csv', index=False)
head_cross_df.to_csv(OUTPUT_PATH + 'head_cross_df.csv', index=False)
cross_df.to_csv(OUTPUT_PATH + 'cross_df.csv', index=False)
head_df.to_csv(OUTPUT_PATH + 'head_df.csv', index=False)
regular_df.to_csv(OUTPUT_PATH + 'regular_df.csv', index=False)
