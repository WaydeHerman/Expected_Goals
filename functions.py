
import pandas as pd
import numpy as np
import math
import ast
from matplotlib import pyplot
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.preprocessing import minmax_scale, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib


def preprocess_data(df):

    # 'one-hot' encoding for league feature:
    df['Premier League'] = df['league'].apply(
        lambda x: 1 if x == 'Premier League' else 0)
    df['La Liga'] = df['league'].apply(lambda x: 1 if x == 'La Liga' else 0)
    df['Ligue 1'] = df['league'].apply(lambda x: 1 if x == 'Ligue 1' else 0)
    df['Bundesliga'] = df['league'].apply(
        lambda x: 1 if x == 'Bundesliga' else 0)
    df['Serie A'] = df['league'].apply(lambda x: 1 if x == 'Serie A' else 0)

    # 'one-hot' encoding for state feature:
    df['state_-1'] = df['state'].apply(lambda x: 1 if x <= -1 else 0)
    df['state_0'] = df['state'].apply(lambda x: 1 if x == 0 else 0)
    df['state_1'] = df['state'].apply(lambda x: 1 if x >= 1 else 0)
    df.drop(['state'], axis=1, inplace=True)

    # Correct x-axis:
    df['x'] = 100 - df['x']

    # distance to the centre of the goal posts:
    df['distance'] = np.sqrt(df['x']**2 + (50 - df['y'])**2)

    # angle of goals seen:
    for index, row in df[['x', 'y', 'distance']].iterrows():
        if row['y'] <= 44.3:
            if row['x'] != 0:
                b = 44.3 - row['y']
                c = 55.7 - row['y']
                angle = math.degrees(
                    math.atan(c/row['x']) - math.atan(b/row['x']))
            else:
                angle = 0
        elif row['y'] >= 55.7:
            if row['x'] != 0:
                b = row['y'] - 55.7
                c = row['y'] - 44.3
                angle = math.degrees(
                    math.atan(c/row['x']) - math.atan(b/row['x']))
            else:
                angle = 0
        else:
            if row['x'] != 0:
                d = row['y'] - 44.3
                e = 55.7 - row['y']
                angle = math.degrees(
                    math.atan(d/row['x']) + math.atan(e/row['x']))
            else:
                angle = 180
        df.loc[index, 'angle'] = angle

    return df


def get_tuned_model(model_type, params, seed=17):
    if model_type == 'LogisticRegression':
        C = params['logisticregression__C']
        solver = params['logisticregression__solver']
        estimator = LogisticRegression(random_state=seed, C=C, solver=solver)
    if model_type == 'RandomForestClassifier':
        n_estimators = params['randomforestclassifier__n_estimators']
        max_features = params['randomforestclassifier__max_features']
        max_depth = params['randomforestclassifier__max_depth']
        min_samples_split = params['randomforestclassifier__min_samples_split']
        min_samples_leaf = params['randomforestclassifier__min_samples_leaf']
        bootstrap = params['randomforestclassifier__bootstrap']
        estimator = RandomForestClassifier(n_estimators=n_estimators, max_features=max_features, max_depth=max_depth,
                                           min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                           bootstrap=bootstrap, random_state=seed)
    return estimator


def export_calibration_plot(probability, y_test, id_num, dataset, model_type, calibrated):
    fop, mpv = calibration_curve(
        y_test, probability, n_bins=10)

    pyplot.figure(id_num, figsize=(5, 10))

    pyplot.subplot(211)
    pyplot.title('Calibration Curve')
    pyplot.ylabel('Accuracy')
    pyplot.xlabel('Confidence')
    pyplot.plot([0, 1], [0, 1], linestyle='--')
    pyplot.plot(mpv, fop, marker='.')

    pyplot.subplot(212)
    pyplot.hist(minmax_scale(probability), bins=100)
    pyplot.title('Probability Distribution')
    pyplot.ylabel('Count')
    pyplot.xlabel('Probability')

    pyplot.subplots_adjust(hspace=0.4)
    pyplot.savefig('Results/{}_{}_{}_{}.png'.format(id_num,
                                                    dataset, model_type, calibrated), bbox_inches='tight')
    pyplot.close('all')


def train_model(dataset, id_num, results, input_path='data/processed/', output_path='Models/'):

    config = results.loc[id_num]
    model_name = config['model_type']
    scale = config['scale']
    calibration = config['calibration']

    df = pd.read_csv(input_path + '{}.csv'.format(dataset))
    X_train = df.drop(['goalYN'], axis=1)
    y_train = df['goalYN']

    if scale == True:
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        joblib.dump(scaler, (output_path + '{}_scaler.joblib'.format(dataset)))

    params = ast.literal_eval(config['params'])

    if dataset != config['dataset']:
        print('Incorrect model choice, Model Dataset {} used with dataset {}'.format(
            config['dataset'], dataset))

    model = get_tuned_model(model_name, params)
    model.fit(X_train, y_train)

    if calibration == True:
        model = CalibratedClassifierCV(model, method='sigmoid', cv=10)
        model.fit(X_train, y_train)

    joblib.dump(model, (output_path + '{}_model.joblib'.format(dataset)))


def make_prediction(index, row, features, scaler, model):
    X = row[features].drop(['goalYN'])
    if scaler != None:
        X = pd.DataFrame(scaler.transform(
            X.values.reshape((1, -1))))
    # predict with model
    xg = model.predict_proba(X.values.reshape((1, -1)))[0][1]
    return xg

import matplotlib.pyplot as plt

def get_game_xg(game_df, pitch="#195905", line="#faf0e6", orientation="h",view="full"):
    
    
    orientation = orientation
    view = view
    line = line
    pitch = pitch
    
    if orientation.lower().startswith("h"):
        
        if view.lower().startswith("h"):
            fig,ax = plt.subplots(figsize=(6.8,10.4))
            plt.xlim(49,105)
            plt.ylim(-1,69)
        else:
            fig,ax = plt.subplots(figsize=(10.4,6.8))
            plt.xlim(-1,105)
            plt.ylim(-1,69)
        ax.axis('off') # this hides the x and y ticks
    
        # side and goal lines #
        ly1 = [0,0,68,68,0]
        lx1 = [0,104,104,0,0]

        plt.plot(lx1,ly1,color=line,zorder=5)


        # boxes, 6 yard box and goals

            #outer boxes#
        ly2 = [13.84,13.84,54.16,54.16] 
        lx2 = [104,87.5,87.5,104]
        plt.plot(lx2,ly2,color=line,zorder=5)

        ly3 = [13.84,13.84,54.16,54.16] 
        lx3 = [0,16.5,16.5,0]
        plt.plot(lx3,ly3,color=line,zorder=5)

            #goals#
        ly4 = [30.34,30.34,37.66,37.66]
        lx4 = [104,104.2,104.2,104]
        plt.plot(lx4,ly4,color=line,zorder=5)

        ly5 = [30.34,30.34,37.66,37.66]
        lx5 = [0,-0.2,-0.2,0]
        plt.plot(lx5,ly5,color=line,zorder=5)


           #6 yard boxes#
        ly6 = [24.84,24.84,43.16,43.16]
        lx6 = [104,99.5,99.5,104]
        plt.plot(lx6,ly6,color=line,zorder=5)

        ly7 = [24.84,24.84,43.16,43.16]
        lx7 = [0,4.5,4.5,0]
        plt.plot(lx7,ly7,color=line,zorder=5)

        #Halfway line, penalty spots, and kickoff spot
        ly8 = [0,68] 
        lx8 = [52,52]
        plt.plot(lx8,ly8,color=line,zorder=5)


        plt.scatter(93,34,color=line,zorder=5)
        plt.scatter(11,34,color=line,zorder=5)
        plt.scatter(52,34,color=line,zorder=5)

        circle1 = plt.Circle((93.5,34), 9.15,ls='solid',lw=1.5,color=line, fill=False, zorder=1,alpha=1)
        circle2 = plt.Circle((10.5,34), 9.15,ls='solid',lw=1.5,color=line, fill=False, zorder=1,alpha=1)
        circle3 = plt.Circle((52, 34), 9.15,ls='solid',lw=1.5,color=line, fill=False, zorder=2,alpha=1)

        ## Rectangles in boxes
        rec1 = plt.Rectangle((87.5,20), 16,30,ls='-',color=pitch, zorder=1,alpha=1)
        rec2 = plt.Rectangle((0, 20), 16.5,30,ls='-',color=pitch, zorder=1,alpha=1)

        ## Pitch rectangle
        rec3 = plt.Rectangle((-1, -1), 106,70,ls='-',color=pitch, zorder=1,alpha=1)

        ax.add_artist(rec3)
        ax.add_artist(circle1)
        ax.add_artist(circle2)
        ax.add_artist(rec1)
        ax.add_artist(rec2)
        ax.add_artist(circle3)
        
    else:
        if view.lower().startswith("h"):
            fig,ax = plt.subplots(figsize=(10.4,6.8))
            plt.ylim(49,105)
            plt.xlim(-1,69)
        else:
            fig,ax = plt.subplots(figsize=(6.8,10.4))
            plt.ylim(-1,105)
            plt.xlim(-1,69)
        ax.axis('off') # this hides the x and y ticks

        # side and goal lines #
        lx1 = [0,0,68,68,0]
        ly1 = [0,104,104,0,0]

        plt.plot(lx1,ly1,color=line,zorder=5)


        # boxes, 6 yard box and goals

            #outer boxes#
        lx2 = [13.84,13.84,54.16,54.16] 
        ly2 = [104,87.5,87.5,104]
        plt.plot(lx2,ly2,color=line,zorder=5)

        lx3 = [13.84,13.84,54.16,54.16] 
        ly3 = [0,16.5,16.5,0]
        plt.plot(lx3,ly3,color=line,zorder=5)

            #goals#
        lx4 = [30.34,30.34,37.66,37.66]
        ly4 = [104,104.2,104.2,104]
        plt.plot(lx4,ly4,color=line,zorder=5)

        lx5 = [30.34,30.34,37.66,37.66]
        ly5 = [0,-0.2,-0.2,0]
        plt.plot(lx5,ly5,color=line,zorder=5)


           #6 yard boxes#
        lx6 = [24.84,24.84,43.16,43.16]
        ly6 = [104,99.5,99.5,104]
        plt.plot(lx6,ly6,color=line,zorder=5)

        lx7 = [24.84,24.84,43.16,43.16]
        ly7 = [0,4.5,4.5,0]
        plt.plot(lx7,ly7,color=line,zorder=5)

        #Halfway line, penalty spots, and kickoff spot
        lx8 = [0,68] 
        ly8 = [52,52]
        plt.plot(lx8,ly8,color=line,zorder=5)


        plt.scatter(34,93,color=line,zorder=5)
        plt.scatter(34,11,color=line,zorder=5)
        plt.scatter(34,52,color=line,zorder=5)

        circle1 = plt.Circle((34,93.5), 9.15,ls='solid',lw=1.5,color=line, fill=False, zorder=1,alpha=1)
        circle2 = plt.Circle((34,10.5), 9.15,ls='solid',lw=1.5,color=line, fill=False, zorder=1,alpha=1)
        circle3 = plt.Circle((34,52), 9.15,ls='solid',lw=1.5,color=line, fill=False, zorder=2,alpha=1)


        ## Rectangles in boxes
        rec1 = plt.Rectangle((20, 87.5), 30,16.5,ls='-',color=pitch, zorder=1,alpha=1)
        rec2 = plt.Rectangle((20, 0), 30,16.5,ls='-',color=pitch, zorder=1,alpha=1)

        ## Pitch rectangle
        rec3 = plt.Rectangle((-1, -1), 70,106,ls='-',color=pitch, zorder=1,alpha=1)

        ax.add_artist(rec3)
        ax.add_artist(circle1)
        ax.add_artist(circle2)
        ax.add_artist(rec1)
        ax.add_artist(rec2)
        ax.add_artist(circle3)
        
    game_df.loc[game_df['homeTeam'] == game_df['team'], 'xM'] = game_df.x / 100 * 104
    game_df.loc[game_df['awayTeam'] == game_df['team'], 'xM'] = (100 - game_df.x) / 100 * 104
    game_df.loc[game_df['homeTeam'] == game_df['team'],'yM'] = game_df.y / 100 * 68
    game_df.loc[game_df['awayTeam'] == game_df['team'],'yM'] = (100 - game_df.y) / 100 * 68

    home_team = game_df['homeTeam'][0]
    away_team = game_df['awayTeam'][0]
    season = game_df['season'][0].replace('/','-')
    home_score = game_df[game_df['homeTeam'] == game_df['team']]['goalYN'].sum()
    away_score = game_df[game_df['awayTeam'] == game_df['team']]['goalYN'].sum()
    home_xg = round(game_df[game_df['homeTeam'] == game_df['team']]['xg'].sum(), 2)
    away_xg = round(game_df[game_df['awayTeam'] == game_df['team']]['xg'].sum(), 2)
        
    x1 = game_df[game_df['homeTeam'] == game_df['team']]['xM']
    y1 = game_df[game_df['homeTeam'] == game_df['team']]['yM']
    s1 = game_df[game_df['homeTeam'] == game_df['team']]['xg'] * 250
    x2 = game_df[game_df['awayTeam'] == game_df['team']]['xM']
    y2 = game_df[game_df['awayTeam'] == game_df['team']]['yM']
    s2 = game_df[game_df['awayTeam'] == game_df['team']]['xg'] * 250

    plt.scatter(x1,y1,s1,marker='o',color='red',edgecolors="black", zorder=12)
    plt.scatter(x2,y2,s2,marker='o',color='blue',edgecolors="black", zorder=12)
    plt.text(4,62,'{} - {}'.format(home_team, away_team), color='white', fontsize=15, fontweight='bold')
    plt.text(4,58,'xg: {} - {}'.format(home_xg, away_xg), color='white', fontsize=15, fontweight='bold')
    plt.savefig('Games/{}_{}_{}.png'.format(home_team, away_team, season))
    plt.close('all')
