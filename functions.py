
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
