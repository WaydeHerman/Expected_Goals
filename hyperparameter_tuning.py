import pandas as pd
import numpy as np
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, minmax_scale
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
from functions import get_tuned_model, export_calibration_plot

SEED = 17
OUTPUT_PATH = './Results/'
n_top_models = 3

# list_of_datasets = ['direct', 'head_cross', 'cross', 'head', 'regular']
# list_of_models = ['LogisticRegression', 'RandomForestClassifier', 'BaggingClassifier',
#                   'GradientBoostingClassifier', 'XGBClassifier', 'SVC']
list_of_datasets = ['direct', 'head_cross', 'cross', 'head', 'regular']
list_of_models = ['LogisticRegression']

results = []
id_num = 0
for dataset in list_of_datasets:
    for model_type in list_of_models:
        with open("config/{}_config.yml".format(model_type), 'r') as stream:
            config_dict = yaml.load(stream)
        model_name = config_dict['model_name']
        scale = config_dict['scale']
        search_method = config_dict['search_method']
        hyperparameter_space = config_dict['hyperparameter_space']

        data_df = pd.read_csv('data/processed/{}.csv'.format(dataset))
        X = data_df.drop(['goalYN'], axis=1)
        y = data_df['goalYN']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=SEED, shuffle=True)

        if model_name == 'LogisticRegression':
            estimator = LogisticRegression(random_state=SEED)
        elif model_name == 'RandomForestClassifier':
            estimator = RandomForestClassifier(random_state=SEED)

        if scale == True:
            pipe = make_pipeline(StandardScaler(), estimator)
        elif scale == False:
            pipe = make_pipeline(estimator)

        if search_method == 'GridSearchCV':
            grid = GridSearchCV(
                pipe, hyperparameter_space, scoring='roc_auc', cv=5, n_jobs=-1, verbose=0, return_train_score=True)
        elif search_method == 'RandomizedSearchCV':
            grid = RandomizedSearchCV(
                pipe, hyperparameter_space, scoring='roc_auc', cv=5, n_jobs=-1, verbose=0, return_train_score=True)

        grid.fit(X_train, y_train)

        for i in range(1, n_top_models+1):
            id_num += 1
            model_id = np.flatnonzero(
                grid.cv_results_['rank_test_score'] == i)[0]

            params = grid.cv_results_['params'][model_id]

            result_dict_uncal = {'id_num': id_num, 'model_type': model_type, 'dataset': dataset, 'calibration': False, 'scale': scale,
                                 'val_mean': grid.cv_results_['mean_test_score'][model_id], 'val_std': grid.cv_results_['std_test_score'][model_id],
                                 'train_mean': grid.cv_results_['mean_train_score'][model_id], 'train_std': grid.cv_results_['std_train_score'][model_id],
                                 'params': params}
            results.append(result_dict_uncal)

            if scale == True:
                scaler = StandardScaler().fit(X_train)
                X_train = scaler.transform(X_train)
                X_test = scaler.transform(X_test)

            tuned_model = get_tuned_model(model_type, params)
            tuned_model.fit(X_train, y_train)
            probability = (np.array(tuned_model.predict_proba(X_test)[:, 1]))
            export_calibration_plot(
                probability, y_test, id_num, dataset, model_type, calibrated=False)
            id_num += 1

            calibrator = CalibratedClassifierCV(
                tuned_model, method='sigmoid', cv=10)
            calibrator.fit(X_train, y_train)

            probability_cal_val = (
                np.array(calibrator.predict_proba(X_test)[:, 1]))
            probability_cal_train = (
                np.array(calibrator.predict_proba(X_train)[:, 1]))

            val_auc_score = roc_auc_score(y_test, probability_cal_val)
            train_auc_score = roc_auc_score(y_train, probability_cal_train)

            result_dict_cal = {'id_num': id_num, 'model_type': model_type, 'dataset': dataset, 'calibration': True, 'scale': scale,
                               'val_mean': val_auc_score, 'val_std': 0,
                               'train_mean': train_auc_score, 'train_std': 0,
                               'params': params}
            results.append(result_dict_cal)

            export_calibration_plot(
                probability_cal_val, y_test, id_num, dataset, model_type, calibrated=True)

results = pd.DataFrame(results)

results.to_csv(OUTPUT_PATH + 'results.csv', index=False)
