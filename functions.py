
from matplotlib import pyplot
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import minmax_scale
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def get_tuned_model(model_type, params, seed):
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
