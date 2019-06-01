# Expected_Goals

A football expected goals model

Built an end-to-end prediction model for each shot in a football match. The predictions are calibrated so each value is the probability of a shot resulting in a goal.

Interesting things about this project include the structure as well as the calibration. The code is structured as a pipeline with the code intending to be reusable and modular as opposed to a once off project. Models are tuned using predefined hyperparameter space stored in config files.

The models are evaluated both by the prediction performance as well as the calibration of their predictions. The latter is evaluated using visual inspection of their calibration curves as opposed to a single metric, which is usually advised.

Keywords: Data Mining / Python (pandas, matplotlib) / Random Forest / Logistic Regression / XGBoost / SVM / Gradboost / Calibration / Model Pipeline
