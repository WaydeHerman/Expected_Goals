# Expected_Goals

A football expected goals model

An end-to-end prediction model for each shot in a football match. The predictions are calibrated so each value is the probability of a shot resulting in a goal.

Interesting things about this project include the structure as well as the calibration. The code is structured as a pipeline with the code intending to be reusable and modular as opposed to a once off project. Models are tuned using predefined hyperparameter space stored in config files.

The models are evaluated both by the prediction performance as well as the calibration of their predictions. The latter is evaluated using visual inspection of their calibration curves as opposed to a single metric, which is usually advised.

Keywords: Data Mining / Python (pandas, matplotlib) / Random Forest / Logistic Regression / XGBoost / SVM / Gradboost / Calibration / Model Pipeline

### Setting up:

1. Create virtual environment:
   ```bash
   virtualenv -p python3 venv
   ```
2. Activate virtual environment:
   ```bash
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Data ingest and pre-processing:

1. Run the data JSON processing script:

   ```bash
   python JSON_to_raw.py
   ```

   This will extract the individual shot data from the JSON files. There is a
   JSON file for each season in each of the top five european leagues, namely
   'Premier League', 'Serie A', 'La Liga', 'Bundesila', and 'Ligue 1'. The output
   of these are stored in the `Data/raw` directory as `shots_league_season.csv`.

   Note: these are available to download at 'Insert Download Link'.

2. Run the data raw processing script:

   ```bash
   python raw_to_features.py
   ```

   This script divides the shot data into subtypes. These subtypes differ significantly
   and are therefore modelled seperately. These subtypes are 'direct', 'head_cross',
   'cross', 'head', and 'regular'. One Hot Encoding and other feature engineering is
   applied here. The features to be used by each subtype are stored in a dictionary
   within `config/features_config.yaml`. The output of these are stored within the
   `Data/processed` directory as csv's.

### Hyperparameter optimisation

1. Run the hyperparameter optimisation script:

   ```bash
   python hyperparameter_tuning.py
   ```

1) tune models (hyperparameter_tuning.py)

   - hyperparameter space defined by config files.
   - results evaluated by roc auc then my inspecing the calibration graphs.

1) train the final models (train_models.py)

1) make predictions
