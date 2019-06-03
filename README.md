# Expected_Goals

An end-to-end football expected goals model. The predictions are the probability of a shot resulting in a goal.

Interesting things about this project include the structure as well as the evaluation of calibration. The code is structured as a pipeline with the code intending to be reusable and modular as opposed to a once off project. Models are tuned using predefined hyperparameter space stored in config files.

The models are evaluated both by the prediction performance as well as the calibration of their predictions. The latter is evaluated using visual inspection of their calibration curves as opposed to a single metric, which is usually advised.

**Keywords:** Python (Pandas, Matplotlib) / Logistic Regression / SVM / Random Forest / Gradboost / XGBoost / Calibration / Model Pipeline

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

1. Run the JSON processing script:

   ```bash
   python JSON_to_raw.py
   ```

   This will extract the individual shot data from the scraped JSON files. These are
   Opta data scraped from a website I won't be mentioning. there is a JSON file for
   each season in each of the top five european leagues, namely 'Premier League',
   'Serie A', 'La Liga', 'Bundesila', and 'Ligue 1'. The output of these are stored
   in the `Data/raw` directory as `shots_league_season.csv`.

   Note: these are available to download [Here](https://drive.google.com/file/d/1RgRst1HY7AmfaEySKf7qE-TpPzSXc9kD/view?usp=sharing).

2) Run the raw processing script:

   ```bash
   python raw_to_features.py
   ```

   This script divides the shot data into subtypes. These subtypes differ significantly
   and are therefore modelled seperately. These subtypes are 'direct', 'head_cross',
   'cross', 'head', and 'regular'. One Hot Encoding and other feature engineering is
   applied here. The features to be used by each subtype are stored in a dictionary
   within `config/features_config.yaml`. The output of these are stored within the
   `Data/processed` directory as csv's and may be inspected by using `EDA.ipynb`.

### Hyperparameter tuning:

1. Run the hyperparameter tuning script:

   ```bash
   python hyperparameter_tuning.py
   ```

   This script tunes models for each shot subtype. The models to be tuned may be chosen
   from a predefined list. The hyperparameter space to be searched, the way they are to
   be searched (ie grid search or random search), whether scaling is required by the model,
   as well as the number of iterations for random search are all stored in yaml config
   files for each model type. ie `config/LogisticRegression_config.yaml`. Models are
   selected based on their roc auc score. The best N models are then further evaluated
   by their calibration curves. Calibration curves for each top n model as well as their
   calibrated version are saved in `Results` as well as a csv storing the best N model's
   hyperparameters as well as their scores.

2. Inspect results:

   The scores for each model may be inspected using the `Results.ipynb` notebook. These
   scores along with the calibration curves for each model, stored in `Results`, are
   used to determine which model to use for prediction. Calibration curves are plots of
   the relative frequency between the target and the predictions. These are plotted
   above the distribution of the prediction frequences to determine which frequencies
   are the most important.

   ![example calibration curve](/Results/55_regular_RandomForestClassifier_False.png "Example Calibration Curve")

3. Run the train models script:

   ```bash
   python train_models.py
   ```

   This script trains the final model for each subtype using the `id_num`. These models,
   along with any scaling, are saved as joblib files.

### Predictions:

1. Run the prediction script:

   ```bash
   python predictions.py
   ```

   This script predicts the expected goals for a game. Producing an image revealing the
   shot location and expected goal for each shot in the game. It also provides the sum
   for each team.

   ![example result](/Matches/Arsenal_Chelsea_2014-2015.png "Example Result")
