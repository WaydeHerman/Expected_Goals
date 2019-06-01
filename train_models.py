
import pandas as pd
from functions import train_model

results = pd.read_csv('Results/results.csv', index_col='id_num')

train_model('direct', 2)
# train_model('head_cross', 2)
# train_model('cross', 2)
# train_model('head', 2)
# train_model('regular', 2)
