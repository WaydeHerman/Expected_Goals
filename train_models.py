
import pandas as pd
from functions import train_model

results = pd.read_csv('Results/results.csv', index_col='id_num')

train_model('direct', 4, results)
train_model('head_cross', 12, results)
train_model('cross', 17, results)
train_model('head', 21, results)
train_model('regular', 25, results)
