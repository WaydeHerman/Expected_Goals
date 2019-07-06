
from functions import train_model

# train final model for each shot subtype using hyperparameter id_num:
train_model('direct', 2)
train_model('head_cross', 7)
train_model('cross', 9)
train_model('head', 16)
train_model('regular', 17)
