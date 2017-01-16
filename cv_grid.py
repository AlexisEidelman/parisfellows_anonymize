# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 10:46:55 2016

@author: babou
"""

import pandas as pd
import numpy as np

import xgboost as xgb
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import f1_score

from scipy.stats import sem

def display_scores(params, scores, append_star=False):
    """Format the mean score +/- std error for params"""
    params = ", ".join("{0}={1}".format(k, v)
                      for k, v in params.items())
    line = "{0}:\t{1:.3f} (+/-{2:.3f})".format(
        params, np.mean(scores), sem(scores))
    if append_star:
        line += " *"
    return line

def display_grid_scores(grid_scores, top=None):
    """Helper function to format a report on a grid of scores"""

    grid_scores = sorted(grid_scores, key=lambda x: x[1], reverse=True)
    if top is not None:
        grid_scores = grid_scores[:top]

    # Compute a threshold for staring models with overlapping
    # stderr:
    _, best_mean, best_scores = grid_scores[0]
    threshold = best_mean - 2 * sem(best_scores)

    for params, mean_score, scores in grid_scores:
        append_star = mean_score + 2 * sem(scores) > threshold
        print(display_scores(params, scores, append_star=append_star))



def xg_f1(yhat, y):
    y = y.get_label()
    y_bin = [1. if y_cont > 0.5 else 0. for y_cont in yhat] # binaryzing your output
    return 'f1',f1_score(y, y_bin)

path_jurinet = '/home/sgmap/data/jurinet'
path_out = os.path.join(path_jurinet, 'words.csv')
data = pd.read_csv(path_out, encoding='utf-8')#, dtype=dtype)

# useless means "not in the model"
mots_cols = [x for x in data.columns if x.startswith('mot')]
useless_col = mots_cols + ['doc_name', 'paragraph_nb']
useful_col = [col for col in data.columns if col not in useless_col]

useless_data = data[useless_col].copy()

data = data[useful_col]
target = data['is_target']
data = data.drop('is_target', axis=1)
X = data

ratio = float(np.sum(target == 0)) / np.sum(target==1)

clf = xgb.XGBClassifier(learning_rate=0.1, max_depth=8, n_estimators=220,
                        nthread=-1, scale_pos_weight=ratio, seed=42)
skf = StratifiedKFold(target, n_folds=5, random_state=2016)

grid = GridSearchCV(clf, param_grid=params, cv=skf, scoring='f1', early_stopping_rounds=25)
                    , early_stopping_rounds=25)

grid.fit(X, target)
xgtrain = xgb.DMatrix(X.values, target.values)
cvresult = xgb.cv(params, xgtrain, 300, nfold=5, feval=xg_f1, maximize=True, early_stopping_rounds=30)

score = cross_val_score(clf, X, target, cv=skf, scoring='f1')

params = {
    'max_depth':[6, 7, 8],
    'min_child_weight': [0.8, 0.9, 0.95],
    'subsample' : [0.8, 0.9, 0.95],
}


gsearch1 = GridSearchCV(estimator = xgb.XGBClassifier(learning_rate =0.1, n_estimators=350, max_depth=8,
                                        objective= 'binary:logistic', nthread=4, scale_pos_weight=ratio, seed=42),
                       param_grid = params, scoring='f1',n_jobs=4, cv=skf)
gsearch1.fit(X,target)

print(display_grid_scores(gsearch1.grid_scores_))
