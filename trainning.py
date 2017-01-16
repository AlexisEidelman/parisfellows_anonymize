# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 18:07:26 2016

@author: babou
"""
import os
import random
import pandas as pd
import numpy as np
import pickle

import xgboost as xgb
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression

from evaluate import (xg_f1, xg_fbeta_5, fbeta_score, f1_score, get_metric,
                      confusion_matrix, get_graph_features_mean)

DUMMY = False #To dummys caterogical features
MODEL_NAME = "stem"


#################################################
###                 GENERAL                   ###
#################################################

#dtype = {'word_encoded' : 'str',
#         'word_encoded_shift_1b' : 'str',
#         'word_encoded_shift_2b' : 'str',
#         'word_encoded_shift_1a' : 'str',
#         'word_encoded_shift_2a' : 'str'}

path_jurinet = '/home/sgmap/data/jurinet'
path_out = os.path.join(path_jurinet, 'words.csv')
all_data = pd.read_csv(path_out, encoding='utf-8')

# useless means "not in the model"
mots_cols = [x for x in all_data.columns if x.startswith('mot')]
stem_not_encoded = [x for x in all_data.columns if x.startswith('stem') and
    not x.startswith('stem_encoded')]
useless_col = mots_cols + stem_not_encoded + ['doc_name']
useful_col = [col for col in all_data.columns if col not in useless_col]

data = all_data[useful_col]
target = data['tagged']
data = data.drop('tagged', axis=1)

y = target
X = data

ratio = float(np.sum(y == 0)) / np.sum(y==1)

if DUMMY == True:
    X = pd.get_dummies(X)

X.fillna(False, inplace=True)

# On coupe l'échantillon en 3
def split(X, y, ratio_size, groupe=None):
    ''' retourne deux quatres tables. Les deux premières issues de X,
        les deux suivantes issues de y
        - X1 et X2, ont une taille (1-ratio_size)*len(X) et ratio_size*len(X)
        - y1 et y2, ont une taille (1-ratio_size)*len(y) et ratio_size*len(y)
    Sachant que la sélection est faite simultanément sur X et y, on maintient
    donc la cohérence
    '''
    assert ratio_size >= 0
    assert ratio_size <= 1
    # methode 1: on considère les mots indépendamment.
    if groupe is None:
        return train_test_split(X, y, stratify=y, test_size=ratio_size, random_state=20)

    # méthode 2 : on choisit des documents aléatoire
    integer_size = int(ratio_size*(groupe.nunique()))
    valide_doc_sample = random.sample(groupe.unique().tolist(), integer_size)
    cond_valide = groupe.isin(valide_doc_sample)
    X1, y1 = X[~cond_valide], y[~cond_valide]
    X2, y2 = X[cond_valide], y[cond_valide]
    return X1, X2, y1, y2


doc_name = all_data['doc_name']
X_trainning, X_valide, y_trainning, y_valide = split(X, y, 0.2,
                                                     groupe=all_data['doc_name'])

all_data_trainning = all_data.loc[X_trainning.index]
all_data_valide = all_data.loc[X_valide.index]
X_train, X_test, y_train, y_test  = split(X_trainning, y_trainning, 0.33,
                                          groupe=all_data_trainning['doc_name'])
all_data_test = all_data_trainning.loc[X_train.index]
all_data_train = all_data_trainning.loc[X_train.index]

dtrain = xgb.DMatrix(X_train, y_train, missing=-1)
dtest = xgb.DMatrix(X_test, y_test, missing=-1)
evallist = [(dtrain, 'train'), (dtest, 'test')]

params = {'max_depth':9,#12,
         'eta':0.1,#0.01,
         'subsample':0.9,#0.8,
#         'colsample_bytree':0.95,#0.7,
         'silent':1,
         'scale_pos_weight' : ratio,
#         'min_child_weight': 6,
        # 'max_delta_step': 0.086,
         'objective':'binary:logistic',
         'nthread':8,
         'seed':42}

num_round = 40


bst = xgb.train(params, dtrain, num_round, evallist, early_stopping_rounds=5,
               feval=xg_fbeta_5, maximize=True)

# Validation on X_test
y_pred = bst.predict(xgb.DMatrix(X_test, missing=-1), ntree_limit=bst.best_ntree_limit)
y_pred_b = [True if y_cont > 0.5 else False for y_cont in y_pred] # binaryzing your output
f1_test = f1_score(y_test, y_pred_b)
print("F1 score on Test dataset: "+ str(f1_test))

# Validation on Valide dataset
y_pred_valide = bst.predict(xgb.DMatrix(X_valide, missing=-1), ntree_limit=bst.best_ntree_limit)
y_pred_valide_b = [True if y_cont > 0.5 else False for y_cont in y_pred_valide] # binaryzing your output
y_pred_valide_b = pd.Series(y_pred_valide_b, index=X_valide.index)

f1_valide = f1_score(y_valide, y_pred_valide_b)
print("F1 score on unknow dataset: "+ str(f1_valide))


path_model = 'model/' + MODEL_NAME + "_" + str(f1_valide) + ".model"
#print("Export model in " + str(path_model))
#f = open(path_model, 'wb')
#    pickle.dump(bst, f)
#    f.close()


def evaluation_generale(prediction_validation, X = all_data):

    data = X.loc[prediction_validation.index]
    y_valide = data['tagged']

    print("_"*54)
    print("Some metrics : ")
    get_metric(y_valide, prediction_validation)

    print("_"*54)
    print("Confusion matrix : ")
    cm = confusion_matrix(y_valide, prediction_validation)
    print(cm)


first_char_upper = (X_valide.is_first_char_upper)
first_char_upper_and_not_first_place = (X_valide.is_first_char_upper) & (X_valide['end_point_cum_word'] != 1)


logistic = LogisticRegression()
logistic.fit(X_trainning, y_trainning)
logisitc_pred = logistic.predict(X_valide)
logisitc_pred = pd.Series(logisitc_pred, index=X_valide.index)

for prediction in [y_pred_valide_b, first_char_upper,
                   first_char_upper_and_not_first_place,
                   logisitc_pred]:
    evaluation_generale(prediction)



# Analyse error :
X = all_data
prediction = logisitc_pred
data = X.loc[prediction.index]
error = prediction != data['tagged']
data_error = data[error]
fp = data[(error) & (~prediction)] # False Positive selector
fn = data[(error) & (prediction)] # False Negative selector
good = X_valide[(~error) & (prediction)]



bench_features = [col for col in X_valide.columns
                if col not in ['word_encoded_shift_2b', 'word_encoded_shift_2a',
                               'word_encoded_shift_1b', 'word_encoded_shift_1a']]


bench_features_bool = [u'is_firstname', u'is_stopword',
                        u'is_first_char_upper', u'is_upper',
                        u'is_mister_word',u'is_mister_word_1b',
                        u'is_mister_word_2b', u'admin_firstname',
                        u'is_mister_word_1a', u'is_mister_word_2a',
                        u'firstname_is_french', u'admin_name',
                        u'add_row']

bench_continus = [
    # u'paragraph_cum_word',
                 u'end_point_cum_word',
                 u'end_comma_cum_word',
                 # u'paragraph_nb'
                 u'len_word']

analyse_mean = pd.DataFrame(
    {'features' : good.mean().index,
     'good' : good.mean().get_values(),
     'error' : error.mean().get_values(),
     })

get_graph_features_mean(analyse_mean, bench_features_bool)

proba = X_valide['y_pred_proba']
proba[(proba > 0.01) & (proba < 0.99)].hist(bins=100)

fn.doc_name.value_counts()
    # base :                                                                                                # 0.9412 ***
    # without : u'is_mister_word_1b', u'is_mister_word_2b', u'is_mister_word_1a', u'is_mister_word_2a'      # 0.9344
# without : word_encoded'                                                                               # 0.9268