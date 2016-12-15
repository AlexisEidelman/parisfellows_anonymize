# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 18:07:26 2016

@author: babou
"""

import pandas as pd
import numpy as np
import pickle

import xgboost as xgb
from sklearn.cross_validation import train_test_split

from evaluate import xg_f1, f1_score, get_metric, confusion_matrix, get_graph_features_mean

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

data = pd.read_csv('data/data.csv', encoding='utf-8')#, dtype=dtype)

# To test features selection
delcol = [u'word_encoded'] # with 0.9344
data = data.drop(delcol, axis=1)

useful_col = [col for col in data.columns if col not in ['mot', 'doc_name', 'paragraph_nb', 'firstname_is_french',
                                                         'admin_name', 'add_row', 'admin_firstname']]
useful_col = [col for col in useful_col if col[:3] not in ['ste', 'mot']]

word_save = data['mot']
doc_name_save = data['doc_name']
#paragraph_nb_save = data['paragraph_nb']
firstname_is_french_save = data['is_french_firstname']
#admin_name_save = data['admin_name']
#add_row_save = data['add_row']
#admin_firstname_save = data['admin_firstname']


data = data[useful_col]
y = data['tagged']
data = data.drop('tagged', axis=1)
X = data

ratio = float(np.sum(y == 0)) / np.sum(y==1)

if DUMMY == True:
    X = pd.get_dummies(X)

X.fillna(False, inplace=True)

# Split data to get an unknow dataset (valide):
X_trainning, X_valide, y_trainning, y_valide = train_test_split(X, y, stratify=y,
                                                                   test_size=0.20, random_state=20)


# Split data to get X_train / X_test :
X_train, X_test, y_train, y_test = train_test_split(X_trainning, y_trainning, stratify=y_trainning,
                                                                   test_size=0.33, random_state=21)



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

num_round = 400

bst = xgb.train(params, dtrain, num_round, evallist, early_stopping_rounds=25,
                feval=xg_f1, maximize=True)

# Validation on X_test
y_pred = bst.predict(xgb.DMatrix(X_test, missing=-1), ntree_limit=bst.best_ntree_limit)
y_pred_b = [1. if y_cont > 0.5 else 0. for y_cont in y_pred] # binaryzing your output
f1_test = f1_score(y_test, y_pred_b)
print("F1 score on Test dataset: "+ str(f1_test))

# Validation on Valide dataset
y_pred_valide = bst.predict(xgb.DMatrix(X_valide, missing=-1), ntree_limit=bst.best_ntree_limit)
y_pred_valide_b = [1. if y_cont > 0.5 else 0. for y_cont in y_pred_valide] # binaryzing your output
f1_valide = f1_score(y_valide, y_pred_valide_b)
print("F1 score on unknow dataset: "+ str(f1_valide))


X_valide = X_valide.join(word_save)
X_valide = X_valide.join(doc_name_save)
#X_valide = X_valide.join(paragraph_nb_save)
#X_valide = X_valide.join(admin_name_save)
#X_valide = X_valide.join(add_row_save)
#X_valide = X_valide.join(admin_firstname_save)


X_valide['tagged'] = y_valide
X_valide['y_pred'] = y_pred_valide_b
X_valide['y_pred_proba'] = y_pred_valide
X_valide['error'] = X_valide['tagged'] != X_valide['y_pred']

data['mot'] = word_save
data['doc_name'] = doc_name_save
#data['admin_name'] = admin_name_save
#data['add_row'] = add_row_save
#data['admin_firstname'] = admin_firstname_save

print("_"*54)
print("Some metrics : ")
get_metric(y_valide, y_pred_valide_b)

print("_"*54)
print("Confusion matrix : ")
cm = confusion_matrix(y_valide, y_pred_valide_b)
print(cm)


path_model = 'model/' + MODEL_NAME + "_" + str(f1_valide) + ".model"
print("Export model in " + str(path_model))
f = open(path_model, 'wb')
pickle.dump(bst, f)
f.close()


# Analyse error :
error = X_valide[X_valide.error == 1]
# False Positive selector
fp = X_valide[(X_valide.error == 1) & (X_valide.tagged == 1)]
# False Negative selector
fn = X_valide[(X_valide.error == 1) & (X_valide.tagged == 0)]

good = X_valide[(X_valide.error ==0) & ( X_valide.tagged == 1)]

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