# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 18:02:27 2016

@author: aeidelman
"""

import operator
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score, classification_report, accuracy_score, roc_auc_score
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns



def recall(yhat, y):
    y = y.get_label()
    y_pred = [1 if pred > 0.5 else 0 for pred in yhat]
    return "recall", recall_score(y, y_pred)


def xg_f1(yhat, y):
    y = y.get_label()
    y_bin = [1. if y_cont > 0.5 else 0. for y_cont in yhat] # binaryzing your output
    return 'f1',f1_score(y, y_bin)


def create_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    for i, feat in enumerate(features):
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
    outfile.close()


def xgboost_feature_importance(model, train, return_df=False):


    features = train.columns
    create_feature_map(features)

    importance = model.get_fscore(fmap='xgb.fmap')
    importance = sorted(importance.items(), key=operator.itemgetter(1))

    df = pd.DataFrame(importance, columns=['feature', 'fscore'])
    df['fscore'] = df['fscore'] / df['fscore'].sum()

    sns.barplot(x="fscore", y="feature", data=df)
#    plt.xticks(range(len(df)), df.feature.tolist(), rotation=60)
    plt.title('Feature Importances')
    plt.ylabel('Relative Importance')

    print(df)

    if return_df is True:
        return df


def get_metric(y_test, y_pred, plot=False):
    """
    Calcul metrics.
    In : y_test, y_pred
    Return :
    If plot == True, then plot CM normalize
    """
    # Metrics
    metrics_classification = classification_report(y_test, y_pred)

    accuracy = accuracy_score(y_test, y_pred)

    auc = roc_auc_score(y_test, y_pred)

    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("Metrics classification : ")
    print(metrics_classification)
    print("Accuracy score : ")
    print(accuracy )
    print("Roc auc score : ")
    print(auc)
    print("Recall score : ")
    print(recall)
    print("F1 score : ")
    print(f1)


def get_confusion(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    label_unique = y_test.unique()
#    #Graph Confusion Matrix
    tick_marks = np.arange(len(label_unique))
#    plt.figure(figsize=(8,6))
    sns.heatmap(cm_normalized, cmap='Greens',annot=True,linewidths=.5)
#    plt.title('confusion matrix')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.xticks(tick_marks + 0.5, list(label_unique))
    plt.yticks(tick_marks + 0.5,list(reversed(list(label_unique))) , rotation=0)
#
#    plt.imshow(cm_normalized, interpolation='nearest', cmap='Greens')
#    plt.title('confusion matrix')
#    plt.colorbar()
#    tick_marks = np.arange(len(label_unique))
#    plt.xticks(tick_marks + 0.5, list(reversed(list(label_unique))))
#    plt.yticks(tick_marks + 0.5,list(label_unique) , rotation=0)
#    plt.tight_layout()
#    plt.ylabel('True label')
#    plt.xlabel('Predicted label')


def get_graph_features_mean(tab, col_list):

    am = tab.set_index('features')
    am = am.stack().reset_index()
    am.columns = ['features', 'type', 'score']
    plt.figure(figsize=(18,6));

    if isinstance(col_list, list):
        print("List")
        sns.barplot('score', 'features', data=am[am.features.isin(col_list)],
                    hue='type', hue_order=['error','good'])
    else:
        sns.barplot('score', 'features', data=am[am.features == col_list],
                    hue='type', hue_order=['error','good'])

    #    plt.xticks(rotation=40)
    plt.legend()
