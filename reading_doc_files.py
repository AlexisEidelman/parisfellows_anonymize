﻿"""
Created on Mon Jul 18 12:24:13 2016

@author: babou
"""
import pandas as pd
import re
import nltk
import random

from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.preprocessing import LabelEncoder


stopword_fr = [word for word in stopwords.words('french')]

target_dict = {u'X\xe2\x80\xa6' : 'X', u'X' : 'X', u'X..' : 'X', u'X.' : 'X',
               u'X\u2026' : 'X', u'M.X' : 'X', u'M.X\u2026' : 'X',
               u'Y\xe2\x80\xa6' : 'Y', u'Y' : 'Y', u'Y..' : 'Y', u'Y.' : 'Y',
               u'Y\u2026' : 'Y', u'M.Y' : 'Y', u'M.Y\u2026' : 'Y',
               u'Z\xe2\x80\xa6': 'Z', u'Z' : 'Z', u'Z..' : 'Z', u'Z.' : 'Z',
               u'Z\u2026' : 'Z', u'M.Z' : 'Z', u'M.Z\u2026' : 'Z'}

#punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''

#################################################
###                 FUNCTIONS                 ###
#################################################


#def get_target_name(sentence):
#    for i in range(len(sentence.split(' '))):
#        w = sentence.split(' ')[i].replace(',', '').replace('.', '')
#        if w in target_list:
#            return target_dict.keys.index(w)
def noms_retires(mots_init, mots_modif, verbose=False):
    ''' travaille au niveau mot via word_tokenize
        retoure une liste de booléens de la taille du text_init en mots
        La valeur correpondant u k-ième mot vaut True ssi le mot est différent
        dans text_modifie
    '''
    pseudonym = [False]*len(mots_init)
    for k in range(len(mots_init)):
        if mots_modif[k] != mots_init[k]:
            if verbose:
                print(mots_modif[k], mots_init[k])
            pseudonym[k] = True
    return pseudonym


def get_target(sentence):
    for i in range(len(sentence.split(' '))):
        w = sentence.split(' ')[i].replace(',', '').replace('.', '')
        if w in target_dict.keys():
            return True
    return False

#def remove_ponctuation(sentence):
#    sentence_with_no_punct = ""
#    for char in sentence:
#       if char not in punctuations:
#           sentence_with_no_punct = sentence_with_no_punct + char
#    return sentence_with_no_punct



#################################################
###                 GENERAL                   ###
#################################################

import os
#path_jurinet = "C:\git\pseudonymisation\jurinet"
#path_data = os.path.join(path_jurinet, 'good')
#jurinet_files = [x for x in os.listdir(path_data) if 'jurinet' in x]
#path_files = [os.path.join(path_data, f) for f in jurinet_files]
#
#
#def df_from_all_docs(path_files):
#    '''
#    retourne un df avec tous les mots de chaque document de la
#    liste path_files
#        - les mots sont entendu au sens word_tokenize
#        - le dataframe présente sur chaque ligne :
#            - le mot (dans le texte jurinet)
#            - le mot (dans le texte dila même si c'est inutile)
#            - est-ce que ce mot est "tagged" ou non
#            - le nom du document auquel appartient le doc
#    '''
#
#    words_df = pd.DataFrame()
#    document_temp = pd.DataFrame()
#
#    for path_file in path_files:
#        num_file = os.path.basename(path_file)[:-4].split('_')[0]
#        print ("Loading file : " + os.path.basename(path_file))
#
#        with open(path_file, 'r') as jurinet:
#            jurinet_text = jurinet.read()
#        with open(path_file.replace('_jurinet', '_dila'), 'r') as dila:
#            dila_text = dila.read()
#
#        mots_init = word_tokenize(jurinet_text, language='french')
#        text_modifie = re.sub(r'([A-Z])\.\.\.', r'\1zzz', dila_text)
#        text_modifie = re.sub(r'zzz-\w', r'', text_modifie)
#        mots_modif = word_tokenize(text_modifie, language='french')
#
#        tagged = noms_retires(mots_init, mots_modif, verbose=False)
#
#        document_temp = pd.DataFrame.from_dict({
#        'mot': mots_init,
#        'tagged': tagged,
#        })
#        document_temp['doc_name'] = num_file
#        document_temp['rank_word'] = range(len(document_temp))
#
#        words_df = pd.concat([words_df, document_temp])
#
#    return words_df
#
#words_df = df_from_all_docs(path_files)
#print(words_df.groupby('doc_name')['tagged'].sum())
#print(words_df.groupby('doc_name')['tagged'].sum().value_counts())


path_jurinet = '/home/sgmap/data/jurinet'
path_jurinet = 'D:\data\jurinet'


path_file = os.path.join(path_jurinet, 'labelises.csv')

tab = pd.read_csv(path_file, nrows=None, sep=';')

tab['text'] = tab['jurinet_standard'].apply(
    lambda x: word_tokenize(x, language='french')
    )

tab['label'] = tab['label'].str[1:-1].str.split(', ')

assert all(tab['text'].apply(len) == tab['label'].apply(len))


# Old
#    tags = tagger.TagText(row.['paragraph'])
#    tags2 = treetaggerwrapper.make_tags(tags)
#    for w in tags2:
#        word_list.append({'mot' : w[0],
#                          'type' : w[1],
#                            'lemma' : w[2]})

print("-"*54)
print(" PREPROCESSING...")



def read_name_list():
    '''Loading annexe files (Firstaname / Names etc;..)'''
    # Reading French's firstnames file
    firstname_df = pd.read_csv(os.path.join(path_jurinet, 'nat2015.csv'))
    # old version before Insee firstname file
    # firstname_df = pd.read_csv('other/data/prenom_clean.csv', encoding='utf-8',
    #                            header=None, names = ["firstname"])
    # Use Majuscule in first carac
    firstname_list = firstname_df['preusuel'].str.title().unique().tolist()

    # Reading foreign's firstnames file
    foreign_firstname_df =  pd.read_csv('other/data/foreign_fistname_clean.csv', encoding='utf-8',
                                 header=None, names = ["firstname"])
    foreign_firstname_list = foreign_firstname_df.firstname.tolist()

    # Name list
    name_df = pd.read_csv('other/data/top_8k_name.csv', encoding='utf-8')
    name_list = name_df.name.tolist()
    return firstname_list, foreign_firstname_list, name_list

firstname_list, foreign_firstname_list, name_list = read_name_list()

mister_list = [u'M', u'M.', u'Madame', u'Mme', u'Monsieur', u'Dr', u'Monsieur',
               u'MM', 'Demoiselle',
               'consorts',
               'veuve']


def caracteristique_du_mot(words_df,
                          firstname_list,
                          foreign_firstname_list,
                          mister_list):
    # nombre occurence dans le doc
    count_mot = words_df.groupby(['doc_name', 'mot']).size().to_frame('nb_mot').reset_index()
    words_df = words_df.merge(count_mot, on = ['doc_name', 'mot'])

    stopword_fr = [word for word in stopwords.words('french')]
    words_df['is_stopword'] = words_df['mot'].str.lower().isin(stopword_fr)
    words_df['is_first_char_upper'] = words_df['mot'].str[0].str.isupper()
    words_df['is_upper'] = words_df['mot'].str.isupper()
    words_df['len_word'] = words_df['mot'].str.len()
    words_df['is_firstname'] = words_df['mot'].isin(foreign_firstname_list)
    words_df['is_french_firstname'] = words_df['mot'].isin(firstname_list)
    words_df['is_mister_word'] = words_df['mot'].isin(mister_list)
    # id encoding usefull ?
    lbl = LabelEncoder()
    words_df['word_encoded'] = lbl.fit_transform(list(words_df['mot'].values))

    stemmer = SnowballStemmer("french")
    words_df['stem'] = words_df['mot'].apply(stemmer.stem)
    lbl = LabelEncoder()
    words_df['stem_encoded'] = lbl.fit_transform(list(words_df['stem'].values))
    return words_df


def shift_words_data(words_df,
                     nb_mot_avant, nb_mot_apres,
                     liste_caracteristiques):
    ''' update words_df table (by doc_name'''
    for k in range(nb_mot_avant, nb_mot_apres):
        liste_caracteristiques_k = [nom + ' ' + str(k) for nom in liste_caracteristiques]
        if k  != 0:
            words_df[liste_caracteristiques_k] = \
                words_df.groupby(['doc_name'])[liste_caracteristiques].apply(lambda x: x.shift(k))
    return words_df


caracteristiques_mot = ['mot', 'is_stopword', 'is_first_char_upper',
                       'is_upper', 'len_word', 'is_mister_word', 'word_encoded',
                       'is_firstname','is_french_firstname',
                       'nb_mot']


for k in range(20):

    print(k)
    list_doc_name = range(k, len(tab), 20)
    words_df = pd.DataFrame()
    subset = tab.iloc[list_doc_name]
    subset = subset[['text', 'label']]
    
    for idx, decision in subset.iterrows():
        document_temp = pd.DataFrame.from_dict({
        'mot': decision['text'],
        'tagged': decision['label'],
        })
        document_temp['doc_name'] = idx
        document_temp['rank_word'] = range(len(document_temp))

        words_df = pd.concat([words_df, document_temp])

    words_df['tagged'] = words_df['tagged'] == 'True'
    words_df_k = words_df[words_df['doc_name'].isin(list_doc_name)]
    words_df_k = caracteristique_du_mot(words_df_k,
                          firstname_list,
                          foreign_firstname_list,
                          mister_list)
    words_df_k = shift_words_data(words_df_k, -4, 5, caracteristiques_mot)



    ##########################################################
    ####             Caractéristique de la position        ###


    # to have granularite
    words_df_k['temp_count'] = 1
    # Cumulative sum of word by paragraph
    #words_df_k['paragraph_cum_word' ] = words_df_k.groupby(['doc_name', 'paragraph_nb'])['temp_count'].cumsum()
    # rank since last ";" or "."
    ## Create a bool a each end of sentence
    words_df_k["end_point"] = words_df_k['mot'].isin([";", "."])
    #end_point = words_df_k['mot'].isin([";", "."])
    # words_df_k['end_point'] = words_df_k['rank_word'][end_point]

    words_df_k['temp_count'] = 1
    words_df_k['end_point_cum' ] = words_df_k.groupby(['doc_name'])['end_point'].cumsum()
    words_df_k['end_point_size'] = words_df_k.groupby(['doc_name', 'end_point_cum'])['temp_count'].transform(sum)
    words_df_k['end_point_cum_word'] = words_df_k.groupby(['doc_name', 'end_point_cum'])['temp_count'].cumsum()
    words_df_k['end_point_cum_word_reverse'] = words_df_k['end_point_size'] - words_df_k['end_point_cum_word']
    #words_df_k = words_df_k.drop(['temp_count', 'end_point', 'end_point_cum'], axis=1)

    # Cumulative sum of word by senstence end by ","
    ## Create a bool a each end of sentence
    words_df_k["end_comma"] = words_df_k['mot'].isin([",",";", "."])
    words_df_k['end_comma_cum' ] = words_df_k.groupby(['doc_name'])['end_comma'].cumsum()
    words_df_k['end_comma_size'] = words_df_k.groupby(['doc_name', 'end_comma_cum'])['temp_count'].transform(sum)
    words_df_k['end_comma_cum_word' ] = words_df_k.groupby(['doc_name', 'end_comma_cum'])['temp_count'].cumsum()
    words_df_k['end_comma_cum_word_reverse' ] = words_df_k['end_comma_size'] - words_df_k['end_comma_cum_word']

    # Del temp preprocessing features
    words_df_k = words_df_k.drop(['temp_count', 'end_comma', 'end_comma_cum',
                           'end_point', 'end_point_cum'], axis=1)

    # TODO: entre is_entre_guillemet

    # Fillna Nan word shift
    #words_df_k = words_df_k.fillna(-1)

    print(" EXPORT...")
    path_out = os.path.join(path_jurinet, 'words' + str(k) + '.csv')
    words_df_k.to_csv(path_out, encoding='utf-8', index=False)
