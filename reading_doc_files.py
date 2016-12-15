# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 12:24:13 2016

@author: babou
"""
import pandas as pd
import re
import nltk
import random

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

mister_list = [u'M', u'M.', u'Madame', u'Mme', u'Monsieur', u'Dr', u'Monsieur', u'MM']

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
path_jurinet = "C:\git\pseudonymisation\jurinet"
path_data = os.path.join(path_jurinet, 'good')
jurinet_files = [x for x in os.listdir(path_data) if 'jurinet' in x]
path_files = [os.path.join(path_data, f) for f in jurinet_files]


#    with open(dila_file, 'r') as dila:
#        dila_text = dila.read()
#
#    dila_text = re.sub(r'([A-Z])\.\.\.', r'\1zzz', dila_text)
##    text1bis = text1.replace('...', 'zzz')
#    dila_text = re.sub(r'zzz-\w', r'', dila_text)
#    mots_dila = word_tokenize(dila_text, language='french')
#    mots_juri = word_tokenize(jurinet_text, language='french')
#
#    for k in range(len(mots_dila)):
#        if mots_dila[k] != mots_juri[k]:
#            print(mots_dila[k], mots_juri[k])

words_df = pd.DataFrame()
document_temp = pd.DataFrame()


for path_file in path_files:
    num_file = os.path.basename(path_file)[:-4].split('_')[0]
    print ("Loading file : " + os.path.basename(path_file))
    
    with open(path_file, 'r') as jurinet:
        jurinet_text = jurinet.read()
    with open(path_file.replace('_jurinet', '_dila'), 'r') as dila:
        dila_text = dila.read()
        
    mots_init = word_tokenize(jurinet_text, language='french')
    text_modifie = re.sub(r'([A-Z])\.\.\.', r'\1zzz', dila_text)
    text_modifie = re.sub(r'zzz-\w', r'', text_modifie)
    mots_modif = word_tokenize(text_modifie, language='french')
    
    tagged = noms_retires(mots_init, mots_modif, verbose=False)
    
    document_temp = pd.DataFrame.from_dict({
    'mot': mots_init,
    'tagged': tagged,    
    })
    document_temp['doc_name'] = num_file
    document_temp['rank_word'] = range(len(document_temp))
    
    words_df = pd.concat([words_df, document_temp])

print(words_df.groupby('doc_name')['tagged'].sum())
print(words_df.groupby('doc_name')['tagged'].sum().value_counts())

# Old
#    tags = tagger.TagText(row.['paragraph'])
#    tags2 = treetaggerwrapper.make_tags(tags)
#    for w in tags2:
#        word_list.append({'mot' : w[0],
#                          'type' : w[1],
#                            'lemma' : w[2]})

print("-"*54)
print(" PREPROCESSING...")





## Loading annexe files (Firstaname / Names etc;..)
# Reading French's firstnames file
firstname_df = pd.read_csv('other/data/prenom_clean.csv', encoding='utf-8',
                           header=None, names = ["firstname"])
# Use Majuscule in first carac
firstname_df['firstname'] = firstname_df['firstname'].str.title()
firstname_list = firstname_df.firstname.tolist()

# Reading foreign's firstnames file
foreign_firstname_df =  pd.read_csv('other/data/foreign_fistname_clean.csv', encoding='utf-8',
                             header=None, names = ["firstname"])
foreign_firstname_list = foreign_firstname_df.firstname.tolist()

# Name list
name_df = pd.read_csv('other/data/top_8k_name.csv', encoding='utf-8')
name_list = name_df.name.tolist()


##################################################
####             Caractéristique du mot        ###

# Check if word is Firstname then is_target :

words_df['is_stopword'] = words_df['mot'].str.lower().isin(stopword_fr)
words_df['is_first_char_upper'] = words_df['mot'].str[0].str.isupper()
words_df['is_upper'] = words_df['mot'].str.isupper()
words_df['len_word'] = words_df['mot'].str.len()
words_df['is_firstname'] = words_df['mot'].isin(firstname_list)
words_df['is_foreign_firstname'] = words_df['mot'].isin(foreign_firstname_list)
words_df['is_mister_word'] = words_df['mot'].isin(mister_list)
#words_df['is_firstname'] = words_df['mot'].str.title().isin(firstname_list)
## Label encoding word
lbl = LabelEncoder()
words_df['word_encoded'] = lbl.fit_transform(list(words_df['mot'].values))

caracteristique_mot = ['mot', 'is_firstname', 'is_stopword', 'is_first_char_upper',
                       'is_upper', 'len_word', 'is_mister_word', 'word_encoded',
                       'is_foreign_firstname']
for k in range(-4, 5):
    caracteristique_mot_k = [nom + ' ' + str(k) for nom in caracteristique_mot]
    if k  != 0:
        words_df[caracteristique_mot_k] = \
            words_df.groupby(['doc_name'])[caracteristique_mot].apply(lambda x: x.shift(k))
# TODO: peut-être une petite optimisation pour le firstname on peut aller chercher la valuer 

# nombre occurence dans le doc
count_mot = words_df.groupby(['doc_name', 'mot']).size().to_frame('nb_mot').reset_index()
words_df = words_df.merge(count_mot, on = ['doc_name', 'mot'])



##########################################################
####             Caractéristique de la postiion        ###

# to have granularite
words_df['temp_count'] = 1
# Cumulative sum of word by paragraph
#words_df['paragraph_cum_word' ] = words_df.groupby(['doc_name', 'paragraph_nb'])['temp_count'].cumsum()
# rank since last ";" or "."
## Create a bool a each end of sentence
words_df["end_point"] = words_df['mot'].isin([";", "."])
#end_point = words_df['mot'].isin([";", "."])
# words_df['end_point'] = words_df['rank_word'][end_point]

words_df['temp_count' ] = 1
words_df['end_point_cum' ] = words_df.groupby(['doc_name'])['end_point'].cumsum()
words_df['end_point_size'] = words_df.groupby(['doc_name', 'end_point_cum'])['temp_count'].transform(sum)
words_df['end_point_cum_word'] = words_df.groupby(['doc_name', 'end_point_cum'])['temp_count'].cumsum()
words_df['end_point_cum_word_reverse'] = words_df['end_point_size'] - words_df['end_point_cum_word']
#words_df = words_df.drop(['temp_count', 'end_point', 'end_point_cum'], axis=1)

# Cumulative sum of word by senstence end by ","
## Create a bool a each end of sentence
words_df["end_comma"] = words_df['mot'].isin([",",";", "."])
words_df['end_comma_cum' ] = words_df.groupby(['doc_name'])['end_comma'].cumsum()
words_df['end_comma_size'] = words_df.groupby(['doc_name', 'end_comma_cum'])['temp_count'].transform(sum)
words_df['end_comma_cum_word' ] = words_df.groupby(['doc_name', 'end_comma_cum'])['temp_count'].cumsum()
words_df['end_comma_cum_word_reverse' ] = words_df['end_comma_size'] - words_df['end_comma_cum_word']

# Del temp preprocessing features
words_df = words_df.drop(['temp_count', 'end_comma', 'end_comma_cum',
                       'end_point', 'end_point_cum'], axis=1)

# TODO: entre is_entre_guillemet

# Fillna Nan word shift
words_df = words_df.fillna(-1)

#print " EXPORT..."
words_df.to_csv('data/data.csv', encoding='utf-8', index=False)
