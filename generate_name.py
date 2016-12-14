# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 16:25:27 2016

@author: User
"""

#################################################
###         Simulate Name & Firstame          ###

words_df['is_target'] = words_df['word'].apply(lambda x: 1 if x in target_dict.keys() else 0)
# To make some difference beetwen a name is_target and other target. To delete btw
words_df['is_name'] = 0
words_df.loc[words_df['is_target'] == 1, 'is_name'] = 1

## Insert random first name in place of 'X', Mr 'X...' ...
#

# /!\ Delete when true data
def get_random_firstname():
    """
    Use a first random (0/1) to chose between French's firstname or foreign
    And a random firstname in the random chosen dataset
    """
    # first random to chose the dataset
    first_random = random.randint(0,1)
    # French
    if first_random == 0:
        random_idx = random.randint(0, 2046) # len de firstname_list
        return firstname_list[random_idx]
    # foreign
    else:
        random_idx = random.randint(0, 36114) # len de firstname_list
        return foreign_firstname_list[random_idx]

def get_random_name():
    """
    Return a random name en CAPS
    """
    random_idx = random.randint(0, 7999) # len de name_list
    return name_list[random_idx]

def insert_row(x, idx):
    """
    Return a Dataframe with :
    - Random Firstname (French or Foreigner)
    - is_target = 1
    -
    """
    line = {'word' : get_random_firstname() ,
            'doc_name' :  x['doc_name'],
            'paragraph_nb' :  x['paragraph_nb'],
            'is_target' :  1,
            'is_firstname' :  0,
            'word_shift_1b' :  '',
            'is_firstname_1b' :  '',
            'is_firstname_1b_shift' :  '',
            'add_row' : 1,#
            'is_target_1b' : ''}
    return pd.DataFrame(line, index=[idx])


# Replace X.. Y by Radnom Name
words_df.loc[words_df['is_target'] ==1, 'word'] = words_df['word'].apply(lambda x: get_random_name())
# Check if word is Firstname then is_target :
words_df['is_firstname'] = words_df['word'].apply(lambda x: 1 if x.title() in firstname_list else 0)



i = 0
for idx, row in words_df[(words_df.is_target_1b == 1) & (words_df.is_firstname == 0)
                        & (words_df.is_firstname_1a == 0)].iterrows():

# Random Insert Random Firstname before Name if no firstname detect
print("  INSERT ROWS FOR SIMULATE FIRSTNAME")

words_df['add_row'] = 0

i = 0
for idx, row in words_df[(words_df.is_target_1b == 1) & (words_df.is_firstname == 0)
                        & (words_df.is_firstname_1a == 0)].iterrows():
    # Add some random (sometime Firstane sometime no...)
    if random.randint(0, 1) == 1:
        print(str(idx+i))
        # Si la row suivante est target et que ce n'est pas un Firstname
        # et que la row actuelle n'est pas un Firstname ==> Add Random Firstname
        line = insert_row(row, idx)
        words_df = pd.concat([words_df.ix[:idx + i ], line, words_df.ix[idx+1 + i:]]).reset_index(drop=True)
        i+=1
    

#Delete old features
words_df = words_df.drop(['is_firstname_1b', 'is_firstname_1b_shift', 'is_target_1b', 
                        'word_shift_1b', 'is_name', 'is_firstname_1a', 'add_row'], axis=1)

words_df = words_df[words_df.word != u'\u2026'] # To clean anonymasation from dataset and bad ML
words_df = words_df.reset_index(drop=True)
