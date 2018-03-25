"""
Feature Extraction - TITANIC
1. fillna - embarked
2. extract is_male
3. label embared
4. extract is_master, is_mr
5. fillna and extract age and cat_age
"""

import numpy as np
import pandas as pd

def feature_extraction(df):
    df = fillna_embarked(df)
    df = label_male(df)
    df = label_embarked(df)
    df = extract_title(df)
    df = process_age(df)
    return df


# col Embarked - map string to int
def fillna_embarked(df):
    emb_mode = df.Embarked.mode()[0]
    df.Embarked = df.Embarked.fillna(emb_mode)
    return df

# col is_male - 1 if True
def label_male(df):

    def is_male(x):
        if x == 'male':
            return True
        else:
            return False
    
    df['is_male'] = df.Sex.apply(is_male)
    return df

# col embarked
def label_embarked(df):
    df.Embarked = pd.Categorical(df.Embarked)
    df['Embarked_codes'] = df.Embarked.cat.codes
    return df

# col is_master and is_mr
def extract_title(df):

    def is_master(x):
        if x == 'Master':
            return True
        return False

    def is_mr(x):
        if x == 'Mr':
            return True
        return False

    df['is_master'] = df.Title.apply(is_master)
    df['is_mr'] = df.Title.apply(is_mr)
    return df

# process age
def process_age(df):

    def cat_age(age):
        if age <= 5:
            return 0
        elif age > 5 and age <= 15:
            return 1
        elif age > 15 and age <= 30:
            return 2
        elif age > 30 and age <= 60:
            return 3
        elif age > 60:
            return 4
        else:
            return np.nan

    def fillna_median(x):
        return x.fillna(x.median())
        df['cat_age'] = df.Age.apply(cat_age)

    df.Age = (df[['Age', 'Title']]
                .groupby('Title')
                .transform(fillna_median))

    df.cat_age = (df[['cat_age', 'Title']]
                    .groupby('Title')
                    .transform(fillna_median))

    df.cat_age = df.cat_age.astype(int)
    
    return df
