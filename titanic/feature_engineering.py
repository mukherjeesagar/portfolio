"""
Feature Engineering - TITANIC
1. extract names, titles
2. extract fancy titles
3. extract u15 and many siblings
4. extract is_alone
"""

import numpy as np
import pandas as pd
import re


def feature_engineering(df):
    """
    Feature Engineering of the TITANIC data set
    input df, output df
    Process:
    1. extract names, titles
    2. extract fancy titles
    3. extract u15 and many siblings
    4. extract is_alone
    """
    df, titles = fea_extract_names(df)
    df = fea_fancy_title(df, titles)
    df = fea_u15_many_silbings(df)
    df = fea_is_alone(df)
    return df

# FEATURE: Extract surname and titles
def fea_extract_names(df):
    surnames = []
    titles = []

    def extract_names(full_names):
        for full_name in full_names:
            pattern = "^(.*), (.*?)\."
            result = re.findall(pattern, full_name)
            if not result:
                surnames.append('error')
                titles.append('error')
            else:
                surnames.append(result[0][0])
                titles.append(result[0][1])

    full_names = df.Name.values
    extract_names(full_names)
    df['Surname'] = surnames
    df['Title'] = titles
    return df, titles

# FEATURE: Fancy_title
def fea_fancy_title(df, titles):

    def fancy_title(title):
        not_fancy_title = ['Mr', 'Miss', 'Mrs', 'Master']
        if title not in not_fancy_title:
            return True
        else:
            return False

    fancy_titles = list(map(fancy_title, titles))
    df['Fancy_title'] = fancy_titles
    return df

# FEATURE: U15 and sibilings > 3
def fea_u15_many_silbings(df):

    def u15_many_siblings(row):
        if row['Age'] < 15 and row['SibSp'] > 3:
            return True
        else:
            return False

    df['U15_many_siblings'] = df.apply(u15_many_siblings, axis=1)
    return df

# FEATURE: is_alone
def fea_is_alone(df):

    def is_lone_traveller(row):
        if row['SibSp'] == 0 and row['Parch'] == 0:
            return True
        else:
            return False

    df['alone'] = df.apply(is_lone_traveller, axis=1)
    return df
