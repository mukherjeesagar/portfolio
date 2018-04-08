"""
Script for pre-process of the titanic data
25/03/2018

TODO: fill test set age using training set data
"""

import numpy as np
import pandas as pd
import re
import feature_engineering as fea_eng
import feature_extraction as fea_ext

def main(paths):
    df_train = pd.read_csv(paths[0])
    df_test = pd.read_csv(paths[1])
    print('Read {}'.format(paths))
    df_train = df_train.set_index('PassengerId')
    df_test = df_test.set_index('PassengerId')
    df_train = fea_eng.feature_engineering(df_train)
    df_test = fea_eng.feature_engineering(df_test)
    df_train = fea_ext.feature_extraction(df_train)
    df_test = fea_ext.feature_extraction(df_test)
    df_test.Fare = df_test.Fare.fillna(df_test.Fare.median())

    df_train, df_test = fea_ext.process_age(df_train, df_test)
    drop_cols = ['Name', 'Ticket', 'Cabin', 'Age',
                 'Sex', 'Embarked', 'Title', 'Surname']
    df_train = df_train.drop(drop_cols, axis=1)
    df_test = df_test.drop(drop_cols, axis=1)

    # Save file, added "-processed" as suffix
    df_train.to_csv('./data/train-processed.csv')
    df_test.to_csv('./data/test-processed.csv')

def process_data(paths):
    for path in paths:
        main(path)


if __name__ == '__main__':
    paths = ['./data/train.csv', './data/test.csv']
    main(paths)
    print('End of Script.')

