"""
Building Classifier - TITANIC

Requirement:
    * Processed datasets -- train and test

1. build classifier using training data (with tuned parameters)
2. classify target dataset
3. save Kaggle submission file
"""

import pandas as pd
import numpy as np
from datetime import date
from sklearn.ensemble import RandomForestClassifier

params = {'max_features': 10, 'n_estimators': 16, 'max_depth': 4}


def main(df_trian, df_test):
    clf = clf_build(df_train)
    y_pred = clf_predict(df_test, clf)
    kaggle_sub_file(df_test, y_pred)


def clf_build(df):
    X = df.drop('Survived', axis=1).values
    y = df['Survived'].values
    clf = RandomForestClassifier(**params).fit(X, y)
    return clf


def clf_predict(df_test, clf):
    X = df_test.values
    y_pred = clf.predict(X)
    return y_pred


def kaggle_sub_file(df_test, y_pred):
    sub = pd.DataFrame(y_pred, index=df_test.index, columns=['Survived'])
    filename = 'Kaggle_sub_{}.csv'.format(date.today())
    sub.to_csv('./data/{}'.format(filename))
    print('WROTE to {}'.format(filename))


if __name__ == '__main__':
    df_train = pd.read_csv('./data/train-processed.csv').set_index('PassengerId')
    df_test = pd.read_csv('./data/test-processed.csv').set_index('PassengerId')
    print(df_test.isnull().sum())
    main(df_train, df_test)
    print('End of Script.')
