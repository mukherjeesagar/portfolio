"""
Script for pre-process of the titanic data
25/03/2018
"""

import numpy as np
import pandas as pd
import re
import feature_engineering as fea_eng
import feature_extraction as fea_ext

def main(path):
    df = pd.read_csv(path)
    print('Read {}'.format(path))
    df = df.set_index('PassengerId')
    df = fea_eng.feature_engineering(df)
    df = fea_ext.feature_extraction(df)
    df = df.drop(['Name', 'Ticket', 'Cabin'], axis=1)

    # Save file, added "-c" as suffix
    filename = path[:-4] + '-c.csv'
    df.to_csv(filename)

def process_data(paths):
    for path in paths:
        main(path)


if __name__ == '__main__':
    paths = ['./data/train.csv', './data/test.csv']
    
