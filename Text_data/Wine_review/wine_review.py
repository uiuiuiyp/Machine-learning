"""
wine_review.py:
    Generate tf-idf features from descriptions of different wines
    and predict the review score based on it.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import operator
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

import xgboost as xgb

NROWS = 50000
NFOLDS = 5
SEED = 42

def plot_distribution(df, dfcol, bins=20):
    plt.figure()
    plt.hist(df[dfcol], bins=bins)
    plt.title(dfcol + ' distribution')
    plt.savefig(dfcol + '_distribution.png')

def build_model_input():
    
    data = pd.read_csv('../input/winemag-data-130k-v2.csv', index_col=0, nrows=NROWS)
    print('Input rows', len(data), '\n')

    print('Explore data distribution')
    data['desc_len'] = data.description.apply(len)
    plot_distribution(data, 'desc_len', bins=40)
    plot_distribution(data, 'points', bins=20)

    print('Remove duplicate entries')
    data.drop_duplicates(subset='description', inplace=True)
    print('Input rows', len(data), '\n')

    print('Generate tf-idf features')
    start = time.time()
    n = 3
    tfv = TfidfVectorizer(min_df=3,                # minimum document frequency of the word
                          max_features=None,
                          analyzer='word',
                          token_pattern=r'\w{1,}', # alphanumeric character + 1 or more repetition
                          ngram_range=(1,n),       # make vocabulary with 1 to 3 words ngram
                          use_idf=1,               # use inverse document frequency
                          smooth_idf=1, 
                          sublinear_tf=1,
                          stop_words = 'english'
                         )

    x_train = tfv.fit_transform(data.description)

    end = time.time()
    print('Time to fit and transform text: {:.2f}'.format(end - start))

    print('Generate index to word dictionary')
    sorted_dict = sorted(tfv.vocabulary_.items(), key=operator.itemgetter(1))
    indices_word = list(list(zip(*sorted_dict))[0])

    y_train = data.points.values
    y_train = (y_train - y_train.mean()) / y_train.std()

    return x_train, y_train, indices_word

class XgbWrapper(object):
    def __init__(self, seed=0, params=None):
        self.param = params
        self.param['seed'] = seed
        self.nrounds = params.pop('nrounds', 250)

    def train(self, x_train, y_train):
        dtrain = xgb.DMatrix(x_train, label=y_train)
        self.gbdt = xgb.train(self.param, dtrain, self.nrounds)

    def predict(self, x):
        return self.gbdt.predict(xgb.DMatrix(x))

    def get_score(self, importance_type='weight'):
        return self.gbdt.get_score(importance_type=importance_type)

def get_oof(clf, x_train, y_train, folds):

    ntrain = x_train.shape[0]
    oof_train = np.zeros((ntrain,))    

    for i, (train_index, test_index) in enumerate(folds.split(x_train)):
        print("Fold", (i + 1))
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        clf.train(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)

    return oof_train.reshape(-1, 1)

def train_model(x_train, y_train, folds):

    weights = None

    xgb_params = {
        'seed': 0,
        'colsample_bytree': 0.7,
        'silent': 1,
        'subsample': 0.7,
        'learning_rate': 0.075,
        'objective': 'reg:linear',
        'max_depth': 7,
        'num_parallel_tree': 1,
        'min_child_weight': 1,
        'eval_metric': 'rmse',
        'nrounds': 350
    }

    print('Training model')
    xg = XgbWrapper(seed=SEED, params=xgb_params)

    start = time.time()
    xg_oof_train = get_oof(xg, x_train, y_train, folds)
    end = time.time()
    print('Time to fit features: {:.2f}'.format(end - start))

    xg_all = XgbWrapper(seed=SEED, params=xgb_params)
    xg_all.train(x_train, y_train)

    print("XG-CV: {}".format(2.92 * mean_squared_error(y_train, xg_oof_train)))

    return xg_all

def display_importance(xg=None, idx2word=[]):
    
    features = xg.get_score(importance_type='gain')
    importances = np.zeros(len(idx2word))
    for i in range(len(idx2word)):
        key = 'f' + str(i)
        val = features.get(key, 0)
        importances[i] = val

    coef = pd.Series(importances, index=idx2word)
    
    num_coef = 30
    imp_coef = coef.sort_values().tail(num_coef)

    plt.figure(figsize=(12,8))
    imp_coef.plot(kind='barh')
    plt.title('Feature importance')
    plt.xlabel('Average gain per split')

    plt.savefig('feature_importances.png')

if __name__ == '__main__':
    
    # Build model inputs
    x_train, y_train, idx2word = build_model_input()
    # Generate KFolds
    folds = KFold(n_splits=NFOLDS, shuffle=True, random_state=SEED)
    # Train model
    xg_all = train_model(x_train, y_train, folds)
    # Display graphs
    display_importance(xg=xg_all, idx2word=idx2word)

    # plt.show()