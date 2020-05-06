import arff
import dill
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.datasets import load_breast_cancer
from statsmodels.stats.outliers_influence import variance_inflation_factor


def provide_cancer():
    cancer = load_breast_cancer()

    x = cancer.data
    y = cancer.target

    x_train = pd.DataFrame(x, columns=cancer.feature_names)
    y_train = pd.DataFrame(y)

    return x_train, y_train, {'y': dict(zip([0, 1], cancer.target_names))}


def provide_KDD():
    # make sure that data files are in the NSL-KDD subfolder:
    xl = []
    for row in arff.load('NSL-KDD/KDDTrain+.arff'):
        xl.append(row._data)
    x = pd.DataFrame.from_dict(xl)
    # remove numbered columns:
    for i in range(0, np.int(len(x.columns) / 2)):
        x.drop(i, axis=1, inplace=True)
    x.columns = [c.strip("'") for c in x.columns]
    mapping_indices = {}
    for c in x.columns:
        if x[c].dtype == 'object':
            cc = pd.Categorical(x[c])
            mapping_indices[c] = dict(zip(cc.codes, cc.categories))
            x[c] = cc.codes
    x_train, y_train = x.drop('class', axis=1), pd.DataFrame(list(x['class']))
    return x_train, y_train, mapping_indices


def calc_spearman(x_train):
    corr = spearmanr(x_train.astype('float64')).correlation
    corlst = []
    for i in range(0, corr.shape[0]):
        row = corr[i]
        for j in range(i + 1, row.shape[0]):
            corlst.append([x_train.columns[i], x_train.columns[j], corr[i, j], abs(corr[i, j])])
    cordf = pd.DataFrame(corlst, columns=['F1', 'F2', 'corr', 'abscorr'])
    cordf.sort_values('abscorr', inplace=True, ascending=False)
    return cordf


def calc_vif(x_train):
    '''
    https://stats.stackexchange.com/questions/155028/how-to-systematically-remove-collinear-variables-in-python#318668
    :param x_train: DataFrame
    :return: DataFrame
    '''
    cols = x_train.columns
    variables = np.arange(x_train.shape[1])
    c = x_train[cols[variables]].values
    vif = [variance_inflation_factor(c, ix) for ix in np.arange(c.shape[1])]
    r = pd.DataFrame(zip(cols, vif), columns=['feature', 'value']).sort_values(by='value', ascending=False)
    return r


def save_data(fname):
    dill.dump_session('%s.pkl' % fname)


def load_data_cancer():
    dill.load_session('UCI-BCW.pkl')


def load_data_KDD():
    dill.load_session('NSL-KDD.pkl')
