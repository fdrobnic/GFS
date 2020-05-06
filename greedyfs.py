import datetime
import time

from numpy import mean
from pandas import DataFrame
from sklearn.base import clone


def adjust_tuned_par(x, tp):
    n = []
    tp2 = {}
    for p in tp:
        cm = 0
        if p == 'max_features':
            for c in tp[p]:
                if c <= len(x.columns):
                    n.append(c)
                else:
                    if c > cm:
                        cm = c
            if cm > max(n):
                n.append(len(x.columns))
            tp2[p] = n
        else:
            tp2[p] = tp[p]
    return tp2


def gfs_step_one_f(xc, est, x_train, y_train, tuned_parameters, c, bp, u, e):
    # evaluate one feature
    xn = xc.copy()
    xn[c] = x_train[c]
    est2 = clone(est)
    est2.param_grid = adjust_tuned_par(xn, tuned_parameters)
    est2.fit(xn, y_train[0])
    pred2 = est2.best_estimator_.predict(xn)
    acc2 = mean(pred2 == y_train[0])
    bp[c] = est2.best_params_
    u[c] = acc2
    e[c] = (pred2 == y_train[0])


def gfs_step(xc, est, x_train, y_train, tuned_parameters, callback=None, tie_min_trees=True):
    # find one additional best feature
    bp = {}
    u = {}
    e = {}
    s = len(xc.columns)
    i = 0
    p = len(x_train.columns)
    for c in x_train.columns:
        i += 1
        if c not in xc.columns:
            gfs_step_one_f(xc, est, x_train, y_train, tuned_parameters, c, bp, u, e)
            time.sleep(0.1)
            if callback:
                callback(s, i, p)

    m = max(u.values())
    if len([c for c in u.keys() if u[c] == m]) == 1 or not tie_min_trees:
        c_max = max(u, key=u.get)
    else:
        v = {}
        for c in u.keys():
            if u[c] == m:
                v[c] = bp[c]['n_estimators']
        c_max = min(v, key=v.get)
    if c_max != '':
        xc[c_max] = x_train[c_max]

    return xc, c_max, bp, m, u, e


def greedy_feature_selection(est, x_train, y_train, margin, tuned_parameters, callback=None, tie_min_trees=True):
    """
    Main algorithm of GFS

    :param est:
        trained estimator, instance of RandomForestClassifier
    :param x_train:
        input training dataset: DataFrame
    :param y_train:
        input class labels: DataFrame
    :param margin:
        lower limit as stop criterion for training: float
    :param tuned_parameters:
        dict of hyperparameters to be tuned during training: dict
    :param callback:
        progressbar callback function with three parameters: step, feature number, max features
    :param tie_min_trees:
        whether the least-trees-used criterion is applied: boolean
    :return:
        reduced dataset: DataFrame,
        algorithm details: list
    """
    xc = DataFrame()
    ld = []
    u = {}
    rez_stari = 0.0
    while (ld == [] or (rez_stari <= max(u.values()) < margin)) and len(xc.columns) < len(x_train.columns):
        if ld:
            rez_stari = max(u.values())

        xc, c_max, bp, m, u, e = gfs_step(xc, est, x_train, y_train, tuned_parameters, callback, tie_min_trees)

        ld.append([c_max, bp, m, u, e])

    return xc, ld  # output dataset and algorithm details


if __name__ == '__main__':
    pass
