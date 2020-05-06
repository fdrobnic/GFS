import matplotlib.pyplot as plt
import numpy as np

import data


def plot_hist(Y_train, ime_izhoda=''):
    if len(Y_train) > 0:
        axh = Y_train.hist(bins=(Y_train[0].values.max() - Y_train[0].values.min() + 1).astype('int'))
        if ime_izhoda != '':
            axh[0][0].get_figure().savefig('%s/hist.png' % ime_izhoda)
        plt.show(block=False)
        plt.pause(0.1)
        return axh[0][0]


def show_spearman(X_train):
    SpearDF = data.calc_spearman(X_train)
    # mera multikolinearnosti podatkov (površina pod krivuljo):
    # nn = len(X_train.columns)
    # preverjeno, da je len(SpearDF['abscorr']) = (nn * (nn + 1) / 2) - nn  # zadnje je diagonala
    M = np.mean(SpearDF['abscorr'])
    axs = SpearDF.plot(y='abscorr', use_index=False, grid=True)
    axs.set_title('Multicollinearity of features (M = %s)' % M)
    axs.set_xlabel('No. of feature pairs')
    axs.set_ylabel('Spearman\'s rank correlation coefficient')
    return axs, M, SpearDF
