from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def f_apply_pca(trn, tst, k=None):
    pca = PCA(n_components=None, svd_solver='full')
    pca_trn = pca.fit_transform(trn)
    pca_tst = pca.transform(tst)
    explain = pca.explained_variance_
    var_rate = np.cumsum(np.round(explain, decimals=3) * 100)

    plt.ylabel('% Variance Explained')
    plt.xlabel('# of Features')
    plt.title('PCA Analysis')
    plt.style.context('seaborn-whitegrid')

    print("Variance Rate: ", var_rate)
    plt.plot(var_rate)
    plt.show()

    pca = PCA(n_components=k, svd_solver='full')
    pca_trn = pca.fit_transform(trn)
    pca_tst = pca.transform(tst)
    explain = pca.explained_variance_
    var_rate = np.cumsum(np.round(explain, decimals=3) * 100)

    feat_amnt = len(explain)
    return pca_trn, pca_tst, explain, var_rate, feat_amnt


def f_normalize(train, test):
    # Normalization applied on both sets
    feature_norm = MinMaxScaler()
    n_train = feature_norm.fit_transform(train)
    n_test = feature_norm.transform(test)
    return n_train, n_test


def f_splitdata(data, test_split_ratio):
    df = data
    features = df.iloc[:, 1:]
    label = df.iloc[:, 0]
    # split into 2 parts.
    x_train, x_test, y_train, y_test = train_test_split(features, label, test_size=test_split_ratio)
    return x_train, x_test, y_train, y_test


def pca():
    data_input = pd.read_csv('ml-latest/base.csv')
    x_train, x_test, y_train, y_test = f_splitdata(data_input, 0.2)
    n_train, n_test = f_normalize(x_train, x_test)
    pca_trn, pca_tst, explain, var_rate, feat_amnt = f_apply_pca(n_train, n_test, k=6)
    return pca_trn, pca_tst, y_train, y_test
