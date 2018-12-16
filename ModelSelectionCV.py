# Import dependencies and/or other modules
import numpy as np
import time
import warnings
import pandas as pd
# Import scikit-learn modules
from sklearn import linear_model, svm, model_selection
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost.sklearn import XGBRegressor
import scipy.stats as st
from sklearn.model_selection import RandomizedSearchCV




def model_selection_cv(pca_trn, pca_tst, y_train, y_test):
    #warnings.filterwarnings("ignore")

    # Load workspace variable from saved file
    #groupby_mean = pd.read_csv('ml-latest/base.csv')

    # -------------------------matrix of the numerical features-------------------------#
    ##import matplotlib.pyplot as plt
    ##cax = plt.matshow(np.cov(D_Arr[:,:-1].T))
    ##cax = plt.matshow(np.cov(D_Arr[:,0:15].T))
    ##plt.clim(-1,1)
    ##plt.colorbar(cax)
    ##plt.title('Covariance matrix of numerical features')
    ##plt.show()

    #rating = groupby_mean.rating
    #groupby_mean.drop(['rating'], axis=1, inplace=True)

    # Split the dataset in the ratio train:test = 0.9:0.1
    #X_train, X_test, y_train, y_test = model_selection.train_test_split(groupby_mean, rating, test_size=0.1,
    #                                                                    random_state=0)

    X_train, X_test= pca_trn, pca_tst

    # Create OLS linear regression object
    regrOLS = linear_model.LinearRegression()

    # Perform 5 fold cross-validation and store the MSE resulted from each fold
    scores = model_selection.cross_val_score(regrOLS, X_train, y_train, scoring='r2', cv=5)

    # Note: Due to a known issue in scikit-learn the results return are flipped in sign
    print('OLS: Least CV error: %.2f\n' % np.min(-scores))

    # ---------------- Cross validation for Ridge and Lasso ------------------------#
    # Range of hyper-parameters to choose for CV
    lambdas = [0.0001, 0.001, 0.01, 0.02, 0.05, 0.1, 1, 10]
    for l in lambdas:
        print('Lambda = %.5f' % l)
        # Start time for the 5-fold CV
        start = time.time()
        # Create ridge regression object
        knn_reg = linear_model.Ridge(alpha=l)
        scores = model_selection.cross_val_score(knn_reg, X_train, y_train, scoring='r2', cv=5)
        end = time.time()
        t = end - start
        print('Ridge: Least CV error: %.2f and time : %.3f' % (np.min(-scores), t))
        start = time.time()
        # Create lasso object
        regrLasso = linear_model.Lasso(alpha=l)
        scores = model_selection.cross_val_score(regrLasso, X_train, y_train, scoring='r2', cv=5)
        # Measure and compute time for the 5-fold CV
        end = time.time()
        t = end - start
        print('Lasso: Least CV error: %.2f and time : %.3f' % (np.min(-scores), t))
        print('\n')

    # -------------------- Cross validation for Elastic Net ------------------------#
    # Range of hyper-parameters to choose for CV
    l1Ratios = [0.1, 0.25, 0.5, 0.75, 0.9]
    for l in lambdas:
        print('Lambda = %.5f' % l)
        for l1R in l1Ratios:
            start = time.time()
            # Create elastic net object
            regrElasNet = linear_model.ElasticNet(alpha=l, l1_ratio=l1R)
            scores = model_selection.cross_val_score(regrElasNet, X_train, y_train, scoring='r2',
                                                     cv=5)
            end = time.time()
            t = end - start
            print('Elastic Net: l1Ratio = %.2f, Least CV error: %.2f and time : %.3f' % (l1R, np.min(-scores), t))
        print('\n')

    # ------------- Cross validation for Random Forest Regressor -------------------#
    # Range of hyper-parameters to choose for CV
    n_estimator = [1, 2, 5, 10, 20, 35, 50, 100, 200]
    maxFeatures = [0.25, 0.5, 0.75, 1]
    maxDepth = [3, 6, 8, 10, 15, 25]
    for n in n_estimator:
        for mf in maxFeatures:
            for d in maxDepth:
                print('Number of trees/estimators = %d, max depth = %d' % (n, d))
                start = time.time()
                # Create Random Forest Regressor object
                randFor = RandomForestRegressor(max_depth=d, random_state=0, n_estimators=n, max_features=mf)
                scores = model_selection.cross_val_score(randFor, X_train, y_train, scoring='r2',
                                                         cv=5)
                end = time.time()
                t = end - start
                print('Random forest regressor: %% of features = %.2f, Least CV error: %.2f and time : %.3f' % (
                    100 * mf, np.min(-scores), t))
            print('\n')

    # ------------- Cross validation for regressor using AdaBoost  -----------------#
    # Range of hyper-parameters to choose for CV
    n_estimator = [1, 2, 5, 10, 20, 35, 50, 100, 200]
    learning_rate = ['linear', 'square', 'exponential']
    maxDepth = [3, 6, 8, 10, 15, 25]
    for l in learning_rate:
        for n in n_estimator:
            for d in maxDepth:
                print('Number of trees/estimators = %d, max depth = %d' % (n, d))
                start = time.time()
                # Create Boosting Regressor object
                boosting = AdaBoostRegressor(DecisionTreeRegressor(max_depth=d), random_state=0, n_estimators=n, loss=l)
                scores = model_selection.cross_val_score(boosting, X_train, y_train, scoring='r2',
                                                         cv=5, n_jobs=1)
                end = time.time()
                t = end - start
                print('Regressor using AdaBoosting: loss type = %s, Least CV error: %.2f and time : %.3f' % (
                    l, np.min(-scores), t))
            print('\n')

    # ---------------- Cross validation for KNeighborsRegressor------------------------#
    # Range of hyper-parameters to choose for CV
    lambdas = [2, 3, 4, 5, 6, 7, 8, 9]
    for l in lambdas:
        print('Lambda = %.5f' % l)
        # Start time for the 5-fold CV
        start = time.time()
        # Create KNeighborsRegressor object
        knn_reg = KNeighborsRegressor(n_neighbors=l)
        scores = model_selection.cross_val_score(knn_reg, X_train, y_train, scoring='r2', cv=5)
        end = time.time()
        t = end - start
        print('KNeighborsRegressor: Least CV error: %.2f and time : %.3f' % (np.min(-scores), t))
        print('\n')

    # ------------- Cross validation for regressor using XGBoost  -----------------#
    # Range of hyper-parameters to choose for CV
    # n_estimator = [100, 1000, 10000]
    # learning_rate = [0.02, 0.05, 0.07, 0.1, 0.2, 0.5, 0.7, 1]
    # maxDepth = [3, 6, 8, 10, 15, 25]
    # gamma = [0,0.03,0.1,0.3]
    # colsample_bytree = [0.4,0.6,0.8]
    # reg_alpha = [1e-5, 1e-2,  0.75]
    # reg_lambda = [1e-5, 1e-2, 0.45]
    # subsample = [0.6,0.95]
    # min_child_weight = [1.5,6,10]
    #
    # for l in learning_rate:
    #     for n in n_estimator:
    #         for d in maxDepth:
    #             for g in gamma:
    #                 for c in colsample_bytree:
    #                     for alp in reg_alpha:
    #                         for lam in reg_lambda:
    #                             for s in subsample:
    #                                 for min_child in min_child_weight:
    #                                     print('Number of trees/estimators = %d, max depth = %d' % (n, d))
    #                                     start = time.time()
    #                                     # Create Boosting Regressor object
    #                                     xgb_model = xgboost.XGBRegressor(colsample_bytree=c,
    #                                                          gamma=g,
    #                                                          learning_rate=l,
    #                                                          max_depth=d,
    #                                                          min_child_weight=min_child,
    #                                                          n_estimators=n,
    #                                                          reg_alpha=alp,
    #                                                          reg_lambda=lam,
    #                                                          subsample=s,
    #                                                          seed=42)
    #                                     scores = model_selection.cross_val_score(xgb_model, X_train, y_train, scoring='r2',
    #                                                                              cv=5, n_jobs=1)
    #                                     end = time.time()
    #                                     t = end - start
    #                                     print("Regressor using XGboosting: learning rate = %s, Least CV error: %.2f, "
    #                                           "gamma =.3f, min_child_weight =.4f, n_estimators = .5f, reg_alpha = "
    #                                           ".6f, reg_lambda = .7f, subsample= .8f, max_depth = %.9f  and time : %.10f"
    #                                           % (l, np.min(-scores), g, min_child, n, alp, lam, s, d, t))
    one_to_left = st.beta(10, 1)
    from_zero_positive = st.expon(0, 50)

    params = {
        "n_estimators": st.randint(100, 1000, 10000),
        "max_depth": st.randint(3, 40),
        "learning_rate": st.uniform(0.05, 0.4),
        "colsample_bytree": one_to_left,
        "subsample": one_to_left,
        "gamma": st.uniform(0, 10),
        'reg_alpha': from_zero_positive,
        "min_child_weight": from_zero_positive,
    }

    xgbreg = XGBRegressor(nthreads=-1)

    gs = RandomizedSearchCV(xgbreg, params, n_jobs=1)
    gs.fit(X_train, y_train)

    print("Regressor using XGboosting: ", "\nBest Index: ", gs.best_index_, "\nBest estimator: ", gs.best_estimator_,
          "\nBest Params: ", gs.best_params_)
    print('\n')

    # ------------ Cross validation for Support Vector Regressor -------------------#
    # Range of hyper-parameters to choose for CV
    C = [0.01, 0.1, 1, 10, 20, 50]
    eps = [0.0005, 0.001, 0.01, 0.05, 0.1, 0.5, 1, 10, 100]
    for c in C:
        print('C = %.5f' % c)
        for e in eps:
            start = time.time()
            # Create SVR object
            svr = svm.SVR(C=c, epsilon=e)
            scores = model_selection.cross_val_score(svr, X_train, y_train, scoring='r2', cv=5)
            end = time.time()
            t = end - start
            print('SVR: eps = %.4f, Least CV error: %.2f and time : %.3f' % (e, np.min(-scores), t))
        print('\n')