# Number of trees/estimators = 1, max depth = 25
# Random forest regressor: % of features = 50.00, Least CV error: 0.57 and time : 0.414

# Number of trees/estimators = 1, max depth = 25
# # Regressor using AdaBoosting: loss type = linear, Least CV error: 0.50 and time : 1.101

# Lambda = 2.00000
# KNeighborsRegressor: Least CV error: 0.22 and time : 0.161

from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from math import sqrt


# Number of trees/estimators = 1, max depth = 25
# Random forest regressor: % of features = 50.00, Least CV error: 0.57 and time : 0.414

# Number of trees/estimators = 1, max depth = 25
# # Regressor using AdaBoosting: loss type = linear, Least CV error: 0.50 and time : 1.101

# Lambda = 2.00000
# KNeighborsRegressor: Least CV error: 0.22 and time : 0.161

from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from math import sqrt


def training(pca_trn, pca_tst, y_train, y_test):
    forest = RandomForestRegressor(max_depth=25, random_state=0, n_estimators=1, max_features=6)
    forest.fit(pca_trn, y_train)
    forest_predict = forest.predict(pca_tst)

    ada = AdaBoostRegressor(DecisionTreeRegressor(max_depth=25), random_state=0, n_estimators=1, loss='linear')
    ada.fit(pca_trn, y_train)
    ada_predict = ada.predict(pca_tst)

    knn = KNeighborsRegressor(n_neighbors=2)
    knn.fit(pca_trn, y_train)
    knn_predict = knn.predict(pca_tst)


    # corrcoef_results = []
    # corrcoef_results.append(metrics.matthews_corrcoef(y_test, forest_predict))
    # corrcoef_results.append(metrics.matthews_corrcoef(y_test, ada_predict))
    # corrcoef_results.append(metrics.matthews_corrcoef(y_test, knn_predict))

    columns = ('Forest', 'Ada', 'KNN')
    y_pos = np.arange(len(columns))
    
    
    mae_results = []
    mae_results.append(metrics.mean_absolute_error(y_test, forest_predict))
    mae_results.append(metrics.mean_absolute_error(y_test, ada_predict))
    mae_results.append(metrics.mean_absolute_error(y_test, knn_predict))

    medae_results = []
    medae_results.append(metrics.median_absolute_error(y_test, forest_predict))
    medae_results.append(metrics.median_absolute_error(y_test, ada_predict))
    medae_results.append(metrics.median_absolute_error(y_test, knn_predict))

    rmse_results = []
    rmse_results.append(sqrt(metrics.mean_squared_error(y_test, forest_predict)))
    rmse_results.append(sqrt(metrics.mean_squared_error(y_test, ada_predict)))
    rmse_results.append(sqrt(metrics.mean_squared_error(y_test, knn_predict)))

    msle_results = []
    msle_results.append(metrics.mean_squared_log_error(y_test, forest_predict))
    msle_results.append(metrics.mean_squared_log_error(y_test, ada_predict))
    msle_results.append(metrics.mean_squared_log_error(y_test, knn_predict))
    
    print("For Forest, Ada, KNN\n", "mae, medae, rmse, msle scores: \n", mae_results, '\n', medae_results, '\n', rmse_results, '\n', msle_results)

    plt.bar(y_pos, mae_results, align='center', alpha=0.5)
    plt.xticks(y_pos, columns)
    plt.ylabel('Mean Absolute Error')
    plt.title('Mean Absolute Error Plot')
    plt.show()
    
    plt.bar(y_pos, medae_results, align='center', alpha=0.5)
    plt.xticks(y_pos, columns)
    plt.ylabel('Median Absolute Error')
    plt.title('Median Absolute Error Plot')
    plt.show()
    
    plt.bar(y_pos, rmse_results, align='center', alpha=0.5)
    plt.xticks(y_pos, columns)
    plt.ylabel('Root Mean Square Error')
    plt.title('Root Mean Square Error Plot')
    plt.show()
    
    plt.bar(y_pos, msle_results, align='center', alpha=0.5)
    plt.xticks(y_pos, columns)
    plt.ylabel('Mean Squared Log Error')
    plt.title('Mean Squared Log Error Plot')
    plt.show()
