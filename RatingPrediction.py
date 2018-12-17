import Preprocessing, EDA, ModelSelectionCV, PCA, Training, IsolForest
import pandas as pd


print("-------------------------------------------------------------------------------------")

print("----------------------------- No outlier removal, No PCA-----------------------------")
Preprocessing.preprocess_data()
EDA.explanatory_data_analysis()
data_input = pd.read_csv('ml-latest/base.csv')
pca_trn, pca_tst, y_train, y_test = PCA.f_splitdata(data_input, 0.2)
ModelSelectionCV.model_selection_cv(pca_trn, pca_tst, y_train, y_test)
# Training.training(pca_trn, pca_tst, y_train, y_test)

print("-------------------------------------------------------------------------------------")

print("----------------------------- outlier removed, No PCA-----------------------------")
#Preprocessing.preprocess_data()
#EDA.explanatory_data_analysis()
data_input = pd.read_csv('ml-latest/base.csv')
data_input = IsolForest.isolation_forest(data_input)
pca_trn, pca_tst, y_train, y_test = PCA.f_splitdata(data_input, 0.2)
ModelSelectionCV.model_selection_cv(pca_trn, pca_tst, y_train, y_test)
# Training.training(pca_trn, pca_tst, y_train, y_test)

print("-------------------------------------------------------------------------------------")

print("----------------------------- outlier removed, PCA Applied-----------------------------")
#Preprocessing.preprocess_data()
EDA.explanatory_data_analysis()
pca_trn, pca_tst, y_train, y_test = PCA.pca()
ModelSelectionCV.model_selection_cv(pca_trn, pca_tst, y_train, y_test)
Training.training(pca_trn, pca_tst, y_train, y_test)