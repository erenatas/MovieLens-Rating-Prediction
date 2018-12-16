import Preprocessing, EDA, ModelSelectionCV, PCA

# Preprocessing.preprocess_data()
# EDA.explanatory_data_analysis()
pca_trn, pca_tst, y_train, y_test = PCA.pca()
ModelSelectionCV.model_selection_cv(pca_trn, pca_tst, y_train, y_test)
