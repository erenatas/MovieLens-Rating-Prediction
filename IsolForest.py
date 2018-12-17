from sklearn.ensemble import IsolationForest


def isolation_forest(groupby_mean):
  # For outliers
  clf = IsolationForest(max_samples = 100, random_state = 42)
  clf.fit(groupby_mean)
  y_noano = clf.predict(groupby_mean)
  y_noano = pd.DataFrame(y_noano, columns = ['Top'])
  y_noano[y_noano['Top'] == 1].index.values

  groupby_mean = groupby_mean.iloc[y_noano[y_noano['Top'] == 1].index.values]
  groupby_mean.reset_index(drop = True, inplace = True)
  print("Number of Outliers:", y_noano[y_noano['Top'] == -1].shape[0])
  print("Number of rows without outliers:", groupby_mean.shape[0])
  return groupby_mean