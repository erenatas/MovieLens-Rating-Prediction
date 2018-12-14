# Explanatory Data Analysis For the Project

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def explanatory_data_analysis():
    base_df = pd.read_csv('ml-latest/base.csv')

    plt.hist(base_df.year)
    plt.xlim(1700)
    plt.show()

    print(base_df.year.value_counts(normalize=True))

    plt.hist(base_df.rating)
    plt.title("Rating Histogram")
    plt.show()

    print(base_df.rating.value_counts(normalize=True).head(10))
    print(base_df.rating.describe())
    print("Worst Movies in the history: \n", base_df[base_df.rating == 0.500000])

    base_df_corr = base_df.corr()

    plt.figure(figsize=(20, 10))
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.title("Correlation Matrix")
    sns.heatmap(base_df_corr.corr(),
                annot=True,
                linewidths=.5,
                center=0,
                cbar=False,
                cmap="YlGnBu")

    plt.show()
