# Preprocessing Part of the project.

# Import Libraries
import os
import pandas as pd
import dask.dataframe as dd


def preprocess_data():
    # print("Download the necessary data-set with cURL")
    # os.system("curl -O -X GET 'http://files.grouplens.org/datasets/movielens/ml-latest.zip' && unzip ml-latest.zip && rm "
    #         "ml-latest.zip")

    # Dask setting to enable seeing all columns on print
    pd.set_option('display.max_columns', 25)

    # Put Movies and Ratings to dataframe
    movies_dd = dd.read_csv('ml-latest/movies.csv')
    ratings_dd = dd.read_csv('ml-latest/ratings.csv')

    print("Merge Movie with Ratings")
    movie_dd = dd.merge(movies_dd, ratings_dd, how='inner', on='movieId').compute()

    print("Merged dataframe")
    print(movie_dd.head())

    print("Drop Ids which is unnecessary")
    movie_dd = movie_dd.drop(['userId', 'movieId'], axis=1)

    print("Remove years from titles")
    movie_dd['year'] = movie_dd['title'].apply(lambda x: x.strip()[-5:-1])
    movie_dd['title'] = movie_dd['title'].astype(str).str[:-7]  # Remove year from title.
    movie_dd['title_length'] = movie_dd['title'].apply(lambda x: len(x))

    #  Noise removal for the titles which do not have any year information.
    movie_dd['year'] = pd.to_numeric(movie_dd['year'],
                                     errors='coerce')
    movie_dd.dropna(inplace=True)

    print("Shape of the dataframe", movie_dd.shape)
    print(movie_dd.head())

    # Remove unnecessary variables so that memory can be freed.
    del movies_dd
    del ratings_dd

    print("Data and Feature Engineering")
    movie_dd['identity'] = 1
    lifespan_in_movielens = movie_dd.groupby('title').max().timestamp - movie_dd.groupby('title').min().timestamp
    identity_sum = movie_dd.groupby('title').sum().identity
    groupby_mean = movie_dd.groupby('title').mean()
    groupby_mean['rating_count'] = identity_sum
    groupby_mean = groupby_mean[groupby_mean['rating_count'] > 9] # To ensure that there are at least 10 voters.
    groupby_mean['lifespan_in_movielens'] = lifespan_in_movielens
    groupby_mean['genres'] = movie_dd.groupby('title').first().genres
    groupby_mean.reset_index(level=0, drop=True, inplace=True)
    groupby_mean.drop(['identity'], axis=1, inplace=True)
    groupby_mean.drop(['timestamp'], axis=1, inplace=True)

    # Remove the wrong data when year is 201
    print(groupby_mean.head())
    #groupby_mean.iloc[:, 1].replace(0.0, 0.5)
    groupby_mean = groupby_mean[groupby_mean.year > 1850]
    groupby_mean = groupby_mean[groupby_mean['year'].apply(lambda x: x.is_integer())]

    groupby_mean['Age'] = groupby_mean['year'].apply(lambda x: 2019 - x)
    groupby_mean['lifespan_in_movielens'] = groupby_mean['lifespan_in_movielens'].apply(lambda x: x / 60 / 60 / 24 / 365)

    groupby_mean.drop(['year'], axis=1, inplace=True)




    print("Movie dataframe with new features")
    print(movie_dd.head())

    del identity_sum
    del lifespan_in_movielens
    del movie_dd  #  We will use groupby_mean from now on.

    print("Hot Encoding of Genres which is a categorical attribute")
    groupby_mean['genres'] = groupby_mean['genres'].apply(lambda row: row.split('|'))
    groupby_mean['genre_count'] = groupby_mean['genres'].apply(lambda row: len(row))
    genre_columns = groupby_mean['genres'].apply(lambda x: pd.Series(1, index=x)).fillna(0)
    groupby_mean = pd.concat([groupby_mean, genre_columns], axis=1)
    groupby_mean = groupby_mean.drop(['genres'], axis=1)

    print("One Hot Encoded Movie dataframe")
    print(groupby_mean.head())

    print("Put dataframe into csv so that it won't be necessary to run this step once more")
    groupby_mean.to_csv('ml-latest/base.csv', index=False)
