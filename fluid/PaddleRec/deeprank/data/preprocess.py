import sys
import pickle
import pandas
import numpy as np


def read_to_pandas(file_path, sep):
    df = pandas.read_csv(file_path, sep=sep, header=0, engine="python")
    return df


def build_map(df, col_name):
    key = sorted(df[col_name].unique().tolist())
    m = dict(zip(key, range(len(key))))
    df[col_name] = df[col_name].map(lambda x: m[x])
    return m, key


def preprocess(rating_path, movie_path, sep=","):
    print "reading ratings"
    ratings = read_to_pandas(rating_path, sep)
    ratings = ratings[['userId', 'movieId', 'rating', 'timestamp']]
    print ratings[:4]

    print "reading movies"
    movies = read_to_pandas(movie_path, sep)
    print movies[:4]

    movies = movies[movies['movieId'].isin(ratings['movieId'].unique())]
    movies = movies.reset_index(drop=True)
    movies = movies[['movieId', 'genres']]
    #temporarily consider the first category, actually should use all the categories
    movies['genres'] = movies['genres'].map(lambda x: x.split('|')[0])

    movie_map, movie_key = build_map(movies, 'movieId')
    cate_map, cate_key = build_map(movies, 'genres')
    user_map, user_key = build_map(ratings, 'userId')

    user_count = len(user_map)
    movie_count = len(movie_map)
    cate_count = len(cate_map)
    example_count = ratings.shape[0]
    print "user count: %d, movie count: %d, cate count: %d, example count:%d" % (
        user_count, movie_count, cate_count, example_count)

    movies = movies.sort_values('movieId')
    movies = movies.reset_index(drop=True)
    ratings['movieId'] = ratings['movieId'].map(lambda x: movie_map[int(x)])
    #ratings = ratings[(ratings['rating']>=4) | (ratings['rating']<=2)]
    ratings['rating'] = ratings['rating'].map(lambda x: 1 if x >= 4 else 0)
    ratings = ratings.sort_values(['userId', 'timestamp'])
    ratings = ratings.reset_index(drop=True)

    cate_list = [movies['genres'][i] for i in range(len(movie_map))]
    cate_list = np.array(cate_list, dtype=np.int32)

    with open('./remap' + sys.argv[1] + '.pkl', 'wb') as f:
        pickle.dump(ratings, f,
                    pickle.HIGHEST_PROTOCOL)  #uid, mid, rating, time
        pickle.dump(cate_list, f, pickle.HIGHEST_PROTOCOL)  #cateid
        pickle.dump((user_count, movie_count, cate_count, example_count), f,
                    pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    data_number = int(sys.argv[1])
    file_path = "ml-" + str(data_number) + "m/"
    if data_number == 1:
        preprocess(file_path + "ratings.dat", file_path + "movies.dat", "::")
    elif data_number == 20:
        preprocess(file_path + "ratings.csv", file_path + "movies.csv")
    elif data_number == 0:
        preprocess(file_path + "ratings.csv", file_path + "movies.csv")

#1 movieId,title,genres
#1 userId,movieId,rating,timestamp
