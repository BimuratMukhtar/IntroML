import numpy as np
from tqdm import trange
from scipy.stats.stats import pearsonr

#-------------------------------------------------------------------------------
# Part 1 - Naive Recommendation Algorithm
#-------------------------------------------------------------------------------

def predict_ratings_naive(Y):
    """Uses a naive algorithm for predicting movie ratings.

    If the user/movie rating is known, predict the known rating.
    Otherwise predict the average rating for that movie.
    If there are no ratings for the movie, predict the average rating
    of all movies.

    Arguments:
        Y(ndarray): A matrix of ratings where each row represents a
                    user and each column represents a movie.
                    -1 represents and unknown rating.

    Returns:
        A matrix with the predicted ratings for all users/movies.
    """

    number_of_users = Y.shape[0]
    number_of_movies = Y.shape[1]

    predictions = np.zeros(Y.shape)
    average_rating_all = np.average(Y[Y != -1])

    for user in range(number_of_users):
        movie_ratings = Y[user]
        average_rating_movie = np.average(movie_ratings[movie_ratings != -1])
        for movie in range(number_of_movies):
            rating = Y[user, movie]
            if rating != -1:
                predictions[user, movie] = rating
            elif average_rating_movie > 0:
                predictions[user, movie] = average_rating_movie
            else:
                predictions[user, movie] = average_rating_all
    return predictions


#-------------------------------------------------------------------------------
# Part 2 - Nearest-Neighbor Prediction
#-------------------------------------------------------------------------------

def predict_ratings_nearest_neighbor(user_to_movie_ratings, k=10):
    """Uses user similarity to make a nearest neighbor prediction.

    First computes the similarity (correlation) between users
    based on how users rate movies that both have seen.
    Then to make predictions for a user, the algorithm finds
    the k most similar users who have seen the movie and averages
    their ratings. However, rather than directly averaging their
    ratings, the algorithm instead uses the mean rating given by
    this user and adds the difference between the ratings for this
    movie and the mean ratings for each of the other users, weighted
    by similarity.

    Arguments:
        user_to_movie_ratings(ndarray): A matrix of ratings where each row represents a
                    user and each column represents a movie.
                    -1 represents and unknown rating.
        k(int): The number of nearest neighbors to use.

    Returns:
        A matrix with the predicted ratings for all users/movies.
    """
    number_of_users = user_to_movie_ratings.shape[0]
    number_of_movies = user_to_movie_ratings.shape[1]

    S = np.zeros((number_of_users, number_of_users))
    X = np.zeros((number_of_users, number_of_movies))

    def get_correlation(user_a: int, user_b: int) -> float:
        ar = user_to_movie_ratings[(user_a, user_b), :]
        ratings = ar[:, np.all(ar > -1, axis=0)]
        if len(ratings[0]) > 0:
            return get_pearson_correlation(ratings[0, :], ratings[1, :])
        else:
            return 0.0

    # t_ar = np.loadtxt("array.txt")
    t_ar = [[0]]
    if len(t_ar[0]) == number_of_users:
        S = t_ar
    else:
        for user_i in trange(number_of_users):
            for user_j in range(user_i+1, number_of_users):
                S[user_j, user_i] = S[user_i, user_j] = get_correlation(user_i, user_j)
        np.savetxt("array.txt", S)
    sorted_indexes = np.argsort(S)[:, ::-1]

    def get_k_nearest_user_indexes_rated_movie(nearest_user_indixes, k, movie):
        return nearest_user_indixes[user_to_movie_ratings[nearest_user_indixes, movie] != -1][:k]

    for user_a in trange(number_of_users):
        user_a_average_rating_all_movie = np.average(user_to_movie_ratings[user_a, user_to_movie_ratings[user_a] > -1])
        for movie in range(number_of_movies):
            knn = get_k_nearest_user_indexes_rated_movie(sorted_indexes[user_a], k, movie)
            sum_down = np.sum(np.take(S[user_a], knn, axis=0))
            sum_up = 0
            for user_b in knn:
                user_b_rts = user_to_movie_ratings[user_b]
                sum_up += S[user_a, user_b]*(user_b_rts[movie]
                                             - np.average(user_b_rts[user_b_rts > -1]))
            if sum_down != 0:
                X[user_a, movie] = user_a_average_rating_all_movie + sum_up/sum_down
            else:
                X[user_a, movie] = user_a_average_rating_all_movie
    return X


def get_pearson_correlation(x, y):
    mx = x.mean()
    my = y.mean()
    xm, ym = x - mx, y - my
    r_num = np.add.reduce(xm * ym)
    r_den = np.sqrt(np.sum(xm*xm) * np.sum(ym**2))
    if r_den == 0:
        return 0.0
    else:
        return r_num / r_den






#-------------------------------------------------------------------------------
# Part 3 - Matrix Factorization
#-------------------------------------------------------------------------------

# http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.173.2797&rep=rep1&type=pdf
def predict_ratings_matrix_factorization(R, nf=20, lam=0.05, T=20):
    """Uses low-rank matrix factorization to predict movie ratings.

    Arguments:
        R(ndarray): A matrix of ratings where each row represents a
                    user and each column represents a movie.
                    -1 represents and unknown rating. Equivalent to Y.
        nf(int): The number of features for each user/movie. Equivalent
                 to the rank k of the matrices U and M.
        lam(float): The lambda value.
        T(int): The number of times to perform the alternating minimization.

    Returns:
        A matrix with the predicted ratings for all users/movies.
    """

    raise NotImplementedError
