import numpy as np

# Part 1 - Pegasos Algorithm

def pegasos(feature_matrix, labels, T=5, eta=0.1, lam=0.1):
    """Runs the pegasos algorithm on a given set of data.

    Arguments:
        feature_matrix(ndarray): A numpy matrix describing the given data.
            Each row represents a single data point.
        labels(ndarray): A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        T(int): An integer indicating how many times the perceptron algorithm
            should iterate through the feature matrix.
        eta(float): The learning rate.
        lam(float): The lambda value, which controls the amount of regularization.

    Returns:
        A tuple where the first element is a numpy array describing theta and the
        second element is the real number theta_0.
    """
    ft = np.c_[np.ones(feature_matrix.shape[0]), feature_matrix]
    row_count = ft.shape[0]
    theta = np.zeros(ft.shape[1])
    for t in range(T):
        for row_index in range(row_count):
            x_i = ft[row_index]
            y_i = labels[row_index]
            if y_i*np.dot(theta, x_i) <= 1:
                theta = theta - eta*(-y_i*x_i + lam*theta)
            else:
                theta = theta - eta*(lam*theta)
    return theta[1:], theta[0]
