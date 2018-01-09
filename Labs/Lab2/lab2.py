from string import punctuation, digits

import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics


### Part 1 - Perceptron Algorithm

def perceptron_single_step_update(feature_vector, label, current_theta, current_theta_0):
    """Updates theta and theta_0 for a single step of the perceptron algorithm.

    Arguments:
        feature_vector(ndarray): A numpy array describing a single data point.
        label(int): The correct classification of the data point.
        current_theta(ndarray): The current theta being used by the perceptron
            algorithm before this update.
        current_theta_0(float): The current theta_0 being used by the perceptron
            algorithm before this update.

    Returns:
        A tuple where the first element is a numpy array with the value of
        theta after the current update has completed and the second element is a
        real valued number with the value of theta_0 after the current updated has
        completed.
    """
    current_theta = current_theta + label*feature_vector
    current_theta_0 = current_theta_0 + label
    return current_theta, current_theta_0

def perceptron(feature_matrix, labels, T = 5):
    """Runs the average perceptron algorithm on a given set of data.

    Arguments:
        feature_matrix(ndarray): A numpy matrix describing the given data.
            Each row represents a single data point.
        labels(ndarray): A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        T(int): An integer indicating how many times the perceptron algorithm
            should iterate through the feature matrix.

    Returns:
        A tuple where the first element is a numpy array with the value of
        the average theta and the second element is a real number with the
        value of the average theta_0.
    """
    current_theta = np.zeros(feature_matrix.shape[1])
    current_theta_o = 0
    theta_sum = np.zeros(feature_matrix.shape[1])
    theta_0_sum = 0
    number_rows = feature_matrix.shape[0]
    for t in range(T):
        for row_index in range(0, number_rows):
            x_i = feature_matrix[row_index]
            y_i = labels[row_index]
            if y_i*(np.dot(current_theta, x_i) + current_theta_o) <= 0:
                current_theta, current_theta_o = perceptron_single_step_update(x_i, y_i, current_theta, current_theta_o)
            theta_sum = theta_sum + current_theta
            theta_0_sum = theta_0_sum + current_theta_o

    return theta_sum/(number_rows*T), theta_0_sum/(number_rows*T)

### Part 2 - Classifying Reviews

def classify(feature_matrix, theta, theta_0=0):
    """Classifies a set of data points using theta and theta_0.

    Arguments:
        feature_matrix(ndarray): A numpy matrix describing the given data.
            Each row represents a single data point.
        theta(ndarray): A numpy array describing the linear classifier.
        theta_0(float): A real valued number representing the offset parameter.

    Returns:
        A numpy array of 1s and -1s where the kth element of the array is the predicted
        classification of the kth row of the feature matrix using the given theta
        and theta_0.
    """
    ft = np.c_[feature_matrix, np.ones(feature_matrix.shape[0])]
    my_theta = np.r_[theta, theta_0]
    mul = np.matmul(ft, my_theta)
    sign = np.sign(mul)
    sign[sign == 0] = -1
    return sign


def accuracy(feature_matrix, labels, theta, theta_0=0):
    """Determines the accuracy of a linear classifier.

    Arguments:
        feature_matrix(ndarray): A numpy matrix describing the data.
            Each row represents a single data point.
        labels(ndarray): A numpy array where the kth element of the array
            is the correct classification of the kth row of the feature matrix.
        theta(ndarray): A numpy array describing the linear classifier.
        theta_0(float): A real valued number representing the offset parameter.

    Returns:
        The accuracy of the model on the provided data.
    """

    return metrics.accuracy_score(labels, classify(feature_matrix, theta, theta_0))

### Part 3 - Improving the Model

def tune(Ts, train_feature_matrix, train_labels, val_feature_matrix, val_labels):
    """Runs perceptron with each of the T values and returns the results.

    Arguments:
        Ts(list): A list of the T values to try.
        train_feature_matrix(ndarray): A numpy matrix describing the training data.
            Each row represents a single data point.
        train_labels(ndarray): A numpy array where the kth element of the array
            is the correct classification of the kth row of the training
            feature matrix.
        val_feature_matrix(ndarray): A numpy matrix describing the validation data.
            Each row represents a single data point.
        val_labels(ndarray): A numpy array where the kth element of the array
            is the correct classification of the kth row of the validation
            feature matrix.

    Returns:
        A tuple where the first element is an ndarray of train accuracies for each
        parameter value and the second element is an ndarray of validation accuracies
        for each parameter value.
    """
    tr = []
    vl = []
    for t in Ts:
        theta, theta_0 = perceptron(train_feature_matrix, train_labels, t)
        tr.append(accuracy(train_feature_matrix, train_labels, theta, theta_0))
        vl.append(accuracy(val_feature_matrix, val_labels, theta, theta_0))
    return np.array(tr), np.array(vl)

def extract_words(input_string):
    """Returns a list of lowercase words in the string.

    Also separates punctuation and digits with spaces.

    Arguments:
        input_string(str): A string.

    Returns:
        A list of words with punctuation and digits separated.
    """

    for c in punctuation + digits:
        input_string = input_string.replace(c, ' ' + c + ' ')

    return input_string.lower().split()

def bag_of_words(reviews):
    """Creates a bag-of-words representation of text.

    Arguments:
        reviews(list): A list of strings.

    Returns:
        A dictionary which maps each word in the text
        to a unique index.
    """

    dictionary = {}

    for text in reviews:
        word_list = extract_words(text)

        for word in word_list:
            if word not in dictionary:
                dictionary[word] = len(dictionary)

    return dictionary

def extract_bow_feature_vectors(reviews, dictionary):
    """Extracts bag-of-words features from text as a feature matrix.

    Arguments:
        reviews(list): A list of strings with one string for
            each review (data point).
        dictionary(dict): A map from words to unique indices.

    Returns:
        An ndarray with a row for each review with a vector
        with a bag-of-words representation of the text review.
        This matrix is of shape (n,m) where n is the number of
        reviews and m is the total number of entries in the
        dictionary. The bag-of-words vector has a 1 if that
        word appears in the review and a 0 otherwise.
    """

    feature_matrix = np.zeros([len(reviews), len(dictionary)])

    for i, text in enumerate(reviews):
        word_list = extract_words(text)

        for word in word_list:
            if word in dictionary:
                feature_matrix[i, dictionary[word]] = 1

    return feature_matrix

def extract_final_features(reviews, dictionary):
    """Extracts features from the reviews.

    YOU MAY MODIFY THE PARAMETERS OF THIS FUNCTION.

    Arguments:
        reviews(list): A list of strings with one string for
            each review (data point).
        dictionary(dict): A map from words to unique indices.

    Returns:
        An ndarray of shape (n,m) where n is the number of reviews
        and m is the number of total features.
    """

    return extract_bow_feature_vectors(reviews, dictionary)
