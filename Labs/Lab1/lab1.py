"""Utility functions for loading and plotting data."""

import csv
import numpy as np
import matplotlib.pyplot as plt

def load_reviews_data(reviews_data_path):
    """Loads the reviews dataset as a list of dictionaries.

    Arguments:
        reviews_data_path(str): Path to the reviews dataset .csv file.

    Returns:
        A list of dictionaries where each dictionary maps column name
        to value for a row in the reviews dataset.
    """
    result = []
    with open(reviews_data_path) as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        for row in reader:
            result.append(row)

    return result

def load_toy_data(toy_data_path):
    """Loads the 2D toy dataset as numpy arrays.

    Arguments:
        toy_data_path(str): Path to the toy dataset .csv file.

    Returns:
        A tuple (features, labels) in which features is an Nx2 numpy
        matrix and labels is a length-N vector of +1/-1 labels.
    """
    arrs = np.loadtxt(toy_data_path)
    data = arrs[:, 1:]
    labels = arrs[:, 0]
    return data, labels

def plot_toy_data(data, labels):
    """Plots the toy data in 2D.

    Arguments:
        data(ndarray): An Nx2 ndarray of points.
        labels(ndarray): A length-N vector of +1/-1 labels.
    """
    colors = ["red" if y == -1 else "blue" for y in labels]
    plt.scatter(data[:,0], data[:,1], c = colors)
    plt.show()

