import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier
from tqdm import trange

import lab4
import utils

#-------------------------------------------------------------------------------
# Data Generation
#-------------------------------------------------------------------------------

np.random.seed(0)

# Generate 200 random data points
X = np.random.randn(200, 2)

# Assign labels based on different functionsgst
y_linear = np.array([1 if x2 > 2 * x1 else -1 for x1, x2 in X])
y_quad = np.array([1 if x2 > 2 * x1**2 - 1 else -1 for x1, x2 in X])
y_radial = np.array([1 if x1**2 + x2**2 < 1 else -1 for x1, x2 in X])
y_angular = np.array([1 if (np.arctan2(x[0], x[1]) > np.pi / 6 and np.arctan2(x[0], x[1]) < np.pi / 2) else -1 for x in X])

# Plot the labelled data
ax1 = plt.subplot(221)
utils.plot_data(ax1, X, y_linear, title='Linear')
ax2 = plt.subplot(222)
utils.plot_data(ax2, X, y_quad, title='Quadratic')
ax3 = plt.subplot(223)
utils.plot_data(ax3, X, y_radial, title='Radial')
ax4 = plt.subplot(224)
utils.plot_data(ax4, X, y_angular, title='Angular')
plt.show()

#-------------------------------------------------------------------------------
# Part 2 - Linear Models
#-------------------------------------------------------------------------------

# def linear_kernel(x_j, x_i):
#     raise NotImplementedError

# clf = lab4.KernelPerceptron(kernel=linear_kernel)
# clf.fit(X, y_linear)

# ax1 = plt.subplot(121)
# utils.plot_classifier(ax1, X, y_linear, clf, title='Linear Perceptron')



# clf = svm.SVC(kernel='linear')
# clf.fit(X, y_linear)

# ax2 = plt.subplot(122)
# utils.plot_classifier(ax2, X, y_linear, clf, title='Linear SVM')

# plt.show()

#-------------------------------------------------------------------------------
# Part 3 - Quadratic Models
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Part 3.1 - Quadratic Data Transformation
#-------------------------------------------------------------------------------

# def phi_quad(x):
#     raise NotImplementedError

# X_quad = np.array([phi_quad(x) for x in X])

# clf = svm.SVC(kernel='linear')
# clf.fit(X_quad, y_quad)


# ax1 = plt.subplot(221)
# utils.plot_data(ax1, X, y_quad, title='Original data')
# ax3 = plt.subplot(223)
# utils.plot_classifier(ax3, X_quad, y_quad, clf, title='Linear classifier on transformed data')
# ax4 = plt.subplot(224)
# utils.plot_classifier(ax4, X, y_quad, clf, phi=phi_quad, title='Projection of classifier onto original data')
# plt.show()

#-------------------------------------------------------------------------------
# Part 3.2 - Quadratic Kernel
#-------------------------------------------------------------------------------

# def quad_kernel(x_j, x_i):
#     raise NotImplementedError

# clf = lab4.KernelPerceptron(kernel=quad_kernel)
# clf.fit(X, y_quad)

# ax = plt.gca()
# utils.plot_classifier(ax, X, y_quad, clf, title='Quadratic Perceptron')
# plt.show()

#-------------------------------------------------------------------------------
# Part 4 - Radial Models
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Part 4.1 - Radial Data Transformation
#-------------------------------------------------------------------------------

# def phi_radial(x):
#     raise NotImplementedError

# X_radial = np.array([phi_radial(x) for x in X])

# clf = svm.SVC(kernel='linear')
# clf.fit(X_radial, y_radial)

# ax1 = plt.subplot(221)
# utils.plot_data(ax1, X, y_radial, title='Original data')
# ax3 = plt.subplot(223)
# utils.plot_classifier(ax3, X_radial, y_radial, clf, title='Linear classifier on transformed data')
# ax4 = plt.subplot(224)
# utils.plot_classifier(ax4, X, y_radial, clf, phi=phi_radial, title='Projection of classifier onto original data')
# plt.show()

#-------------------------------------------------------------------------------
# Part 4.2 - Radial Kernel
#-------------------------------------------------------------------------------

# def radial_kernel(x_j, x_i):
#     raise NotImplementedError

# clf = lab4.KernelPerceptron(kernel=radial_kernel)
# clf.fit(X, y_radial)

# ax1 = plt.subplot(121)
# utils.plot_classifier(ax1, X, y_radial, clf, title='RBF Perceptron')


# clf = svm.SVC(kernel='rbf')
# clf.fit(X, y_radial)

# ax2 = plt.subplot(122)
# utils.plot_classifier(ax2, X, y_radial, clf, title='RBF SVM')

# plt.show()

#-------------------------------------------------------------------------------
# Part 5 - Angle Models
#-------------------------------------------------------------------------------

# def phi_angular(x):
#     raise NotImplementedError

# X_angular = np.array([phi_angular(x) for x in X])

# clf = svm.SVC(kernel='linear')
# clf.fit(X_angular, y_angular)

# ax1 = plt.subplot(221)
# utils.plot_data(ax1, X, y_angular, title='Original data')
# ax3 = plt.subplot(223)
# utils.plot_classifier(ax3, X_angular, y_angular, clf, title='Linear classifier on transformed data')
# ax4 = plt.subplot(224)
# utils.plot_classifier(ax4, X, y_angular, clf, phi=phi_angular, title='Projection of classifier onto original data')
# plt.show()

#-------------------------------------------------------------------------------
# Part 6 - Bonus: k-Nearest Neighbors
#-------------------------------------------------------------------------------

# # Generate and plot the clustered data
# X_cluster, y_cluster = make_blobs(n_samples=200, centers=[[0,4], [-4,0], [4,0], [0,-4], [0,0]])

# ax1 = plt.subplot(331)
# utils.plot_data(ax1, X_cluster, y_cluster, title='Clusters')

# # Plot KNN predictions for different values of k
# for i in trange(8):
#     k = i + 1
#     clf = KNeighborsClassifier(n_neighbors=k)
#     clf.fit(X_cluster, y_cluster)

#     axi = plt.subplot(330 + i + 2)
#     utils.plot_classifier(axi, X_cluster, y_cluster, clf, title='{}-Nearest Neighbors'.format(k))

# plt.show()
