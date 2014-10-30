import numpy as np
import matplotlib

from matplotlib import pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_moons


def plot_adaboost():
    X, y = make_moons(noise=0.3, random_state=0)

    # Create and fit an AdaBoosted decision tree
    est = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                             algorithm="SAMME.R",
                             n_estimators=200)

    sample_weight = np.empty(X.shape[0], dtype=np.float)
    sample_weight[:] = 1. / X.shape[0]

    est._validate_estimator()
    est.estimators_ = []
    est.estimator_weights_ = np.zeros(4, dtype=np.float)
    est.estimator_errors_ = np.ones(4, dtype=np.float)

    plot_step = 0.02

    # Plot the decision boundaries
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))

    fig, axes = plt.subplots(1, 4, figsize=(14, 4), sharey=True)
    colors = ['#d7191c', '#fdae61', '#ffffbf', '#abd9e9', '#2c7bb6']
    c = lambda a, b, c: map(lambda x: x / 254.0, [a, b, c])
    colors = [c(215, 25, 28),
              c(253, 174, 97),
              c(255, 255, 191),
              c(171, 217, 233),
              c(44, 123, 182),
              ]

    for i, ax in enumerate(axes):
        sample_weight, estimator_weight, estimator_error = est._boost(i, X, y, sample_weight)
        est.estimator_weights_[i] = estimator_weight
        est.estimator_errors_[i] = estimator_error
        sample_weight /= np.sum(sample_weight)

        Z = est.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z,
                    cmap=matplotlib.colors.ListedColormap([colors[1], colors[-2]]),
                    alpha=1.0)
        ax.axis("tight")

        # Plot the training points
        ax.scatter(X[:, 0], X[:, 1],
                   c=np.array([colors[0], colors[-1]])[y],
                   s=20 + (200 * sample_weight) ** 2, cmap=plt.cm.Paired)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel('$x_0$')

        if i == 0:
            ax.set_ylabel('$x_1$')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    plot_adaboost()
