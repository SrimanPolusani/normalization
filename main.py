# Import statements
import numpy as np
import matplotlib.pyplot as plt

# Converting txt data into nd array object using ',' delimiter
X_train = np.loadtxt('./houses.txt', usecols=(0, 1, 2, 3), delimiter=',')
y_train = np.loadtxt('./houses.txt', usecols=4, delimiter=',')

# Name of the features in the model
X_features = ['size(sqft)', 'bedrooms', 'floors', 'age']


def plot_data():
    # Plot the given data using matplotlib
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    for i in range(len(axes)):
        for j in range((len(axes))):
            axes[i, j].scatter(X_train[:, i], y_train)
            axes[i, j].set_xlabel(X_features[i])
            axes[i, j].set_ylabel("Price (1000s)")
    plt.show()


def plot_norm_diff():
    # Calculate mean and standard deviation
    mean = np.mean(X_train, axis=0)
    sigma = np.std(X_train, axis=0)

    # Mean normalization and zscore normalization
    mean_nor = (X_train - mean) / (X_train.max(axis=0) - X_train.min(axis=0))
    zscore_nor = (X_train - mean) / sigma

    # Plot the difference in the spread of data before normalization & with mean and zscore normalization
    graph_titles = ['Unnormalized values', 'Mean normalization', 'Zscore normalization']
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for i in range(len(axes)):
        axes[i].set_xlabel(X_features[0])
        axes[i].set_ylabel(X_features[-1])
        axes[i].set_title(graph_titles[i])
        axes[i].axis('equal')
    axes[0].scatter(X_train[:, 0], X_train[:, -1])
    axes[1].scatter(mean_nor[:, 0], mean_nor[:, -1])
    axes[2].scatter(zscore_nor[:, 0], zscore_nor[:, -1])
    fig.suptitle('Before and After normalization')
    plt.show()


plot_data()
plot_norm_diff()
