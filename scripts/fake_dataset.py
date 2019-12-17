import numpy as np
from matplotlib import pyplot as plt

np.seed = 0

def plot(data, labels):
    '''Simple scatter of 2d data in same figure
    '''
    fig, ax = plt.subplots()
    ax.scatter(data[:, 0], data[:, 1], c=labels.flatten())
    ax.grid()
    # fig.savefig("test.png")
    plt.show()


def generate_observations(N, D, n_classes):
    ''' Returns dataset of N multivariate_normals of dimention D and its labels
        with random covariance and mean at c = 0 : n_classes-1 (evenly)
        (For testing purposes)
    '''
    N_class = int(N/n_classes)

    A = np.random.rand(D,D)
    covariance = np.dot(A,A.transpose())  # Make it sim, positive definite
    observations = np.random.multivariate_normal(np.zeros(D), covariance, N_class)
    labels = np.zeros((observations.shape[0], 1))

    for c in range(1, n_classes):
        A = np.random.rand(D,D)
        covariance = np.dot(A,A.transpose())  # Make it sim, positive definite
        observations_c = np.random.multivariate_normal(np.zeros(D) + c, covariance, N_class)
        observations = np.concatenate((observations, observations_c), axis = 0)
        labels = np.concatenate((labels, np.zeros((observations_c.shape[0], 1)) + c), axis = 0)

    return observations, labels

if __name__ == "__main__":
    N = 300  # Number of observations
    n_classes = 3  # Number of classes
    D = 2  # Y dimension (observations)
    q = 2  # X dimension (latent)
    
    observations, labels = generate_observations(N, D, n_classes)
    # TODO(oleguer): Reduce dimentions to 2D
    # d = 25  # Size of active set
    # T = 10  # Number of iterations
    # ...
    plot(observations, labels)