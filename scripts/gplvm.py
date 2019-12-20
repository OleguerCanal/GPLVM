import numpy as np
from matplotlib import pyplot as plt

from sklearn.decomposition import PCA  # For X initialization
from sklearn.preprocessing import StandardScaler  # To standardize data

from fake_dataset import generate_observations, plot
np.seed = 0


def gplvm(data, active_set_size, iterations):
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(data)
    return principalComponents
    

if __name__ == "__main__":
    N = 100  # Number of observations
    n_classes = 3  # Number of classes
    D = 5  # Y dimension (observations)
    q = 2  # X dimension (latent)
    
    observations, labels = generate_observations(N, D, n_classes)
    # Standardize?
    # x = StandardScaler().fit_transform(x)
    
    ppl_comp = gplvm(data = observations, active_set_size = 25, iterations =0)

    plot(ppl_comp, labels)