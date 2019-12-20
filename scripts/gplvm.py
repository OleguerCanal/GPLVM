import numpy as np
from matplotlib import pyplot as plt

from sklearn.decomposition import PCA  # For X initialization
from sklearn.preprocessing import StandardScaler  # To standardize data

from fake_dataset import generate_observations, plot
np.seed = 0

def fake_ivm(data, size):
    ''' Returns random subset of the data
        TODO(oleguer): Call real ivm algorithm by Federico
    ''' 
    idx = np.random.randint(data.shape[0], size=size)
    return data[idx, :]

def gplvm(data, active_set_size, iterations, latent_dimension = 2):
    # Initialize X through PCA
    x = PCA(n_components = latent_dimension).fit_transform(data)
    for t in range(iterations):
        active_data = fake_ivm(data, active_set_size)
        # return active_data
    return x
    

if __name__ == "__main__":
    N = 100  # Number of observations
    n_classes = 3  # Number of classes
    D = 5  # Y dimension (observations)
    
    observations, labels = generate_observations(N, D, n_classes)
    # Standardize?
    # x = StandardScaler().fit_transform(x)
    
    ppl_comp = gplvm(data = observations, active_set_size = 10, iterations = 1)

    plot(ppl_comp, labels)