from matplotlib import pyplot as plt
import numpy as np
from numpy.core.umath_tests import inner1d  # Fast trace multiplication
from scipy.optimize import fmin_cg  # Non-linear SCG
from scipy.stats import multivariate_normal
from sklearn.decomposition import PCA  # For X initialization
from sklearn.preprocessing import StandardScaler  # To standardize data
from sklearn.gaussian_process import kernels
import time
from tqdm import tqdm
from fake_dataset import generate_observations, plot

name = "experiemnt"

def kernel(X, Y, alpha, beta, gamma):
    kernel = kernels.RBF(length_scale=(1./gamma**2))
    return np.matrix(alpha*kernel(X, Y) + np.eye(X.shape[0])/(beta**2))

def likelihood(var, *args):
    YYT, N, D, latent_dimension, = args

    X = np.array(var[:-3]).reshape((N, latent_dimension))
    alpha = var[-3]
    beta = var[-2]
    gamma = var[-1]
    K = kernel(X, X, alpha, beta, gamma)

    # return -log likelihood
    trace = np.sum(inner1d(K.I, YYT))
    return D*np.log(np.abs(np.linalg.det(K)))/2 + trace/2

def save_vars(var):
    np.save("results/" + str(name) + ".npy", var)

def simple_gplvm(Y, experiment_name="experiment", latent_dimension=2):
    ''' Implementation of GPLVM algorithm, returns data in latent space
    '''
    global name
    name = experiment_name
    Y = np.matrix(Y)
    # TODO(oleguer): Should we center observations?
    
    # Initialize X through PCA
 
    # First X approximation
    X = PCA(n_components=latent_dimension).fit_transform(Y)
    kernel_params = np.ones(3)  # (alpha, beta, gamma) TODO(oleguer): Should we rewrite those at each iteration? I dont thinkn so

    var = list(X.flatten()) + list(kernel_params)
    YYT = Y*Y.T
    N = Y.shape[0]
    D = Y.shape[1]

    # Optimization
    t1 = time.time()
    var = fmin_cg(likelihood, var, args=tuple((YYT,N,D,latent_dimension,)), epsilon = 0.001, disp=True, callback=save_vars)
    print("time:", time.time() - t1)

    var = list(var)

    np.save("results/" + str(name) + "_final.npy", var)

    N = Y.shape[0]
    X = np.array(var[:-3]).reshape((N, latent_dimension))
    alpha = var[-3]
    beta = var[-2]
    gamma = var[-1]

    print("alpha", alpha)
    print("beta", beta)
    print("gamma", gamma)

    return X


if __name__ == "__main__":
    N = 30  # Number of observations
    n_classes = 3  # Number of classes
    D = 4  # Y dimension (observations)

    observations, labels = generate_observations(N, D, n_classes)
    # x = StandardScaler().fit_transform(x)  # Standardize??

    gp_vals = simple_gplvm(Y=observations, experiment_name="test")  # Compute values
    # gp_vals = np.array(list(np.load("results/var.npy"))[:-3]).reshape((N, 2))  # Load from memory

    pca = PCA(n_components=2).fit_transform(observations)

    print("distance:", np.linalg.norm(gp_vals-pca))

    plot(pca, gp_vals, labels)
