from ivm import get_active_set
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

np.seed(244)

def kernel(X, Y, alpha, beta, gamma):
    kernel = kernels.RBF(length_scale=(1./gamma**2))
    return np.matrix(alpha*kernel(X, Y) + np.eye(X.shape[0])/beta**2)

def optimize_active(Xi, Yi):
    YiYiT = np.dot(Yi, Yi.T)

    def active_set_likelihood(var, *args):
        YiYiT, N, D = args
        X = var[:-3].reshape((N,D))
        alpha, beta, gamma = var[-3:]
        Kii = kernel(X, X, alpha, beta, gamma)
        return np.log(np.linalg.det(Kii)) + np.sum(inner1d(Kii.I).dot(YiYiT))

    kernel_params = np.ones(3)
    x0 = np.concatenate(Xi.flatten(), kernel_params)
    res = fmin_cg(active_set_likelihood, x0, disp=True)
    X_opt = res[:-3].reshape(Xi.shape)
    kernel_params = res[-3:]
    return X_opt, kernel_params

def optimize_latent(Xj, Yj):


def gplvm(Y, latent_dimensions, active_set_size):
    Y = Y - np.mean(Y, axis=0)
    # Initialize A with PCA
    X = PCA(n_components=latent_dimension).fit_transform(Y)
    alpha, beta, gamma = np.ones(3)
    # Compute the kernel without noise model (will be treated by ivm)
    K = kernel(X, X, alpha, 0, gamma)
    I, J = [], []

    for t in tqdm(range(n_iterations)):
        I, J = get_active_set(K, beta, active_set_size)
        Xi = X[I]
        Yi = Y[I]
        # Optimize kernel parameters and active set
        X_active, kernel_params = optimize_active(Xi, Yi)
        alpha, beta, gamma = kernel_params
        K = kernel(X, X, alpha, 0, gamma)
        # Select a new active set
        I, J = get_active_set(K, beta, active_set_size)

        for j in J:
            <
