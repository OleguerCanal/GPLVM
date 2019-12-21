from matplotlib import pyplot as plt
import numpy as np
from numpy.core.umath_tests import inner1d  # Fast trace multiplication
from scipy.optimize import fmin_cg  # Non-linear SCG
from scipy.stats import multivariate_normal
from sklearn.decomposition import PCA  # For X initialization
from sklearn.preprocessing import StandardScaler  # To standardize data
from sklearn.gaussian_process import kernels
from fake_dataset import generate_observations, plot

import time
np.seed = 0

def fake_ivm(data, size):
    ''' Returns random partition of data
        TODO(oleguer): Call real ivm algorithm by Federico
    '''
    indices = np.random.permutation(data.shape[0])
    active_set, inactive_set = indices[:size], indices[size:]
    return active_set, inactive_set


def kernel(X, Y, alpha, beta, gamma):
    kernel = kernels.RBF(length_scale=(1./gamma**2))
    return np.matrix(alpha*kernel(X, Y) + np.eye(X.shape[0])/beta)


def active_set_likelihood(params, *args):
    ''' Kernel Optimization: Equation (4) of the paper
    '''
    Yi, YiYiT = args
    Kii = kernel(Yi, Yi, alpha=params[0], beta=params[1], gamma=params[2])
    neg_loglike = 0.5*np.sum(inner1d(Kii.I, YiYiT)) +\
                 np.log(np.linalg.det(Kii))/2.
                # + np.log(2*np.pi)*Yi.shape[1]/2. # OBS: Does not change optimization
    return neg_loglike


def latent_var_prob(xj, *args):
    ''' Latent Var Optimization: Equation (3) of the paper
    '''
    y_j, f_j, sigma_second_term, alpha, beta, gamma = args
    sigma_sq_j = kernel(np.mat(xj), np.mat(xj), alpha, beta, gamma).item()
    sigma_sq_j -= sigma_second_term.item()
    cov = sigma_sq_j*np.eye(f_j.shape[0])
    # f_j = np.array(f_j[:, 0].flatten())[0]
    cosa = multivariate_normal(list(f_j), cov)
    cosa2 = cosa.pdf(y_j)
    return cosa2


def gplvm(Y, active_set_size, iterations, latent_dimension=2):
    ''' Implementation of GPLVM algorithm, returns data in latent space
    '''
    # Initialize X through PCA
    X = PCA(n_components=latent_dimension).fit_transform(Y)
    kernel_params = np.ones(3)  # (alpha, beta, gamma) TODO(oleguer): Should we rewrite those at each iteration?

    for t in range(iterations):
        # Select a new active set using the IVM algorithm
        active_set, _ = fake_ivm(Y, active_set_size)
        Yi = np.matrix(Y[active_set, :])
        YiYiT = Yi*Yi.T  # Precompute this product

        # Optimise (4) wrt the parameters of K using SCG
        kernel_params = fmin_cg(
            f = active_set_likelihood, x0 = kernel_params, args=tuple((Yi,YiYiT)))
        alpha = kernel_params[0]
        beta = kernel_params[1]
        gamma = kernel_params[2]

        # Select a new active set
        active_set, inactive_set = fake_ivm(Y, active_set_size)
        Yi = np.matrix(Y[active_set, :])
        # Yj = np.matrix(Y[inactive_set, :])
        Kii = kernel(Yi, Yi, alpha, beta, gamma)
        Kii_inv = Kii.I
        K = kernel(Y, Y, alpha, beta, gamma)
        for j in inactive_set:
            # Optimise (3) wrt xj using SCG
            y_j = Y[j, :]
            k_j = K[active_set, j]
            f_j = Yi.T*Kii_inv*k_j  # TODO(oleguer): Review this, paper says Y.t but doesnt make sense
            f_j = np.array(f_j[:, 0].flatten())[0]
            sigma_second_term = k_j.T*Kii_inv*k_j
            args = tuple((y_j, f_j, sigma_second_term, alpha, beta, gamma))
            X[j, :] = fmin_cg(latent_var_prob, X[j, :], args=args)
    return X


if __name__ == "__main__":
    N = 100  # Number of observations
    n_classes = 3  # Number of classes
    D = 5  # Y dimension (observations)

    observations, labels = generate_observations(N, D, n_classes)
    # x = StandardScaler().fit_transform(x)  # Standardize??

    gp_vals = gplvm(Y=observations,
                     active_set_size=20,
                     iterations=3)
    pca = PCA(n_components=2).fit_transform(observations)


    plot(pca, gp_vals, labels)
