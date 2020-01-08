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

def kernel_test(x, y, alpha, beta, gamma):
    return alpha*np.exp(-gamma*np.dot(x-y, x-y)/2) + 1./beta

def active_set_likelihood(params, *args):
    ''' Kernel Optimization: Equation (4) of the paper
    '''
    Xi, YiYiT = args
    Kii = kernel(Xi, Xi, alpha=params[0], beta=params[1], gamma=params[2])
    neg_loglike = 0.5*np.sum(inner1d(Kii.I, YiYiT)) +\
                 np.log(np.abs(np.linalg.det(Kii)))/2.
                # + np.log(2*np.pi)*Yi.shape[1]/2. # OBS: Does not change optimization
    return neg_loglike


def latent_var_prob(xj, *args):
    ''' Latent Var Optimization: Equation (3) of the paper
    '''
    y_j, Yi, Kii_inv, k_x_x, X, alpha, beta, gamma, active_set, j = args
    X[j, :] = xj
    K = kernel(X, X, alpha, beta, gamma)
    k_j = K[active_set, j]
    
    # Mean
    f_j = Yi.T*Kii_inv*k_j  # TODO(oleguer): Review this, paper says Y.t but doesnt make sense
    f_j = np.array(f_j[:, 0].flatten())[0]

    # Variance
    sigma_sq_j = k_x_x - (k_j.T*Kii_inv*k_j).item()
    cov = sigma_sq_j*np.eye(Yi.shape[1])
    
    # print(sigma_sq_j)
    # f_j = np.array(f_j[:, 0].flatten())[0]
    return -multivariate_normal(list(f_j), cov).logpdf(y_j)


def gplvm(Y, active_set_size, iterations, latent_dimension=2):
    ''' Implementation of GPLVM algorithm, returns data in latent space
    '''
    # Initialize X through PCA
    X = PCA(n_components=latent_dimension).fit_transform(Y)
    kernel_params = np.ones(3)  # (alpha, beta, gamma) TODO(oleguer): Should we rewrite those at each iteration? I dont thinkn so

    for t in tqdm(range(iterations)):
        # Select a new active set using the IVM algorithm
        active_set, _ = fake_ivm(X, active_set_size)  # TODO(oleguer): Pass Y or X here?
        Xi = np.matrix(X[active_set, :])
        Yi = np.matrix(Y[active_set, :])
        YiYiT = Yi*Yi.T  # Precompute this product

        # Optimise (4) wrt the parameters of K using SCG
        kernel_params = fmin_cg(
            f = active_set_likelihood, x0 = kernel_params, args=tuple((Xi, YiYiT)), disp=False)
        alpha = kernel_params[0]
        beta = kernel_params[1]
        gamma = kernel_params[2]
        # print("kernel_params:", kernel_params)

        # Select a new active set
        active_set, inactive_set = fake_ivm(X, active_set_size)  # TODO(oleguer): Pass Y or X here?
        Xi = np.matrix(X[active_set, :])
        Yi = np.matrix(Y[active_set, :])
        Kii_inv = kernel(Xi, Xi, alpha, beta, gamma).I
        k_x_x = kernel(np.mat(np.zeros(latent_dimension)), np.mat(np.zeros(latent_dimension)), alpha, beta, gamma).item()
        for j in inactive_set:
            # Optimise (3) wrt xj using SCG
            y_j = Y[j, :]
            args = tuple((y_j, Yi, Kii_inv, k_x_x, X, alpha, beta, gamma, active_set, j))
            X[j, :] = fmin_cg(latent_var_prob, X[j, :], args=args, epsilon = 0.1, disp=False)
            # quit()
    return X


if __name__ == "__main__":
    N = 100  # Number of observations
    n_classes = 3  # Number of classes
    D = 4  # Y dimension (observations)

    observations, labels = generate_observations(N, D, n_classes)
    # x = StandardScaler().fit_transform(x)  # Standardize??

    gp_vals = gplvm(Y=observations,
                     active_set_size=20,
                     iterations=10)
    pca = PCA(n_components=2).fit_transform(observations)

    print("ok")

    plot(pca, gp_vals, labels)
    print("ok")
