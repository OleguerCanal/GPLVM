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
from ivm import get_active_set

np.random.seed(1287123)


def kernel(X, Y, alpha, beta, gamma, noise=True):
    kernel = kernels.RBF(length_scale=(1./gamma**2))
    K = np.matrix(alpha*kernel(X, Y))
    if noise:
        K += np.eye(X.shape[0])/beta**2
    return K
    

def kernel_test(x, y, alpha, beta, gamma):
    return alpha*np.exp(-gamma*np.dot(x-y, x-y)/2) + 1./beta

def active_set_likelihood_params(params, *args):
    ''' Kernel Optimization: Equation (4) of the paper
    '''
    Xi, YiYiT = args
    Kii = kernel(Xi, Xi, alpha=params[0], beta=params[1], gamma=params[2])
    neg_loglike = 0.5*np.sum(inner1d(Kii.I, YiYiT)) +\
                 np.log(np.abs(np.linalg.det(Kii)))/2.
                # + np.log(2*np.pi)*Yi.shape[1]/2. # OBS: Does not change optimization
    return neg_loglike

def active_set_likelihood(X, *args):
    YiYiT, alpha, beta, gamma, X_shape = args
    X = X.reshape(X_shape)
    K = kernel(X, X, alpha, beta, gamma)
    neg_loglike = 0.5 * np.sum(inner1d(K.I, YiYiT)) + np.log(np.abs(np.linalg.det(K))) / 2
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
        cur_kernel = kernel(X, X, *kernel_params, noise=False)
        # Select a new active set using the IVM algorithm
        active_set, _ = get_active_set(cur_kernel, kernel_params[1], active_set_size)
        print(active_set)
        
        Xi = np.matrix(X[active_set, :])
        Yi = np.matrix(Y[active_set, :])
        YiYiT = Yi*Yi.T  # Precompute this product

        # Optimise (4) wrt the parameters of K using SCG
        kernel_params = fmin_cg(
            f = active_set_likelihood_params, x0 = kernel_params, args=tuple((Xi, YiYiT)), disp=True)

        # Optimize wrt XI (active set latent variables) (optional, why??)
        Xi = optimize_active(Xi, Yi, *kernel_params)
        for row, idx in zip(Xi, active_set):
            X[idx] = row

        # Select a new active set
        cur_kernel = kernel(X, X, *kernel_params, noise=False)
        active_set, inactive_set = get_active_set(cur_kernel, kernel_params[1], active_set_size)
        Xi = np.matrix(X[active_set, :])
        Yi = np.matrix(Y[active_set, :])
        Kii_inv = kernel(Xi, Xi, *kernel_params).I
        k_x_x = kernel(np.mat(np.zeros(latent_dimension)), np.mat(np.zeros(latent_dimension)),
                       *kernel_params).item()
        for j in inactive_set:
            # Optimise (3) wrt xj using SCG
            y_j = Y[j, :]
            args = tuple((y_j, Yi, Kii_inv, k_x_x, X, *kernel_params, active_set, j))
            X[j, :] = fmin_cg(latent_var_prob, X[j, :], args=args, disp=False)
    return X

def optimize_active(X, Y, alpha, beta, gamma):
    assert X.shape[0] == Y.shape[0]
    Y = np.matrix(Y)
    latent_dimension = X.shape[1]
    X_0 = np.random.normal(0, 1, X.shape).flatten()
    YiYiT = Y * Y.T
    args = tuple((YiYiT, alpha, beta, gamma, X.shape))
    X_opt = fmin_cg(f = active_set_likelihood, x0 = X_0, args=args)
    return X_opt



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
