import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import fmin_cg  # Non-linear SCG
from scipy.stats import multivariate_normal
from sklearn.decomposition import PCA  # For X initialization
from sklearn.preprocessing import StandardScaler  # To standardize data
from sklearn.gaussian_process import kernels
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


def active_set_likelihood(params, *args):
    ''' Kernel Optimization: Equation (4) of the paper
    '''
    Yi = args[0]
    Kii = kernel(Yi, Yi, alpha=params[0], beta=params[1], gamma=params[2])
    # like = np.exp(-0.5*Yi.T*Kii.I*Yi) /\
    #     (((2*np.pi)**(Yi.shape[1]/2.))*np.linalg.det(Kii)**0.5)
    like = -0.5*Yi.T*Kii.I*Yi
    print(like.shape)
    like -= np.log(((2*np.pi)**(Yi.shape[1]/2.))*np.linalg.det(Kii)**0.5)
    return like


def latent_var_prob(xj, *args):
    ''' Latent Var Optimization: Equation (3) of the paper
    '''
    y_j, f_j, sigma_second_term, alpha, beta, gamma = args
    sigma_sq_j = kernel(xj, xj, alpha, beta, gamma) - sigma_second_term
    return multivariate_normal(f_j, sigma_sq_j).pdf(y_j)


def gplvm(Y, active_set_size, iterations, latent_dimension=2):
    ''' Implementation of GPLVM algorithm, returns data in latent space
    '''
    # Initialize X through PCA
    X = PCA(n_components=latent_dimension).fit_transform(Y)

    for t in range(iterations):
        # Select a new active set using the IVM algorithm
        active_set, _ = fake_ivm(Y, active_set_size)
        Yi = np.matrix(Y[active_set, :])

        # Optimise (4) wrt the parameters of K using SCG
        kernel_params_0 = np.ones(3)  # (alpha, beta, gamma)
        optimal_kernel_params = fmin_cg(
            f = active_set_likelihood, x0 = kernel_params_0, args=tuple((Yi,)))
        alpha = optimal_kernel_params[0]
        beta = optimal_kernel_params[1]
        gamma = optimal_kernel_params[2]

        # Select a new active set
        active_set, inactive_set = fake_ivm(Y, active_set_size)
        Yi = np.matrix(Y[active_set, :])
        Yj = np.matrix(Y[inactive_set, :])

        Kii = kernel(Yi, Yi, alpha, beta, gamma)
        Kii_inv = Kii.I
        YT_Kii_inv = Yj.T*Kii_inv
        for j in inactive_set:
            # Optimise (3) wrt xj using SCG
            y_j = Y[j, :]
            k_j = Kii[:, j]
            f_j = YT_Kii_inv*k_j
            sigma_second_term = k_j.T*Kii_inv*k_j
            args = tuple((y_j, f_j, sigma_second_term, alpha, beta, gamma))
            X[j, :] = fmin_cg(latent_var_prob, X[j, :], args=args)
    return X


if __name__ == "__main__":
    N = 500  # Number of observations
    n_classes = 3  # Number of classes
    D = 5  # Y dimension (observations)

    observations, labels = generate_observations(N, D, n_classes)
    # x = StandardScaler().fit_transform(x)  # Standardize??

    ppl_comp = gplvm(Y=observations,
                     active_set_size=100,
                     iterations=15)

    plot(ppl_comp, labels)
