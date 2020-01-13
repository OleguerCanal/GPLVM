from matplotlib import pyplot as plt
import numpy as np
from numpy.core.umath_tests import inner1d  # Fast trace multiplication
from scipy.optimize import fmin_cg, minimize  # Non-linear SCG
from scipy.stats import multivariate_normal
from sklearn.decomposition import PCA  # For X initialization
from sklearn.preprocessing import StandardScaler  # To standardize data
from sklearn.gaussian_process import kernels
import time
from tqdm import tqdm
from fake_dataset import generate_observations, plot
from mice_genes import load_genes_dataset, plot_genes
from datetime import datetime
from ivm import get_active_set

np.random.seed(0)

def fake_ivm(data, size):
    ''' Returns random partition of data
    '''
    indices = np.random.permutation(data.shape[0])
    active_set, inactive_set = indices[:size], indices[size:]
    return active_set, inactive_set

class Timer:
    ''' To evaluate methods costs
    '''
    def __init__(self):
        self.t = None

    def tic(self):
        self.t = time.time()

    def toc(self, name = ""):
        print("Time of", name, time.time() - self.t)


class GPLVM:
    def __init__(self, active_set_size, latent_dim=2):
        self.active_set_size = active_set_size
        self.latent_dim = latent_dim
        self.timer = Timer()

    def __kernel(self, X, Y, alpha, beta, gamma):
        kernel = kernels.RBF(length_scale=(1./gamma**2))
        return np.matrix(alpha*kernel(X, Y) + np.eye(X.shape[0])*beta**2)

    def __kernel_param_loglike(self, params, *args):
        ''' Kernel Optimization: Equation (4) of the paper
        '''
        Xi, YiYiT = args
        Kii = self.__kernel(Xi, Xi, alpha=params[0], beta=params[1], gamma=params[2])
        Kii_inv = Kii.I
        neg_loglike = self.D*np.log(np.abs(np.linalg.det(Kii))) + np.sum(inner1d(Kii_inv, YiYiT))
        return neg_loglike/2

    def __kernel_param_loglike_dif(self, params, *args):
        ''' Kernel Optimization: Differential
        '''
        alpha = params[0]
        beta = params[1]
        gamma = params[2]

        Xi, YYT = args
        Kii = self.__kernel(Xi, Xi, alpha=alpha, beta=beta, gamma=gamma)
        Kii_inv = Kii.I

        dkdl = self.__dk_dl(Kii_inv, YYT).T  # dK/dL transposed

        dkdalpha = (Kii - np.eye(self.active_set_size)*(beta**2))/alpha
        dkdbeta = 2*beta*np.eye(self.active_set_size)
        dkdgamma = alpha*dkdalpha*np.log(dkdalpha)/gamma

        a = np.sum(inner1d(dkdl, dkdalpha))  # Fast product and trace computation
        b = np.sum(inner1d(dkdl, dkdbeta))
        c = np.sum(inner1d(dkdl, dkdgamma))

        return np.array([a, b, c])

    def __dk_dl(self, K_inv, YYT):
        return -K_inv*YYT*K_inv + self.D*K_inv

    def __latent_loglike(self, xj, *args):
        j, = args
        self.X[j, :] = xj
        K_Ij = self.__kernel(self.X, self.X, alpha=self.kernel_params[0], beta=self.kernel_params[1], gamma=self.kernel_params[2])[self.active_set, j]
        mu = self.YTKII_inv*K_Ij
        sigma_sq = self.k_xx - (K_Ij.T*self.Kii_inv*K_Ij).item()
        yj = self.Y[j, :]
        sub = yj.reshape(mu.shape) - mu
        return np.log(sigma_sq)/2 + np.dot(sub.T, sub).item()/(2*sigma_sq)

    def fit_transform(self, Y, iterations, disp = False):
        ''' Implementation of GPLVM algorithm, returns data in latent space
        '''
        # x = StandardScaler().fit_transform(x)  # TODO(oleguer): Standardize data??
        self.N = Y.shape[0]
        self.D = Y.shape[1]
        self.Y = Y

        self.X = PCA(n_components=self.latent_dim).fit_transform(Y)
        self.kernel_params = np.ones(3)  # (alpha, beta, gamma)

        for _ in tqdm(range(iterations)):
            # active_set, _ = fake_ivm(self.X, self.active_set_size)  # TODO(oleguer): Call real ivm by federico
            active_set, _ = get_active_set(
                K = self.__kernel(self.X, self.X, alpha=self.kernel_params[0], beta=0, gamma=self.kernel_params[2]),
                noise_model_var = self.kernel_params[1],
                size=self.active_set_size)
            Xi = np.matrix(self.X[active_set, :])
            Yi = np.matrix(self.Y[active_set, :])
            YiYiT = Yi*Yi.T  # Precompute this product

            # Optimize kernel parameters using active set
            # self.timer.tic()
            out = minimize( fun = self.__kernel_param_loglike, 
                            # jac=self.__kernel_param_loglike_dif,  # TODO(oleguer) Fix differential form!!!
                            x0 = self.kernel_params,
                            args = tuple((Xi, YiYiT)),
                            options = {"disp" : disp})
            # self.timer.toc("First optimization")
            try:
                self.kernel_params = out.x
                # print("kernel_params", out.x)
            except:
                pass

            # self.active_set, inactive_set = fake_ivm(self.X, self.active_set_size)  # TODO(oleguer): Call real ivm by federico
            self.active_set, inactive_set = get_active_set(
                K = self.__kernel(self.X, self.X, alpha=self.kernel_params[0], beta=0, gamma=self.kernel_params[2]),
                noise_model_var = self.kernel_params[1],
                size=self.active_set_size)
            Xi = np.matrix(self.X[self.active_set, :])
            Yi = np.matrix(self.Y[self.active_set, :])
            Kii = self.__kernel(Xi, Xi, alpha=self.kernel_params[0], beta=self.kernel_params[1], gamma=self.kernel_params[2])
            K = self.__kernel(self.X, self.X, alpha=self.kernel_params[0], beta=self.kernel_params[1], gamma=self.kernel_params[2])

            self.Kii_inv = Kii.I
            self.YTKII_inv = Yi.T*self.Kii_inv
            self.k_xx = Kii[0, 0]

            for j in inactive_set:
                out = minimize( fun = self.__latent_loglike, 
                                # jac=self.__latent_loglikedif,  # TODO(oleguer)
                                x0 = self.X[j, :],
                                args = tuple((j,)),
                                options = {"disp" : disp})
                self.X[j, :] = out.x

        return self.X

if __name__ == "__main__":
    N, n_classes, D, observations, labels = load_genes_dataset(100, 30)
    print("N", N)
    print("D", D)
    print("n_classes", n_classes)
    gp_vals = GPLVM(active_set_size = 20).fit_transform(observations, iterations = 10)
    # print(gp_vals)
    pca = PCA(n_components=2).fit_transform(observations)

    plot_genes(pca, gp_vals, labels)