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
from mice_genes import load_genes_dataset


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

    def __kernel_point(self, x, y, alpha, beta, gamma):
        if x == y:
            return alpha*np.exp(-gamma*np.dot(x-y, x-y)/2) + beta**2
        return alpha*np.exp(-gamma*np.dot(x-y, x-y)/2)

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
        dkdgamma = alpha*dkdalpha*np.log(Kii)/gamma

        a = np.sum(inner1d(dkdl, dkdalpha))  # Fast product and trace computation
        b = np.sum(inner1d(dkdl, dkdbeta))
        c = np.sum(inner1d(dkdl, dkdgamma))

        return np.array([a, b, c])

    def __dk_dl(self, K_inv, YYT):
        return -K_inv*YYT*K_inv + self.D*K_inv

    def fit_transform(self, Y, iterations):
        ''' Implementation of GPLVM algorithm, returns data in latent space
        '''
        # x = StandardScaler().fit_transform(x)  # TODO(oleguer): Standardize data??

        self.N = Y.shape[0]
        self.D = Y.shape[1]

        X = PCA(n_components=self.latent_dim).fit_transform(Y)
        kernel_params = np.ones(3)  # (alpha, beta, gamma)

        for _ in tqdm(range(iterations)):
            active_set, _ = fake_ivm(X, self.active_set_size)  # TODO(oleguer): Call real ivm by federico
            Xi = np.matrix(X[active_set, :])
            Yi = np.matrix(Y[active_set, :])
            YiYiT = Yi*Yi.T  # Precompute this product

            # Optimize kernel parameters using active set
            t = time.time()
            self.timer.tic()
            # kernel_params = fmin_cg(f = self.__kernel_param_loglike, 
            #                         # fprime=self.__kernel_param_loglike_dif,
            #                         x0 = kernel_params,
            #                         args=tuple((Xi, YiYiT)),
            #                         disp=True)
            out = minimize(fun = self.__kernel_param_loglike, 
                                    # jac=self.__kernel_param_loglike_dif,  #TODO(oleguer) Fix differential form!!!
                                    x0 = kernel_params,
                                    args = tuple((Xi, YiYiT)),
                                    options = {"disp" : True})
            self.timer.toc("First optimization")
            try:
                kernel_params = out.x
            except:
                pass

            alpha = kernel_params[0]
            beta = kernel_params[1]
            gamma = kernel_params[2]

            return alpha, beta, gamma

if __name__ == "__main__":
    # N = 100  # Number of observations
    # n_classes = 3  # Number of classes
    # D = 4  # Y dimension (observations)
    # observations, labels = generate_observations(N, D, n_classes)

    N, n_classes, D, observations, labels = load_genes_dataset(30, 10)
    gp_vals = GPLVM(active_set_size = 20).fit_transform(observations, iterations = 1)
    print(gp_vals)
    # pca = PCA(n_components=2).fit_transform(observations)

    # plot(pca, gp_vals, labels)