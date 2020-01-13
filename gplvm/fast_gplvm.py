import _pickle as cPickle
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

# from fake_dataset import generate_observations, plot
from exp_mice import load_genes_dataset, plot_genes
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

    def toc(self, name=""):
        print("Time of", name, time.time() - self.t)


class GPLVM:
    def __init__(self, active_set_size, latent_dim=2, name="gplvm_class"):
        self.active_set_size = active_set_size
        self.latent_dim = latent_dim
        self.timer = Timer()
        self.name = name

    def __kernel(self, X, Y, alpha, beta, gamma):
        kernel = kernels.RBF(length_scale=(1./gamma**2))
        return np.matrix(alpha*kernel(X, Y) + np.eye(X.shape[0])*beta**2)

    def __kernel_param_loglike(self, params, *args):
        ''' Kernel Optimization: Equation (4) of the paper
        '''
        Xi, YiYiT = args
        Kii = self.__kernel(
            Xi, Xi, alpha=params[0], beta=params[1], gamma=params[2])
        Kii_inv = Kii.I
        neg_loglike = self.D * \
            np.log(np.abs(np.linalg.det(Kii))) + \
            np.sum(inner1d(Kii_inv, YiYiT))
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

        # Fast product and trace computation
        a = np.sum(inner1d(dkdl, dkdalpha))
        b = np.sum(inner1d(dkdl, dkdbeta))
        c = np.sum(inner1d(dkdl, dkdgamma))

        return np.array([a, b, c])

    def __dk_dl(self, K_inv, YYT):
        return -K_inv*YYT*K_inv + self.D*K_inv

    def __latent_loglike(self, xj, *args):
        j, = args
        self.X[j, :] = xj
        K_Ij = self.__kernel(self.X, self.X, alpha=self.kernel_params[0], beta=self.kernel_params[1], gamma=self.kernel_params[2])[
            self.active_set, j]
        mu = self.YTKII_inv*K_Ij
        sigma_sq = self.k_xx - (K_Ij.T*self.Kii_inv*K_Ij).item()
        yj = self.Y[j, :]
        sub = yj.reshape(mu.shape) - mu
        return np.log(sigma_sq)/2 + np.dot(sub.T, sub).item()/(2*sigma_sq)

    def fit_transform(self, Y, iterations, disp=False, save=False):
        ''' Implementation of GPLVM algorithm, returns data in latent space
        '''
        # x = StandardScaler().fit_transform(x)  # TODO(oleguer): Standardize data??
        self.N = Y.shape[0]
        self.D = Y.shape[1]
        self.Y = Y

        self.X = PCA(n_components=self.latent_dim).fit_transform(Y)
        self.kernel_params = np.ones(3)  # (alpha, beta, gamma)

        for it in tqdm(range(iterations)):
            # active_set, _ = fake_ivm(self.X, self.active_set_size)  # TODO(oleguer): Call real ivm by federico
            active_set, _ = get_active_set(
                K=self.__kernel(
                    self.X, self.X, alpha=self.kernel_params[0], beta=0, gamma=self.kernel_params[2]),
                noise_model_var=self.kernel_params[1],
                size=self.active_set_size)
            Xi = np.matrix(self.X[active_set, :])
            Yi = np.matrix(self.Y[active_set, :])
            YiYiT = Yi*Yi.T  # Precompute this product

            # Optimize kernel parameters using active set
            # self.timer.tic()
            out = minimize(fun=self.__kernel_param_loglike,
                           # jac=self.__kernel_param_loglike_dif,  # TODO(oleguer) Fix differential form!!!
                           x0=self.kernel_params,
                           args=tuple((Xi, YiYiT)),
                           options={"disp": disp})
            # self.timer.toc("First optimization")
            self.kernel_params = out.x

            # self.active_set, inactive_set = fake_ivm(self.X, self.active_set_size)  # TODO(oleguer): Call real ivm by federico
            self.active_set, inactive_set = get_active_set(
                K=self.__kernel(
                    self.X, self.X, alpha=self.kernel_params[0], beta=0, gamma=self.kernel_params[2]),
                noise_model_var=self.kernel_params[1],
                size=self.active_set_size)
            Xi = np.matrix(self.X[self.active_set, :])
            Yi = np.matrix(self.Y[self.active_set, :])
            Kii = self.__kernel(
                Xi, Xi, alpha=self.kernel_params[0], beta=self.kernel_params[1], gamma=self.kernel_params[2])
            K = self.__kernel(
                self.X, self.X, alpha=self.kernel_params[0], beta=self.kernel_params[1], gamma=self.kernel_params[2])

            self.Kii_inv = Kii.I
            self.YTKII_inv = Yi.T*self.Kii_inv
            self.k_xx = Kii[0, 0]

            for j in inactive_set:
                out = minimize(fun=self.__latent_loglike,
                               # jac=self.__latent_loglikedif,  # TODO(oleguer)
                               x0=self.X[j, :],
                               args=tuple((j,)),
                               options={"disp": disp})
                self.X[j, :] = out.x

            if save and it%10 == 0:
                self.save()

        return self.X

    # TODO(oleguer): Review this, I think always returns 0 (good thing?)
    def get_precision(self, x):
        ''' returns latent space precision at point x = [x1, x2] (or any number of dimensions)
        '''
        X = np.array(list(self.X) + [x])
        K = self.__kernel(
            X, X, alpha=self.kernel_params[0], beta=self.kernel_params[1], gamma=self.kernel_params[2])
        k_xx = K[0, 0]
        K_Ij = K[:, -1]
        K_inv = K.I
        sigma_sq = k_xx - (K_Ij.T*K_inv*K_Ij).item()
        return sigma_sq

    def load(self, name):
        f = open(name, 'rb')
        tmp_dict = cPickle.load(f)
        f.close()
        self.__dict__.update(tmp_dict)

    def save(self):
        name = "results/" + self.name + "_" + \
            str(datetime.now()).replace(" ", "_")
        f = open(name, 'wb')
        cPickle.dump(self.__dict__, f, 2)
        f.close()


if __name__ == "__main__":
    N, n_classes, D, observations, labels = load_genes_dataset(100, 10)
    print("N", N)
    print("D", D)
    print("n_classes", n_classes)
    pca = PCA(n_components=2).fit_transform(observations)
    gplvm = GPLVM(active_set_size=20)
    # gplvm.load("results/gplvm_class_2020-01-13_19:57:38.201853")  # How to load previous state
    # gp_vals = gplvm.X
    gp_vals = gplvm.fit_transform(observations, iterations=5, save = True)

    # Precision
    precision = gplvm.get_precision([0.5, 0.3])
    print(precision)

    plot_genes(pca, gp_vals, labels)
