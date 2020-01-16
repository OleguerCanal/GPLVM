import numpy as np
from scipy.linalg import eigh

def kernelPCA(data, components, kernel='poly', sigma=1, p=1):

    def polinomial_kernel(x1, x2, p=1):
        return (np.dot(x1, x2) + 1)**p

    def radial_base_kenel(x1, x2, sigma=1):
        return np.exp(-np.sum((x1 - x2)*(x1 - x2))/(2*(sigma**2)))

    def kenrel_matrix(x, kernel):
        K = np.zeros((len(x), len(x)))
        N_matrix = np.ones_like(K)/len(x)
        K = np.array([kernel(x1, x2) for x1 in x for x2 in x]).reshape(K.shape)
        return K - np.dot(N_matrix,K) - np.dot(K, N_matrix) + np.dot(np.dot(N_matrix, K), N_matrix)

    if kernel == 'poly':
        my_kernel = lambda x1, x2: polinomial_kernel(x1, x2, p)
    # if not polynomial, use RBF kernel
    else:
        my_kernel = lambda x1, x2: radial_base_kenel(x1, x2, sigma)

    K = kenrel_matrix(data, kernel=my_kernel)
    eigvals, eigvecs = eigh(K)

    return np.column_stack(tuple(eigvecs[:,-i] for i in range(1,components+1)))


## CODE FOR TESTING

import matplotlib.pyplot as plt

from sklearn.datasets import make_moons

X, y = make_moons(n_samples=100, random_state=123)

X_pc = kernelPCA(X,2,'radial', sigma=0.2)

plt.figure(figsize=(8,6))
plt.scatter(X_pc[y==0, 0], X_pc[y==0, 1], color='red', alpha=0.5)
plt.scatter(X_pc[y==1, 0], X_pc[y==1, 1], color='blue', alpha=0.5)

plt.title('First 2 principal components after RBF Kernel PCA')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()
