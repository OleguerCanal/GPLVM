import numpy as np

def get_active_set(K, noise_model_var, size):
    n_points = K.shape[0]
    beta = np.empty(n_points)
    beta.fill(noise_model_var)
    diagonals = np.empty((size+1, n_points))
    diagonals[0] = np.diagonal(K)
    J = set({i for i in range(n_points)})
    I = set({})
    M = np.zeros((n_points, n_points))

    for i in range(1, size+1):
        delta_H = np.empty(n_points)
        delta_H.fill(-float('inf'))
        v = np.empty(n_points)
        for n in J:
            v[n] = 1. / (diagonals[i-1, n] + 1. / beta[n])
            delta_H[n] = -0.5 * np.log(1 - v[n] * diagonals[i - 1, n]) 
        n_i = np.argmax(delta_H)
        beta[n_i] = v[n_i] / (1 - v[n_i] * diagonals[i - 1, n_i])
        s = K[n_i] - M[:, n_i].dot(M) # TODO: dot product until M[:,i] and append zeros
        s = np.matrix(s).T # To column vector
        diagonals[i] = diagonals[i - 1] - v[n_i] * np.diagonal(s.dot(s.T))
        M[i] = np.sqrt(v[n_i]) * s.T # Appending row to matrix
        I.add(n_i)
        J.remove(n_i)
    return list(I), list(J)





if __name__ == "__main__":
    I, J = get_active_set(np.identity(10), 0.1, size=4)
    print(I)
    print(J)
