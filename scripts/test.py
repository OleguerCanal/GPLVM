import numpy as np
from numpy.core.umath_tests import inner1d
import time

m, n = 1000, 5000

a = np.random.rand(m, n)
b = np.random.rand(n, m)

t0 = time.time()
for _ in range(10):
    c = np.trace(np.dot(a, b))
print(time.time() - t0)
print(c)

t1 = time.time()
for _ in range(10):
    cc = np.sum(inner1d(a, b.T))
print(time.time() - t1)
print(cc)