# A Nonnegative Matrix Factorization Class implemented in Python

A simple NMF class created using the NumPy library in Python.

<img src="images/python.png" width="120">
<img src="images/atom.png" width="120">

```python
import numpy as np


class NMF(object):
    def __init__(self, V, k, reps):
        """
        Inputs:
        V = Dense numpy matrix to be decomposed
        k = Integer number of latent features to be retained
        reps = Integer number of iterations for optimization for W and H
        """
        self.V = V
        self.k = k
        self.reps = reps
        # n = rows of V and m = columns of V
        self.n = V.shape[0]
        self.m = V.shape[1]
        # Create W and H matrices
        self.create_mat()

    def create_mat(self):
        """
        Initialize the W and H matrices of appropriate dimensions
        Random floats will be optimized by fit function
        """
        self.W = np.random.uniform(size=(self.n, self.k))
        self.H = np.random.uniform(size=(self.k, self.m))

    def fit(self):
        """
        Optimize the values of H and W using least squared error
        Returns: Optimized W and H matrices
        """
        for rep in xrange(self.reps):
            # Find least squares solution for H and set negative values to 0
            self.H = np.linalg.lstsq(self.W, self.V)[0]
            self.H[self.H < 0] = 0
            # Find least squares solution for W and set negative values to 0
            self.W = np.linalg.lstsq(self.H.T, self.V.T)[0].T
            self.W[self.W < 0] = 0
        return self.W, self.H

    def reconstruct(self):
        """
        Returns: Approximation of matrix V from dot product of W and H
        """
        return self.W.dot(self.H)

    def mean_sqr_err(self):
        """
        Returns: Mean squared error of reconstructed approximation of V
        """
        V_recon = self.reconstruct()
        return np.mean(np.square(self.V - V_recon))
```
