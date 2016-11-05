# A Nonnegative Matrix Factorization Class implemented in Python

##### A simple NMF class created using the NumPy library in Python.
<br>
Original matrix V:
<table border="1" class="dataframe">   <tbody>     <tr>       <td>10.1</td>       <td>9.9</td>       <td>8.8</td>       <td>7.7</td>       <td>6.6</td>     </tr>     <tr>       <td>5.5</td>       <td>4.4</td>       <td>3.3</td>       <td>2.2</td>       <td>1.1</td>     </tr>     <tr>       <td>1.1</td>       <td>2.2</td>       <td>3.3</td>       <td>4.4</td>       <td>5.5</td>     </tr>     <tr>       <td>6.6</td>       <td>7.7</td>       <td>8.8</td>       <td>9.9</td>       <td>10.1</td>     </tr>   </tbody> </table>

Matrix V is factorized into matrices W and H with k=5 (no dimensional reduction):
<table border="1" class="dataframe">   <tbody>     <tr>       <td>91.801925</td>       <td>5.610152e+03</td>       <td>0.0</td>       <td>0.0</td>       <td>0.001766</td>     </tr>     <tr>       <td>0.000001</td>       <td>2.204049e+03</td>       <td>0.0</td>       <td>0.0</td>       <td>0.011056</td>     </tr>     <tr>       <td>97.428762</td>       <td>5.627143e-08</td>       <td>0.0</td>       <td>0.0</td>       <td>0.005845</td>     </tr>     <tr>       <td>162.839004</td>       <td>1.503765e+02</td>       <td>0.0</td>       <td>0.0</td>       <td>0.042407</td>     </tr>   </tbody> </table>

<table border="1" class="dataframe">   <tbody>     <tr>       <td>0.001730</td>       <td>0.016409</td>       <td>0.028948</td>       <td>0.041486</td>       <td>0.053577</td>     </tr>     <tr>       <td>0.001733</td>       <td>0.001457</td>       <td>0.001062</td>       <td>0.000667</td>       <td>0.000296</td>     </tr>     <tr>       <td>0.000000</td>       <td>0.000000</td>       <td>0.000000</td>       <td>0.000000</td>       <td>0.000000</td>     </tr>     <tr>       <td>0.000000</td>       <td>0.000000</td>       <td>0.000000</td>       <td>0.000000</td>       <td>0.000000</td>     </tr>     <tr>       <td>143.676185</td>       <td>112.871290</td>       <td>92.063006</td>       <td>71.254722</td>       <td>32.215719</td>     </tr>   </tbody> </table>

The reconstruction of V from W and H has mean-squared-error of 0.00311:
<table border="1" class="dataframe">   <tbody>     <tr>       <td>10.135106</td>       <td>9.877642</td>       <td>8.777642</td>       <td>7.677642</td>       <td>6.635106</td>     </tr>     <tr>       <td>5.408245</td>       <td>4.458434</td>       <td>3.358434</td>       <td>2.258434</td>       <td>1.008245</td>     </tr>     <tr>       <td>1.008245</td>       <td>2.258434</td>       <td>3.358434</td>       <td>4.458434</td>       <td>5.408245</td>     </tr>     <tr>       <td>6.635106</td>       <td>7.677642</td>       <td>8.777642</td>       <td>9.877642</td>       <td>10.135106</td>     </tr>   </tbody> </table>

V is again factorized into matrices W and H with k=3. Notice the dimensional reduction (V is approximated with fewer columns in W and fewer rows in H):
<table border="1" class="dataframe">   <tbody>     <tr>       <td>5.148067</td>       <td>0.000000</td>       <td>1.691900e-14</td>     </tr>     <tr>       <td>2.303943</td>       <td>0.036165</td>       <td>0.000000e+00</td>     </tr>     <tr>       <td>1.109006</td>       <td>0.000000</td>       <td>3.452754e-02</td>     </tr>     <tr>       <td>4.046159</td>       <td>0.000000</td>       <td>4.404242e-02</td>     </tr>   </tbody> </table>

<table border="1" class="dataframe">   <tbody>     <tr>       <td>1.963072</td>       <td>1.917316e+00</td>       <td>1.712553</td>       <td>1.507789</td>       <td>1.272087</td>     </tr>     <tr>       <td>27.020781</td>       <td>0.000000e+00</td>       <td>0.000000</td>       <td>0.000000</td>       <td>0.000000</td>     </tr>     <tr>       <td>0.000000</td>       <td>2.674694e-14</td>       <td>41.750244</td>       <td>83.500488</td>       <td>114.733097</td>     </tr>   </tbody> </table>

Reconstruction of V from W and H with k=3 has mean-squared-error of 0.42197:
<table border="1" class="dataframe">   <tbody>     <tr>       <td>10.106026</td>       <td>9.870472</td>       <td>8.816336</td>       <td>7.762200</td>       <td>6.548787</td>     </tr>     <tr>       <td>5.500000</td>       <td>4.417387</td>       <td>3.945623</td>       <td>3.473860</td>       <td>2.930815</td>     </tr>     <tr>       <td>2.177059</td>       <td>2.126316</td>       <td>3.340765</td>       <td>4.555214</td>       <td>5.372204</td>     </tr>     <tr>       <td>7.942900</td>       <td>7.757765</td>       <td>8.768042</td>       <td>9.778318</td>       <td>10.200187</td>     </tr>   </tbody> </table>

<img src="images/python.png" width="120">
<img src="images/atom.png" width="120">
<img src="images/linux.png" width="120">

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
        self._create_mat()

    def _create_mat(self):
        """
        Initialize the W and H matrices of appropriate dimensions
        Random floats will be optimized by fit function
        """
        self.W = np.random.uniform(size=(self.n, self.k))
        self.H = np.random.uniform(size=(self.k, self.m))

    def fit(self):
        """
        Optimize the values of H and W using alternating least squared error
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
