

import numpy as np
from scipy.optimize import root
import regularizers as reg
class CCA(object):
    def __init__(self, regularization=None, regularization_strength=0.0):
        """
        This is a particular implementation of CCA, where one space is one
        dimensional. The problem is then phrased to ask for the projection
        in the higher dimensional space that is maximally correlated with
        the one-dimensional data.

        If X_t is the high dimensional data and a_t is the one-dimensional
        data paired with each point X_t, then we want to find v such that:
        
        max_v corr(v.T X_t, a_t) / (std(v.T X_t) * std(a_t))

        This is just the normalized correlation coefficient.

        In addition, it is often useful to do some sort of regularization
        to avoid overfitting the data with a complicated solution, v.

        As such, you can modify the denominator to instead be:

        std(v.T X_t) * std(a_t) + \eta f(v)

        Where f(v) can be any differentiable penalty function of v. The \eta
        parameter tunes the scale of the penalty. 

        Parameters
        ----------
        regularization : {'l1', 'l2', 'none'} or function
            type of regularization to use:
                - l1 : approximate l1 regularization that penalizes the sum
                    of the absolute values of the entries in the solution
                - l2 : l2 regularization penalizes the sum of the squares of 
                    the values of the entries in the solution
                - None : no regularization
                - function : custom function that calculates the penalty. This
                    should return the function value as well as the jacobian
        regularization_strength : float, optional
            regularization strength. You should probably fit this with
            cross-validation
        """


        if isinstance(regularization, str):
            regularization = regularization.lower()
            if regularization == 'l1':
                self.regularizer = reg.l1
            elif regularization == 'l2':
                self.regularizer = reg.l2
            else:
                self.regularizer = None

        elif regularization is None:
            self.regularizer = None

        else: 
            # then we have a custom function. 
            if not reg._is_valid_reg(regularization):
                raise Exception("regularization function must return two "
                    "objects: (1) value of penalty (2) value of the "
                    "jacobian.")
            
            self.regularizer = regularization

        self.eta = float(regularization_strength)


    def fit(self, M, a):
        """
        calculate the best projection in M that correlates with the data, a
    
        Parameters
        ----------
        M : np.ndarray
            matrix of high-dimensional data, with features in rows and the
            datapoints in columns (n_features, n_points)
        a : np.ndarray
            one dimensional array corresponding to the scalar value for each
            column of data (n_points,)
        
        self.M = data
        self.a = a.flatten()

        if self.M.shape[1] != self.a.shape[0]:
            raise Exception("Your data should have points stored as columns.")

        """
        a = a.flatten()
        if a.shape[0] != M.shape[1]:
            raise Exception("The number of columns in M (%d) should match the "
                "number of points in a (%d)", M.shape[1], a.shape[0])

        M = M - M.mean(1, keepdims=True)        
        a = a - a.mean()
        a = a / a.std()  # normalize a
        np.reshape(a, (-1, 1))

        Ma = M.dot(a).reshape((-1, 1))
        sigma = M.dot(M.T) / np.float(M.shape[1])  # could do - 1

        def _func(v):

            v = v.reshape((-1, 1))
            sigma_v = sigma.dot(v)
            stdev = np.float(np.sqrt(v.T.dot(sigma_v)))

            # start accumulating the left and right terms
            left = 1.
            right = sigma_v / stdev

            if not self.regularizer is None:
                val, jac = self.regularizer(v)

                left += 2. * self.eta * np.float(v.T.dot(jac) - 0.5 * val)
                right += 2. * self.eta * jac

            left = Ma * left
            right = np.float(v.T.dot(Ma)) * right
            # all done.
            print left.shape, right.shape
            print left - right
            return (left - right).flatten()
            
        v0 = np.ones((M.shape[0], 1), dtype=np.float)
        print sigma.shape, v0.shape

        self._sol = root(_func, x0=v0)

        self.v = self._sol.x

        if not self._sol.success:
            print "error when computing the root (%s)" % self._sol.message
        
        return self
