import numpy as np
from scipy.optimize import root, fsolve, fmin_ncg
from mdtraj import io
import pickle
import regularizers as reg
class CCA(object):
    def __init__(self, regularization=None, regularization_strength=0.0,
                 method='root'):
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
        method : {'root', 'ncg'}
            method to solve the CCA prolbem:
                - root : scipy.optimize.root with default parameters
                - ncg : scipy.fmin_ncg (newton conjugate gradient method)
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

        if method.lower() in ['root', 'ncg']:
            self._method = method.lower()
        
        else:
            self._method = 'root'
            print "unknown method (%s), using 'root' instead"

        self.eta = float(regularization_strength)
        self._has_solution = False


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
        
        """
        a = a.flatten()
        if a.shape[0] != M.shape[1]:
            raise Exception("The number of columns in M (%d) should match the "
                "number of points in a (%d)", M.shape[1], a.shape[0])

        M = M - M.mean(1, keepdims=True)        
        a = a - a.mean()
        a = a / a.std()  # normalize a
        np.reshape(a, (-1, 1))

        #self._sol = self._solver(M, a)
        #self.v = self._sol.x
        self.v = self._solver(M, a)

        #if not self._sol.success:
        #    print "error when computing the root (%s)" % self._sol.message
        self._has_solution = True

        return self


    def _solver(self, M, a):
        
        if self._method == 'root':
            return self._root(M, a)

        elif self._method == 'ncg':
            return self._max_ncg2(M, a)

        else:
            raise Exception("bad value for method")


    def _root(self, M, a):
        """
        private method for solving the CCA problem by using a root-finder on
        the derivative of the lagrangian.
        """

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
            print np.linalg.norm(left - right)
            return (left - right).flatten()
            
        #v0 = np.ones((M.shape[0], 1), dtype=np.float)
        v0 = np.random.normal(size=M.shape[0]).reshape((-1, 1)).astype(np.float64)
        v0 = np.ones(M.shape[0]).reshape((-1, 1)) * 1E-5

        return root(_func, x0=v0).x


    def _max_ncg(self, M, a):
        """
        private method for solving the CCA problem by using the Newton Conjugate Gradient
        method using the lagrangian and its first and second derivatives
        """

        Ma = M.dot(a).reshape((-1, 1))
        sigma = M.dot(M.T) / np.float(M.shape[1])  # could do - 1
        SIGMA_HACKS = None
        HACKS = None

        def _func(X):
            """
            X is the concatenation of v with \lambda: [v_1, v_2, ..., v_d, \lambda]
            """
            v = X[:-1].reshape((-1, 1))
            lam = X[-1]

            sigma_v = sigma.dot(v)
            stdev = np.float(np.sqrt(v.T.dot(sigma_v)))
            #SIGMA_HACKS = [sigma_v, stdev]
            # contains sigma_v and stdev
            #HACKS = self.regularizer(v, hessian=True)
            # contains f(v), J[f(v)], H[f(v)]
    
            fval, fjac = self.regularizer(v)

            vMa = v.T.dot(Ma)
            temp = vMa / M.shape[1] + lam * (stdev + self.eta * fval - 1)

            print "objective: %.4E - vMa: %.4E - lambda: %.4E (%.4E) - f(v): %.4E" % (np.float(temp * -1), vMa, lam, vMa / M.shape[1] / (1 - fval + v.T.dot(fjac)), fval)
            return np.float(temp * -1)


        def _func_jac(X):
            v = X[:-1].reshape((-1, 1))
            lam = np.float(X[-1])

            #sigma_v, stdev = SIGMA_HACKS
            sigma_v = sigma.dot(v)
            stdev = np.sqrt(v.T.dot(sigma_v))

            fval, fjac = self.regularizer(v)

            temp = np.zeros(X.shape[0])

            temp[:-1] = np.reshape(Ma / M.shape[1] + lam * (sigma_v / stdev + self.eta * fjac.reshape((-1, 1))), (-1,))
            temp[-1] = stdev + self.eta * fval - 1

            return temp * -1

        
        def _func_hess(X):
            v = X[:-1].reshape((-1, 1))
            lam = X[-1]

            #sigma_v, stdev = SIGMA_HACKS
            #fval, fjac, fhess = HACKS
            sigma_v = sigma.dot(v)
            stdev = np.sqrt(v.T.dot(sigma_v))

            fval, fjac, fhess = self.regularizer(v, hessian=True)
        
            temp = np.zeros((X.shape[0], X.shape[0]))
            
            # d^2 / dv^2
            temp[:-1, :-1] = lam * (sigma / stdev).dot(np.eye(sigma.shape[0]) - np.outer(v, sigma_v)) + lam * self.eta * fhess
            # d^2 / dlam dv
            temp[:-1, 0:1] = sigma_v / stdev + self.eta * fjac.reshape((-1, 1))
            # d^2 / dv dlam
            temp[0, :-1] = temp[:-1, 0]
            temp[-1, -1] = 0.0

            return temp * -1


        #v0 = np.ones((M.shape[0], 1), dtype=np.float)
        v0 = np.random.normal(size=M.shape[0]).reshape((-1, 1)).astype(np.float64)
        v0 = np.ones(M.shape[0]).reshape((-1, 1)) * 1E-5
        v0 = np.concatenate((v0, [[1]]))
        _func(v0)

        result = fmin_ncg(_func, x0=v0, fprime=_func_jac, fhess=_func_hess)

        return result


    def _max_ncg2(self, M, a):
        """
        private method for solving the CCA problem by using the Newton Conjugate Gradient
        method using the lagrangian and its first and second derivatives
        """

        Ma = M.dot(a).reshape((-1, 1))
        sigma = M.dot(M.T) / np.float(M.shape[1])  # could do - 1
        SIGMA_HACKS = None
        HACKS = None

        def _func(X):
            """
            X is the concatenation of v with \lambda: [v_1, v_2, ..., v_d, \lambda]
            """
            v = X[:].reshape((-1, 1))

            sigma_v = sigma.dot(v)
            stdev = np.float(np.sqrt(v.T.dot(sigma_v)))
            #SIGMA_HACKS = [sigma_v, stdev]
            # contains sigma_v and stdev
            #HACKS = self.regularizer(v, hessian=True)
            # contains f(v), J[f(v)], H[f(v)]
    
            fval, fjac = self.regularizer(v)

            vMa = v.T.dot(Ma)
            lam = -1 * vMa / M.shape[1] / (1.0 - self.eta * fval + self.eta * v.T.dot(fjac))
            temp = vMa / M.shape[1] + lam * (stdev + self.eta * fval - 1)

            print "objective: %.4E - vMa: %.4E - lambda: %.4E - f(v): %.4E - stdev: %.4E" % (np.float(temp * -1), vMa / M.shape[1], lam, fval, stdev)
            return np.float(temp * -1)


        def _func_jac(X):
            v = X[:].reshape((-1, 1))
            vMa = v.T.dot(Ma)

            #sigma_v, stdev = SIGMA_HACKS
            sigma_v = sigma.dot(v)
            stdev = np.sqrt(v.T.dot(sigma_v))

            fval, fjac = self.regularizer(v)

            lam = -1 * vMa / M.shape[1] / (1.0 - self.eta * fval + self.eta * v.T.dot(fjac))
            temp = np.zeros(X.shape[0])

            temp[:] = np.reshape(Ma / M.shape[1] + lam * (sigma_v / stdev + self.eta * fjac.reshape((-1, 1))), (-1,))

            return temp * -1

        
        def _func_hess(X):
            v = X[:].reshape((-1, 1))
            vMa = v.T.dot(Ma)

            #sigma_v, stdev = SIGMA_HACKS
            #fval, fjac, fhess = HACKS
            sigma_v = sigma.dot(v)
            stdev = np.sqrt(v.T.dot(sigma_v))

            fval, fjac, fhess = self.regularizer(v, hessian=True)
        
            lam = -1 * vMa / M.shape[1] / (1.0 - self.eta * fval + self.eta * v.T.dot(fjac))
            temp = np.zeros((X.shape[0], X.shape[0]))
            
            # d^2 / dv^2
            temp = lam * (sigma / stdev).dot(np.eye(sigma.shape[0]) - np.outer(v, sigma_v)) + lam * self.eta * fhess

            return temp * -1


        #v0 = np.ones((M.shape[0], 1), dtype=np.float)
        v0 = np.random.normal(size=M.shape[0]).reshape((-1, 1)).astype(np.float64)

        fval, fjac = self.regularizer(v0)
        nrm = np.sqrt(v0.T.dot(sigma.dot(v0))) + self.eta * fval
        v0 /= np.float(nrm)
        nrm = np.sqrt(v0.T.dot(sigma.dot(v0))) + self.eta * fval
        print np.float(nrm)

        v0 *= np.sign(v0.T.dot(Ma))

        result = fmin_ncg(_func, x0=v0, fprime=_func_jac, fhess=_func_hess)

        return result


    def predict(self, M):
        """
        predict the value of the output variable given the solution.

        Parameters
        ----------
        M : np.ndarray 
            Data to predict. This can be the training data or other data. The shape
            should be the same as passed to fit (n_features, n_points)

        Returns
        -------
        a : np.ndarray
            output prediction. The shape is (n_points,). Remember that we do not
            mean subtract this data so keep that in mind
        """
    
        if not self._has_solution:
            raise Exception("you must run fit() first in order to use this method")


        if not self.v.shape[0] == M.shape[0]:
            raise Exception("Data (%d) is the wrong dimension (%d)" % (M.shape[0], self.v.shape[0]))


        a = self.v.T.dot(M).flatten()

        return a


    def evaluate(self, a_pred, a_ref):
        """
        Evaluate the result of a prediction on test data.

        Paramaters
        ----------
        a_pred : np.ndarray
            predicted data. shape: (n_points,)
        a_ref : np.ndarray
            actual data. shape: (n_points,)

        Returns
        -------
        corr : float
            This is the score of the predicted data. In this case, we're just 
            calculating the covariance, since this is what CCA is attempting
            to maximize.
        """

        a_pred = a_pred - a_pred.mean()
        a_ref = a_ref - a_ref.mean()

        a_pred = a_pred / a_pred.std()
        a_ref = a_ref / a_ref.std()

        return np.dot(a, a_ref)

    
    def predict_evaluate(self, M, a):
        """
        Predict the a-values for test data and score it compared to the actual
        values given by a.

        Parameters
        ----------
        M : np.ndarray
            test data to evaluate. This should have a shape corresponding to:
            (n_features, n_points)
        a : np.ndarray
            the actual a-values for this test data. This should have a shape
            corresponding to: (n_points,)

        Returns
        -------
        corr : float
            This is the score of the predicted data. In this case, we're just 
            calculating the covariance, since this is what CCA is attempting
            to maximize.
        """

        a_pred = self.predict(M)
        return self.evaluate(a_pred, a)


    def save(self, filename):
        """
        Save the results and everything needed to use this object again.

        Parameters
        ----------
        filename : str
            filename to save the data to. Will use mdtraj.io.saveh 

        Returns
        -------
        filename : str
            the same filename in case you want it back.
        """

        kwargs = {}
        kwargs['regularizer'] = np.array([pickle.dumps(self.regularizer)])
        kwargs['eta'] = np.array([self.eta])

        print 'has_solution?', self._has_solution
        if self._has_solution:
            kwargs['sol'] = self.v

        io.saveh(filename, **kwargs)

        return filename


    @classmethod
    def load(cls, filename):
        """
        Load a previously saved CCA object

        Parameters
        ----------
        filename : str
            filename to load data from

        Returns
        -------
        cca_object : CCA
            loaded cca_object
        """
            
        filehandler = io.loadh(filename)

        regularizer = pickle.loads(filehandler['regularizer'][0])
        eta = filehandler['eta'][0]

        cca_object = cls(regularization=regularizer, regularization_strength=eta)

        if 'sol' in filehandler.keys():
            cca_object.v = filehandler['sol']
            cca_object._has_solution = True

        return cca_object
