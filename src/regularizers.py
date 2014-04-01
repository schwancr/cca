import numpy as np

def l1(v, hessian=False):
    return _l1_hyper(v, b=1E-8, hessian=hessian)


def l2(v, hessian=False):
    """
    Compute the l2 norm on a vector v

    Parameters
    ----------
    v : np.ndarray
        vector to compute l2 norm

    Returns
    -------
    val : value of the l2 norm of v
    jac : value of the jacobian of the l2 norm at v
    """

    val = np.sum(np.square(v))
    jac = 2 * v

    if not hessian:
        return val, jac

    else:
        hess = np.eye(jac.shape[0]) * 2
        return val, jac, hess


def _l1_soft_v(v, mu=1E-6, hessian=False):

    temp = np.zeros(v.shape)
    small_ind = np.where(np.abs(v) < mu)
    big_ind = np.where(np.abs(v) >= mu)

    temp[small_ind] = np.square(v[small_ind]) / 2. / mu
    temp[big_ind] = np.abs(v[big_ind]) - mu / 2.

    print v[temp.argmax()], temp.max()
    val = np.sum(temp)

    jac = temp
    jac[small_ind] = 2. * v[small_ind] / mu
    jac[big_ind] = np.sign(v[big_ind]) 

    if not hessian:
        return val, jac

    else:
        hess = np.zeros((jac.shape[0], jac.shape[0]))
        hess[small_ind, small_ind] = 2. / mu
        return val, jac, hess


def _l1_hyper(v, b=1E-7, hessian=False):
    v2 = np.square(v)

    val = np.sum(np.power((v2 + b**2), 0.5) - b)
    jac = v * np.power((v2 + b**2), -0.5)

    if not hessian:
        return val, jac
    
    else:
        hess = np.zeros((jac.shape[0], jac.shape[0]))
        diag_ary = np.power((v2 + b**2), -0.5) * (1 - 0.5 * v2 / (v2 + b**2))
        ind = np.arange(jac.shape[0])
        hess[ind, ind] = diag_ary.flatten()

        return val, jac, hess


def _is_valid_reg(f):
    """
    internal function to check whether a custom function is a good regularizer
    """

    test = np.random.random(1000).reshape((1000, 1))

    result = f(test)
    is_valid = True

    if len(result) != 2:
        print "regularizer must return two items"
        is_valid = False

    else:
        res0, res1 = result

        if not isinstance(res1, np.ndarray):
            print "regularizer must return jacobian as second item"
            is_valid = False

        else:
            if res1.shape != test.shape:
                print "jacobian should be the same shape as the input"
                is_valid = False

        try:
            res0 = np.float(res0)
        except:
            print "first returned item should be a scalar"
            is_valid = False

    
    # now check that a hessian is available
    try:
        result = f(test, hessian=True)
        is_valid = True
    
    except:
        print "regularizer needs to be able to take hessian as a kwarg"
        is_valid = False
        return is_valid

    if len(result) != 3:
        print "regularizer needs to return the hessian when passed hessian=True"
        is_valid = False

    else:
        res0, res1, res2 = result

        if not isinstance(res1, np.ndarray) or not isinstance(res2, np.ndarray):
            print "regularizer must return a number and two arrays when given hessian=True"
            is_valid = False

        else:
            if res1.shape != test.shape:
                print "jacobian should be the same shape as the input"
                is_valid = False

            if res2.shape[0] != test.shape[0] or res2.shape[1] != test.shape[0]:
                print "hessian should have dimensions equal to the length of the input"
                is_valid = False

        try:
            res0 = np.float(res0)
        except:
            print "first returned item should be a scalar"
            is_valid = False

    return is_valid


def _l1_alpha(v, alpha=1E2, hessian=False):
    """
    Compute the approximate l1 norm on vector

    Parameters
    ----------
    v : np.ndarray
        vector to compute l1 norm

    Returns
    -------
    val : value of the l1 norm of v
    jac : value of the jacobian of the approximate l1 norm at v
    """

    temp = 1. / alpha * (np.log(1. + np.exp(- alpha * v)) + np.log(1. + np.exp(alpha * v)))
    ind = np.where(np.isinf(temp) + np.isnan(temp))
    print v[ind]
    temp[ind] = np.abs(v[ind])
    val = temp.sum()

    #print np.abs(v).sum(), np.square(v).sum()

    jac = 1. / (1. + np.exp(- alpha * v)) - 1. / (1. + np.exp(alpha * v))
    ind = np.where(np.isinf(jac) + np.isnan(jac))
    jac[ind] = np.sign(v[ind])

    if not hessian:
        return val, jac
    
    else:
        hess = np.zeros((jac.shape[0], jac.shape[0]))

        return val, jac, hess
