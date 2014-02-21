

import numpy as np

def l1_alpha(v, alpha=1E2):
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
    return val, jac


def l2(v):
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

    return val, jac


def l1(v, mu=1E-4, eps=1E-4):

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

    val += eps * np.square(v).sum()
    jac += eps * v

    #print val, jac.min(), jac.max(), np.abs(jac).min()

    return val, jac


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

    return is_valid
