

import numpy as np

def l1(v, alpha=1E5):
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

    v = np.abs(v)

    val = np.sum(v + 1. / alpha * np.log(1. + np.exp(- alpha * v)))

    jac = 1. / (1. + np.exp(- alpha * v)) - 1. / (1. + np.exp(alpha * v))

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
