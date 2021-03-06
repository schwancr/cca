
import numpy as np
from scipy.optimize import fsolve
from msmbuilder import io
import argparse
import matplotlib
from matplotlib.pyplot import *
import IPython

parser = argparse.ArgumentParser()
parser.add_argument('-d', dest='data', help='data for each state')
parser.add_argument('-f', dest='eig', help='eigenvector value of each state')

args = parser.parse_args()

M = io.loadh(args.data, 'HB_maps')
if len(M.shape) > 2:
    M = M.reshape((M.shape[0], -1))

M = M - M.mean(0)
print M.shape

eig = io.loadh(args.eig, 'arr_0')

b = eig / np.sqrt(eig.dot(eig) / eig.shape[0])
b = np.reshape(b, (-1, 1))

sigma = M.T.dot(M)
pca_vals, pca_vecs = np.linalg.eig(sigma)

ind = np.where(pca_vals > 1E-8)[0]

pca_vals = pca_vals[ind]
pca_vecs = pca_vecs[:, ind]

M = M.dot(pca_vecs)
sigma = np.diag(pca_vals)

Mb = M.T.dot(b)
MbbM = Mb.dot(Mb.T)

def f(p1):
    
    print p1
    p = np.reshape(p1, (-1, 1))
    denom = p.T.dot(Mb)
    left = Mb / denom
    right = sigma.dot(p)

    return (left - right).flatten()

def fprime(p1):

    p = np.reshape(p1, (-1, 1))
    denom = (p.T.dot(Mb))**2

    return (- MbbM / denom - sigma)

p0 = np.ones(M.shape[1])
p0 = p0 / np.sqrt(p0.T.dot(sigma).dot(p0))

sol = fsolve(f, p0, fprime=fprime)

sol_ref = pca_vecs.dot(sol.reshape((-1, 1)))


IPython.embed()
