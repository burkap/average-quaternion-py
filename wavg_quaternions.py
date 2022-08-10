#
# Author: B. Burak Payzun
#
# This code is based on Tolga Birdal's MATLAB code based on the following paper:
# 
# https://ntrs.nasa.gov/api/citations/20070017872/downloads/20070017872.pdf
# by F. Landis Markley, Yang Cheng, John L. Crassidis, Yaakov Oshman
#

import numpy as np


def wavg_quaternions(quaternions, weights):
    """
    quaternions:    Mx4 matrix of quaternions
    weights:        Mx1 weights (sums to 1) 

    returns 4x1 average quaternion
    """
    A = np.zeros((4, 4)) # symmetric accumulator matrix

    M = quaternions.shape[0]

    wSum = 0

    for i in range(M):
        q = quaternions[i, :].reshape(-1, 1)
        if q[0] < 0: # handle antipodal configuration
            q = -q
        w_i = weights[i]
        A = w_i * (q * q.T) + A
        wSum += w_i

    A = (1.0 / wSum) * A

    _, eig_vectors = np.linalg.eigh(A)

    # return the eigen vector corresponding to largest eigen value
    return eig_vectors[:, -1]
