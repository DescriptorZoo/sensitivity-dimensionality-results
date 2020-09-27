#
# License is MIT (2020)
# Author: Berk Onat
# This package is part of https://github.com/DescriptorZoo
# 
#
import os
import sys
import math
import argparse
import numpy as np
import h5py
import copy
import types
import time

import scipy.sparse
import scipy.linalg
import scipy.integrate

from numpy.linalg import svd as svd
from scipy.linalg import pinv2
from scipy.linalg import pinv
from scipy.linalg import svd as s_svd
from sklearn.utils.extmath import randomized_svd

from numba import jit
from numba import njit, prange

PI = 3.14159265359

#@jit(nopython=True, parallel=True)
@njit(fastmath = True, nogil = True, cache = True)
def _dot(A, B):
    return np.dot(A, B)

#@jit(nopython=True, parallel=True)
@njit(fastmath = True, nogil = True, cache = True)
def _pinv(A):
    return pinv(A)

#@jit(nopython=True, parallel=True)
@njit(fastmath = True, nogil = True, cache = True)
def _sum(A):
    return np.sum(A)

#@jit(nopython=True, parallel=True)
@njit(fastmath = True, nogil = True, cache = True)
def _svd(A):
    return svd(A, full_matrices=False)

#@jit(nopython=True, parallel=True)
@njit(fastmath = True, nogil = True, cache = True)
def _argmax(A):
    return np.argmax(A)

def fmt_table(table):
    return '\n'.join([' '.join(['{: ^6}'.format(x) for x in row]) for row in table])

class FPCUR(object):
    """
    """

    def __init__(self, data, pk=-1, k=1, N=0, col_weights=None, error_print=0, nonan=False):
        """
        Parameters
        ----------
        """
        (rows, cols) = data.shape
        self.data = data
        self.pinv_data = None
        self.XXp = None
        self.norm_data = None
        self.norm1_data = None
        self.norm2_data = None
        self.norm3_data = None
        self._rows = rows
        self._cols = cols
        self.sk = k
        self.nonan = nonan
        self._col_weights = col_weights

        if N < 1:
            self._N = self._cols
        else:
            self._N = N

        if error_print < 1:
            self.error_print = self._N
        else:
            self.error_print = int(error_print)

    def importance_score(self):
        dsquare = self.svd_V[:,:]**2
        pi_cols = np.sum(dsquare[range(self.sk),:], axis=0)
        if self._col_weights is not None:
            pi_cols *= self._col_weights
        return pi_cols

    def max_score(self):
        pi_cols = self.importance_score()
        pi_max = np.argmax(pi_cols, axis=0)
        return pi_max, pi_cols

    def ortho_vec(self, X, X_l, X_l_norm2):
        return X - X_l * _dot(X_l, X) / X_l_norm2

    def orthogonalize(self, col=0):
        if self.nonan:
            print('[CUR:] Changing NaN and Inf before orthogonalization.')
            self._X = np.nan_to_num(self._X)
        l_col = self._X[:, col]
        l_col_norm2 = np.linalg.norm(l_col) ** 2
        self._X = np.apply_along_axis(self.ortho_vec, 0, self._X, l_col, l_col_norm2)

    def computeUCR(self):
        self._ccnt = np.ones(len(self._cid))

        if self.pinv_data is None:
            start = time.time()
            self.pinv_data = pinv(self.data[:,:])
            end = time.time()
            print('Elapsed time for pseudo-inverse of data : ' + str(end - start))
            sys.stdout.flush()

        if self.XXp is None: # XX*
            start = time.time()
            self.XXp = _dot(self.data[:,:], self.pinv_data)
            end = time.time()
            print('Elapsed time for XXp multiplication: ' + str(end - start))
            sys.stdout.flush()

        start = time.time()
        self._C = self.data[:, self._cid] # C
        self._R = self.data # R = X
        self._U = _dot(pinv2(self._C), self.XXp) # C*XX*

        if self.norm_data is None:
            self.norm_data = np.linalg.norm(self.data)

        self.error = np.linalg.norm(self.data - _dot(
            _dot(self._C, self._U), self._R)) / self.norm_data
        end = time.time()
        print('Elapsed time for error calculation with CUR : ' + str(end - start))
        sys.stdout.flush()

    def reset(self):
        self._first_feature_no = None
        self.pinv_data = None
        self.XXp = None
        self.norm_data = None
        self.norm1_data = None
        self.norm2_data = None
        self.norm3_data = None
        self._cid = []
        

    def factorize(self):
        """ Factorize s.t. CUR with fingerprint feature selection = data

        Updated Values
        --------------
        ._C (.U): updated values for C.
        ._U (.S): updated values for U.
        ._R (.V): updated values for R.
        """

        self._X = self.data.copy()
        self._colid = []
        all_errors = []
        all_cids = []

        print('[CUR:] Selecting Columns for C Matrix from X data ...')
        for s in range(self._N):
            print('[CUR:] Cycle No : ' + str(s+1))
            sys.stdout.flush()
            print('[CUR:] SVD Decomposition ...')
            sys.stdout.flush()
            start = time.time()
            if self.nonan:
                print('[CUR:] Changing NaN and Inf before SVD Decomposition.')
                self._X = np.nan_to_num(self._X)
            try:
                self.svd_U, self.svd_S, self.svd_V = randomized_svd(self._X, 
                                                                    n_components=500,
                                                                    n_iter=10,
                                                                    random_state=None)
            except ValueError:
                self.svd_U, self.svd_S, self.svd_V = s_svd(self._X, full_matrices=True)
            end = time.time()
            print('Elapsed time for SVD Decomposition: ' + str(end - start))
            print('[CUR:] Importance Scoring ...')
            sys.stdout.flush()
            pi_cols = self.importance_score()
            print('[CUR:] Length of scores : ',len(pi_cols))
            for ci in range(len(pi_cols)):
                pi_max = np.nanargmax(pi_cols, axis=0)
                if pi_max not in self._colid:
                    print('[CUR:] non-recurrence col cycle:',ci,pi_max)
                    break
                else:
                    pi_cols[pi_max] = np.nan
            print('[CUR:] Orthogonalization ...')
            sys.stdout.flush()
            self.orthogonalize(col=pi_max)
            print('[CUR:] Selected Column Number : ' + str(pi_max))
            sys.stdout.flush()
            self._colid.append(pi_max)
            self._cid = np.array(self._colid, np.int32)
            if s == 0 or s == self._N-1 or s % self.error_print == 0:
                self._ccnt = np.ones(len(self._cid))
                print('[CUR:] Non-recurrent Selection size of C : ' + str(len(list(set([c for c in self._cid])))))
                print('[CUR:] Applying CUR Decomposition with Selected C and R=X ...')
                sys.stdout.flush()
                self.computeUCR()
                print('[CUR:] Error between X and X` = CUR : ' + str(self.error))
                sys.stdout.flush()
                all_cids.append(self._colid[:])
                all_errors.append([self.error])

        return all_cids, all_errors

def read_h5(filename):
    with h5py.File(filename, 'r') as hf:
        data = hf['dataset'][:]
    return np.array(data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generates SF descriptors and optionally apply CUR.")
    parser.add_argument("filename", 
            help="File path and name for the database to work.")
    parser.add_argument("-k", "--knum", type=int, required=False, default=1, 
            help="'k' number for CUR decomposition.")
    parser.add_argument("-c", "--colnum", type=int, required=False, default=1, 
            help="Number of features to select for CUR decomposition.")
    parser.add_argument("-p", "--error_print", type=int, required=False, default=0, 
            help="'k' number for CUR decomposition.")
    parser.add_argument("-noh5", "--noh5", action="store_true",
            help="H5 input data file.")
    parser.add_argument("-nonan", "--nonan", action="store_true",
            help="H5 input data file.")
    args = parser.parse_args()

    print('Reading DATABASE FILE : ' + str(args.filename))
    if args.noh5:
        data = np.loadtxt(args.filename)
    else:
        data = read_h5(args.filename)
    
    print('----- Checking NaN and Inf values in Dataset -----')
    hasnaninf = False
    if np.isnan(data).any():
        hasnaninf = True
        if args.ignorenans:
            print(' Warning : Data has NaN values! Ignoring NaNs and set them to zero!')
        else:
            print(' Error : Data has NaN values! Can not continue. Exiting.')
            exit(0)
    if np.isinf(data).any():
        hasnaninf = True
        if args.ignorenans:
            print(' Warning : Data has Inf values! Setting Infs to very large values!')
        else:
            print(' Error : Data has Inf values! Can not continue. Exiting.')
            exit(0)
    if hasnaninf is False:
        print(' There are no NaN or Inf values in dataset. Good to go.')
    else:
        if args.ignorenans:
            print(' Ignoring NaN or Inf values in dataset and '
                  'setting them to zero and very large number, respectively.')
            data = np.nan_to_num(data)
    print('----- CUR Decomposition for Dataset -----')
    print('CUR Parameters :')
    print('  k=' + str(args.knum))
    print('  N=' + str(args.colnum))
    print('  Database size : ' + ' x '.join([ str(s) for s in data.shape]))

    the_CUR = FPCUR(data, k=int(args.knum), N=int(args.colnum), 
                    error_print=int(args.error_print), nonan=args.nonan)

    selections, cur_errors = the_CUR.factorize()

    print('The selected columns : ' + ','.join([ str(ci) for ci in the_CUR._cid]))
    for reci, rec in enumerate(selections):
        print('NumCols: ' + str(len(rec)) + ' Errors: ' + ','.join([str(r) for r in cur_errors[reci]]))
    sys.stdout.flush()

