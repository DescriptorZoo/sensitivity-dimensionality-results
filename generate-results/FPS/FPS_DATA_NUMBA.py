#
# License is GPL-3 (2019)
# Author: Berk Onat
# This package id part of https://github.com/DescriptorZoo
# 
#
import os
import sys
import math
import numpy as np
import h5py
import copy
import types
import time
import subprocess as sub
import argparse

import scipy.sparse
import scipy.linalg

from numpy.linalg import pinv
from numpy.linalg import svd as svd
from scipy.linalg import pinv2
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
    """Formats table as a string for output."""
    return '\n'.join([' '.join(['{: ^6}'.format(x) for x in row]) for row in table])

class FPS(object):
    """      
    FPS(data, first_selection, seed)
        
    Farthest-Point Sampling (FPS) over the data 
    starting from the first_selection if it is given.
    
    """
    
    def __init__(self, data, first_feature_no=None, 
                 error_print=0, seed=None, selected_cols=[], 
                 col_weights=None, nonan=False, restart_file=None):
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
        self._first_feature_no = first_feature_no
        self._seed = seed
        self.nonan = nonan
        self._col_weights = col_weights
        if restart_file is not None:
            self._restart_file = restart_file
        else:
            self._restart_file = 'restart.h5'

        if error_print < 1:
            self.error_print = self._cols
        else:
            self.error_print = int(error_print)
        
    def apply_fps(self, cols=0, rows=0):
        k = []
        if cols > 0:
            farthests = np.zeros((cols, self._rows))
            farthests[0] = self.data.T[self._first]
            fcols = [0]
            k.append(self._first)
            distances = np.linalg.norm(farthests[0] - self.data.T, axis=1)
            record_nans = np.array([False for d in range(len(distances))])
            for i in range(1, cols):
                distances[record_nans] = np.nan
                pos = np.nanargmax(distances)
                if pos not in k:
                    farthests[i] = self.data.T[pos]
                    k.append(pos)
                    distances = np.minimum(distances, np.linalg.norm(farthests[i] - self.data.T, axis=1))
                    fcols.append(i)
                else:
                    distances[pos] = np.nan
                record_nans[pos] = True
        elif rows > 0:
            farthests = np.zeros((rows, self._cols))
            farthests[0] = self.data[self._first]
            k.append(self._first)
            distances = np.linalg.norm(farthests[0] - self.data, axis=1)
            record_nans = np.array([False for d in range(len(distances))])
            for i in range(1, rows):
                pos = np.nanargmax(distances)
                if pos not in k:
                    farthests[i] = self.data[pos]
                    k.append(pos)
                    distances = np.minimum(distances, np.linalg.norm(farthests[i] - self.data, axis=1))
                    distances[record_nans] = np.nan
                else:
                    record_nans[pos] = True
                    distances[pos] = np.nan
        return k
    
    def computeUCR(self):
        self._ccnt = np.ones(len(self._cid))

        if self.pinv_data is None:
            start = time.time()
            self.pinv_data = pinv2(self.data[:,:])
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
        self._seed = None
        self._first = None
        self.pinv_data = None
        self.XXp = None
        self.norm_data = None
        self.norm1_data = None
        self.norm2_data = None
        self.norm3_data = None
        self._cid = []
        
    def select(self, cols=0, rows=0, freq=None, restart_cols=None, 
               restart_errs=None, restart_cycles=None):
        restart=False
        if restart_cols is not None:
            restart=True
            all_cids = restart_cols.tolist()
            self._selected = all_cids[:]
        else:
            all_cids = []
        if restart_errs is not None:
            self._colerr = [i for i in restart_errs]
            all_errors = [i for i in restart_errs]
        else:
            self._colerr = []
            all_errors = []
        if restart_cycles is not None:
            self.cycles = restart_cycles.tolist()
            skips = len(self.cycles)
        else:
            self.cycles = []
            skips = 0

        if freq is None:
            freq = int(cols)
        else:
            freq = int(freq)
        if cols < 0:
            cols = self._cols
        if rows < 0:
            rows = self._rows
        if(rows > 0 or cols > 0) and not restart:
            print('[FPS:] Selecting features from X data ...')
            if self._first_feature_no is not None:
                self._first = int(self._first_feature_no)
            else:
                if cols > 0 :
                    feature_length = self._cols
                    dataset_length = self._rows
                elif rows > 0 :
                    feature_length = self._rows
                    dataset_length = self._cols
                if self._seed is None:
                    best = 0
                    last_nonzeros = 0
                    self._first = None
                    for attempt in range(10):
                        p = sub.Popen(['od','-A','n','-l','-N','4','/dev/urandom'],stdout=sub.PIPE,stderr=sub.PIPE)
                        output, errors = p.communicate()
                        self._seed = int(output)
                        print('SEED: ' + str(self._seed))
                        first_feature_no = self._seed%feature_length
                        nonzeros = np.count_nonzero(self.data[:,first_feature_no])
                        if nonzeros > int(dataset_length*0.005) :
                            self._first = first_feature_no
                            break
                        else:
                            if nonzeros > last_nonzeros:
                                best = first_feature_no
                                last_nonzeros = nonzeros
                    if self._first is None:
                        self._first = best
                    print('FIRST SELECTION NO: ' + str(self._first))
                    print('NON-ZEROS IN SELECTION: ' + str(nonzeros/feature_length) + '%') 
                else:
                    print('SEED: ' + str(self._seed))
                    self._first = self._seed%feature_length
                    nonzeros = np.count_nonzero(self.data[:,self._first])
                    print('FIRST SELECTION NO: ' + str(self._first))
                    print('NON-ZEROS IN SELECTION: ' + str(nonzeros/feature_length) + '%')
                sys.stdout.flush()

        all_errors = []
        all_cids = []
        if freq == 0 or freq < 0:
            freq = cols
            if(cols == self._cols and rows == 0 or
               rows == self._rows and cols == 0):
                if cols == self._cols:
                    self._cid = self.apply_fps(cols=cols)
                elif rows == self._rows:
                    self._cid = self.apply_fps(rows=rows)
                print('[FPS:] Non-recurrent Selection size : ' + str(len(list(set([c for c in self._cid])))))
                print('[FPS:] Calculating Error between FPS Selection and Data ...')
                sys.stdout.flush()
                self.computeUCR()
                self._colerr.append(self.error)
                print('[FPS:] Frobenius Norm Error between X and X` = FPS : ' + str(self.error))     
                print('Saving restart file.') 
                sys.stdout.flush()
                write_restart(self._restart_file, self.data, self._selected, self._colerr, self.cycles)
                all_cids=self._cid
                all_errors.append(self.error)
        else:
            if cols > 0 and not restart:
                start = time.time()
                print('[FPS:] Applying FPS ... ')
                sys.stdout.flush()
                self._selected = self.apply_fps(cols=self._cols)
                print('[FPS:] Selected Columns : ' + str(self._selected))
                end = time.time()
                print('Elapsed time for selection : ' + str(end - start))
                sys.stdout.flush()
            elif rows > 0 and not restart:
                self._selected = self.apply_fps(rows=self._rows)

            skipthis = 0
            range_list = [1]
            range_list.extend([x for x in range(freq-1,cols,freq)])
            range_list.extend([cols])
            for numc in range_list:
                if skips > skipthis:
                    skipthis += 1
                    self._cid = self._selected[:numc+1]
                    continue
                self._cid = self._selected[:numc+1]
                print('[FPS:] Selected Column Number :',numc)
                self.cycles.append(int(numc))
                print('[FPS:] Non-recurrent Selection size : ' + str(len(list(set([c for c in self._cid])))))
                print('[FPS:] Calculating Error between FPS Selection and Data ...')
                sys.stdout.flush()
                self.computeUCR()
                self._colerr.append(self.error)
                print('[FPS:] Frobenius Error between X and X` = FPS : ' + str(self.error))     
                print('Saving restart file.') 
                sys.stdout.flush()
                write_restart(self._restart_file, self.data, self._selected, self._colerr, self.cycles)
                all_cids.append(self._cid)
                all_errors.append(self.error)

        return all_cids, all_errors

# Simple effective FPS version with less information.
def calc_FPS(data, N):
    k = []
    rows = data.shape[0]
    cols = data.shape[1]
    farthests = np.zeros((N, rows))
    first = np.random.randint(cols)
    farthests[0] = data.T[first]
    k.append(first)
    distances = np.linalg.norm(farthests[0] - data.T, axis=1)
    for i in range(1, N):
        pos = np.argmax(distances)
        farthests[i] = data.T[pos]
        k.append(pos)
        distances = np.minimum(distances, np.linalg.norm(farthests[i] - data.T, axis=1))
    return k

def read_h5(filename):
    with h5py.File(filename, 'r') as hf:
        #data = hf.get('dataset')
        data = hf['dataset'][:]
    return np.array(data)

def read_restart(filename):
    with h5py.File(filename, 'r') as hf:
        data = hf['restart/dataset'][:]
        cols = hf['restart/features'][:]
        errs = hf['restart/errors'][:]
        cycles = hf['restart/steps'][:]
    return np.array(data), np.array(cols), np.array(errs), np.array(cycles)

def write_restart(filename, data, cols, errs, cycles):
    with h5py.File(filename, 'w') as hf:
        hf.create_dataset('restart/dataset', data=data)
        hf.create_dataset('restart/features', data=cols)
        hf.create_dataset('restart/errors', data=errs)
        hf.create_dataset('restart/steps', data=cycles)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Apply FPS selection CUR reduction on descriptors.")
    parser.add_argument("filename", 
            help="File path and name for the database to work.")
    parser.add_argument("-c", "--colnum", type=int, required=False, default=-1, 
            help="Number of features to select for CUR decomposition.")
    parser.add_argument("-rn", "--rownum", type=int, required=False, default=-1, 
            help="Number of entries to select for CUR decomposition.")
    parser.add_argument("-p", "--error_print", type=int, required=False, default=0, 
            help="Number of intervals to calculate error.")
    parser.add_argument("-noh5", "--noh5", action="store_true",
            help="H5 input data file.")
    parser.add_argument("-nonan", "--nonan", action="store_true",
            help="H5 input data file.")
    parser.add_argument("-x0", "--removezeros", action="store_true",
            help="Remove zero columns from input data file.")
    parser.add_argument("-r", "--restart", action="store_true",
            help="Restart from H5 restart save file.")
    args = parser.parse_args()
    
    print('Reading DATABASE FILE : ' + str(args.filename))
    sys.stdout.flush()
    dataX = None
    cols = None
    errs = None
    cycles = None
    base = os.path.basename(args.filename)
    filebase = os.path.splitext(base)[0]
    filerestart = filebase+'_restart.h5'
    if args.noh5:
        data = np.loadtxt(args.filename)
    else:
        if args.restart:
            data, cols, errs, cycles = read_restart(filerestart)
        else:
            data = read_h5(args.filename)

    print('----- Checking NaN and Inf values in Dataset -----')
    hasnaninf = False
    if np.isnan(data).any():
        hasnaninf = True
        if args.nonan:
            print(' Warning : Data has NaN values! Ignoring NaNs and set them to zero!')
        else:
            print(' Error : Data has NaN values! Can not continue. Exiting.')
            exit(0)
    if np.isinf(data).any():
        hasnaninf = True
        if args.nonan:
            print(' Warning : Data has Inf values! Setting Infs to very large values!')
        else:
            print(' Error : Data has Inf values! Can not continue. Exiting.')
            exit(0)
    if hasnaninf is False:
        print(' There are no NaN or Inf values in dataset. Good to go.')
    else:
        if args.nonan:
            print(' Ignoring NaN or Inf values in dataset and '
                  'setting them to zero and very large number, respectively.')
            data = np.nan_to_num(data)
    
    col_norm = np.linalg.norm(data, axis=0)
    nonzero_cols = len(col_norm[col_norm>0.])
    print(' Number of non-zero columns: '+str(nonzero_cols))
    print(' Percent of non-zero columns: '+str(100*nonzero_cols/data.shape[1]))
    row_norm = np.linalg.norm(data, axis=1)
    nonzero_rows = len(row_norm[row_norm>0.])
    print(' Number of non-zero rows: '+str(nonzero_rows))
    if args.removezeros:
        if nonzero_cols < data.shape[1]:
            data = data[:,col_norm>0.]
    
    print('  N=' + str(args.colnum))

    print('  Database size : ' + ' x '.join([ str(s) for s in data.shape]))
    the_FPS = FPS(data, restart_file=filerestart)
    selections, fps_errors = the_FPS.select(cols=args.colnum, 
                                            rows=args.rownum, 
                                            freq=args.error_print,
                                            restart_cols=cols,
                                            restart_errs=errs,
                                            restart_cycles=cycles)
    print('The selected columns : ' + ','.join([ str(ci) for ci in the_FPS._cid]))
    for eri, er in enumerate(the_FPS._colerr):
        print(' Errors: ' + str(er))
    sys.stdout.flush()

