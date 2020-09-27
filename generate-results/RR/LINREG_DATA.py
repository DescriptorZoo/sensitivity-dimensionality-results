#
# License is GPL-3 (2019)
# Author: Berk Onat
# This package id part of https://github.com/DescriptorZoo
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

from sklearn import linear_model, kernel_ridge
from sklearn.metrics import mean_squared_error, r2_score

from numpy.linalg import svd as svd
from scipy.linalg import pinv2
from scipy.linalg import pinv
from scipy.linalg import svd as s_svd
from sklearn.utils.extmath import randomized_svd

PI = 3.14159265359

def LINREG(MLdata, selected_features=None, use_krr=False, use_lstsq=False, use_qr=False,
           lr=1e-7, error_print=1, nonan=False, use_lsq=False, kernel='linear', use_l2=False,
           stop_tol=1e-6, stop_num=5000, go_tol=100, alpha=1, degree=3, use_ridge=False,
           use_br=False, ridgetol=1e-3, maxiter=None, solver='auto', gamma=None):
    """
       Linear, Ridge and KRR Regression
    """
    if MLdata.nfeatures is None:
        nfeatures = MLdata.data.shape[1]
    else:
        nfeatures = MLdata.nfeatures
    if selected_features is None:
        selected_features = np.arange(nfeatures)
    sel = np.array(selected_features)
    print("Linear regression with ",nfeatures," parameters.")
    trainN = MLdata.rN
    testN = MLdata.tN
    if MLdata.glob_desc:
        print("Regression on matrix with global descriptor.")
    else:
        print("Regression on per atom matrix with atomic descriptor.")
    if MLdata.kernel:
        print("Regression on kernel matrix with linear kernel.")
    sys.stdout.flush()
    as_glob_desc = False
    if MLdata.glob_desc or use_qr:
        as_glob_desc = True
    if as_glob_desc and (use_lsq or use_br or use_ridge or use_krr or use_l2 or use_lstsq or use_qr):
        if use_krr:
            regr = kernel_ridge.KernelRidge(alpha=alpha, kernel=kernel, degree=degree, gamma=gamma)
        elif use_br:
            regr = linear_model.BayesianRidge(alpha_1=alpha, alpha_2=alpha,
                                              lambda_1=alpha, lambda_2=alpha, 
                                              tol=ridgetol, n_iter=maxiter)
        elif use_ridge:
            regr = linear_model.Ridge(alpha=alpha, solver=solver, tol=ridgetol, max_iter=maxiter)
        else:
            # Create linear regression object
            regr = linear_model.LinearRegression()

        A = MLdata.train()[:,sel[:nfeatures]]
        B = MLdata.test()[:,sel[:nfeatures]]
        y = MLdata.target()
        z = MLdata.test_target()
        if use_l2:
            G = alpha * np.eye(nfeatures)
            AA = np.vstack([A,G])
            yy = np.hstack([y,np.zeros(nfeatures)])
            # Train the model using the training sets
            if use_lstsq:
                c, resid, rnk, s = np.linalg.lstsq(AA, yy)
            elif use_qr:
                Q, R = np.linalg.qr(AA)
                b = np.dot(Q.T, yy)
                c = np.linalg.solve(R, b)
            else:
                regr.fit(AA, yy)
        else:
            if use_lstsq:
                c, resid, rnk, s = np.linalg.lstsq(A, y)
            elif use_qr:
                Q, R = np.linalg.qr(A)
                b = np.dot(Q.T, y)
                c = np.linalg.solve(R, b)
            else:
                # Train the model using the training sets
                regr.fit(A, y)

        if use_lstsq or use_qr:
            y_pred = A.dot(c)
            z_pred = B.dot(c)
        else:
            y_pred = regr.predict(A)
            z_pred = regr.predict(B)
        y_test = y
        # Make predictions using the testing set
        z_test = z


        if use_krr or use_ridge:
            parms = regr.get_params()
            print('Coefficients: \n', parms)
            params = [v for k,v in regr.get_params().items() if isinstance(v,(np.int,np.float))]
        elif use_lstsq or use_qr:
            params = c
            print('Coefficients: \n', params)
        else:
            params = regr.coef_
            print('Coefficients: \n', params)
        # The coefficients
        # The mean squared error
        print('Train Mean squared error: %.2f'
              % mean_squared_error(y_test, y_pred))
        print('Test Mean squared error: %.2f'
              % mean_squared_error(z_test, z_pred))
        # The coefficient of determination: 1 is perfect prediction
        print('Coefficient of determination: %.2f'
              % r2_score(z_test, z_pred))
        tr_na = MLdata.train_natoms()
        tt_na = MLdata.test_natoms()
        if MLdata.stds is not None:
            tr_mse = np.square((y_pred-y_test)*MLdata.stds/tr_na).sum()/trainN
            tt_mse = np.square((z_pred-z_test)*MLdata.stds/tt_na).sum()/testN
        elif MLdata.scales is not None:
            tr_mse = np.square((y_pred-y_test)*MLdata.scales/tr_na).sum()/trainN
            tt_mse = np.square((z_pred-z_test)*MLdata.scales/tt_na).sum()/testN
        else:
            tr_mse = np.square((y_pred-y_test)/tr_na).sum()/trainN
            tt_mse = np.square((z_pred-z_test)/tt_na).sum()/testN
        print('eV/atom Train MSE: ',tr_mse,
              ' Test MSE: ',tt_mse)
        print('eV/atom Train RMSE: ',np.sqrt(tr_mse),
              ' Test RMSE: ',np.sqrt(tt_mse))
        sys.stdout.flush()
        return params, 0.0
    else:
        startover=True
        while startover:
            learnR = lr
            previous_loss = 1e+5
            learn_faster=True
            learn_faster2=True
            w1 = np.random.randn(nfeatures, 1)
            w1 = np.ascontiguousarray(w1, dtype=np.float64)
            b1 = np.random.randn(1)
            for t in range(stop_num+1):
                loss = 0.
                if MLdata.glob_desc:
                    x = MLdata.train()[:,sel[:nfeatures]]
                    preds = x.dot(w1) + b1
                    y_err = preds - MLdata.target().reshape(-1,1)
                    loss = np.square(y_err).sum()
                    y_err *= 2.0
                    grads_w1_e = x.T.dot(y_err)
                    grads_b1_e = y_err.sum()
                else:
                    grads_w1_e = np.ones((trainN,nfeatures))
                    grads_b1_e = np.ones((trainN,1))
                    for i in range(trainN):
                        # Forward pass: compute predicted y
                        x = MLdata.train(i)[:,sel[:nfeatures]]
                        preds = x.dot(w1) + b1
                        #preds = x.dot(w1)
                        y_err = (preds.sum() - MLdata.target(i))/MLdata.train_natoms(i)
                        loss += np.square(y_err)
                        y_err *= 2.0
                        # Compute each contribution for gradients of w1 with respect to loss
                        grads_w1_e[i,:] = x.T.dot(np.full(x.shape[0], y_err))
                        grads_b1_e[i,:] = y_err
                # Compute and print loss
                if t == 0 or t % error_print == 0 or t == stop_num:
                    test_err = 0.
                    if MLdata.glob_desc:
                        testx = MLdata.test()[:,sel[:nfeatures]]
                        test_preds = testx.dot(w1) + b1
                        test_err = (test_preds - MLdata.test_target().reshape(-1,1))/MLdata.test_natoms()
                        test_err = np.square(test_err).sum()
                    else:
                        for i in range(testN):
                            testx = MLdata.test(i)[:,sel[:nfeatures]]
                            test_preds = testx.dot(w1) + b1
                            #test_preds = testx.dot(w1)
                            test_err += np.square(test_preds.sum() - MLdata.test_target(i)/MLdata.test_natoms(i))
                    current = loss/trainN
                    print("Iter: ",t," Train RMSE: ",np.sqrt(current)," Test RMSE: ",np.sqrt(test_err/testN))
                    sys.stdout.flush()
                    if t > go_tol*10:
                        if current < 0.2:
                            startover = False
                        else:
                            startover = True
                            print("Starting over.")
                            sys.stdout.flush()
                            break
                    elif t > 0:
                        if current < go_tol:
                            startover = False
                        else:
                            startover = True
                            print("Starting over.")
                            sys.stdout.flush()
                            break
                    if current > previous_loss * 1.20:
                        learnR *= 0.1
                    previous_loss = current

                if MLdata.glob_desc:
                    grad_w1 = grads_w1_e
                    grad_b1 = grads_b1_e
                else:
                    # Backprop to compute gradients of w1 with respect to loss
                    grad_w1 = grads_w1_e.sum(axis=0)
                    grad_w1 = grad_w1.reshape((nfeatures,1))
                    grad_b1 = grads_b1_e.sum(axis=0)
    
                # Update weights
                w1 -= learnR * grad_w1
                b1 -= learnR * grad_b1

                if loss < stop_tol:
                    break

        return w1, b1

class MLDATA(object):
    def __init__(self, data, targets, split_list=None, percent=0.8, use_l2=False, nfeatures=None,
                 alpha=1e-2, toglobal=False, norm_data=0, kernel=False, use_mean=False, use_qr=False):
        self.data = data
        if nfeatures is None:
            self.nfeatures = data.shape[1] 
        else:
            self.nfeatures = nfeatures
        self.kernel = kernel
        self.means = None
        self.use_mean = use_mean
        self.use_qr = use_qr
        self.stds = None
        self.mins = None 
        self.scales = None
        self.natoms = targets[:,0]
        if data.shape[0] == targets.shape[0]:
            self.glob_desc=True
            if norm_data == 1:
                self.mins, self.scales, self.targets = normalize(targets[:,1])
            if norm_data == 2:
                self.means, self.stds, self.targets = standardize(targets[:,1])
            else:
                self.targets = targets[:,1]
            self.N = len(self.targets)
        else:
            self.glob_desc=False
            self.slice_list = targets[:,0]
            if norm_data == 1:
                self.mins, self.scales, self.targets = normalize(targets[:,1])
            if norm_data == 2:
                self.means, self.stds, self.targets = standardize(targets[:,1])
            else:
                self.targets = targets[:,1]
            self.N = len(self.targets)
            self.slices = self.slice_array()
            if toglobal:
                self.glob_desc=True
                self.data = self.make_global_representation()
                print("Reduced shape: ",self.data.shape)
        self.tr, self.tt = self.split_test_train(
                                split_list=split_list,
                                perc=percent)
        self.rN = len(self.tr)
        self.tN = len(self.tt)
        print("Dataset is splitted to ",self.rN,
              " and ",self.tN," for training and test.")

    def make_global_representation(self):
        sl = self.slices[:,0]
        print("Data shape before reduce: ",self.data.shape)
        return np.add.reduceat(self.data, sl)

    def slice_array(self):
        slices = np.zeros((self.N,2), dtype=np.int)
        i = 0
        for si, s in enumerate(self.slice_list[:self.N]):
            slices[si][0]=i
            i+=s
            slices[si][1]=i
        slices[-1][1] -= 1
        return slices
  
    def split_test_train(self, split_list=None, perc=0.8):
        if split_list is not None:
            Ntr = int(split_list[0])
            print("Using splitted dataset to ",int(Ntr)," and ",int(self.N-Ntr),
                  " for training and testing.")
            return (np.array(split_list[1:int(Ntr+1)], dtype=np.int),
                    np.array(split_list[int(Ntr+1):], dtype=np.int))
        else:
            rand_list = np.arange(self.N)
            np.random.shuffle(rand_list)
            Ntr = np.rint(self.N * perc)
            print("Splitting dataset to ",int(Ntr)," and ",int(self.N-Ntr),
                  " for training and testing.")
            return rand_list[:int(Ntr)].copy(), rand_list[int(Ntr):].copy()

    def linear_kernel(self, test=True):
        sl = self.tt if test else self.tr
        if self.glob_desc:
            return self.data[sl,:]
        else:
            X = np.zeros(len(sl),len())
            for i,j in zip(range(len(sl)),range(len.sl)):
                X[i,j] = self.data[sl[i],:].T.dot(self.data[sl[j],:])
            return X

    def train_natoms(self, i=None, j=None):
        if i is None:
            return self.natoms[self.tr]
        else:
            if j is None:
                return self.natoms[self.tr[i]]
            else:
                return self.natoms[self.tr[i:j]]

    def test_natoms(self, i=None, j=None):
        if i is None:
            return self.natoms[self.tt]
        else:
            if j is None:
                return self.natoms[self.tt[i]]
            else:
                return self.natoms[self.tt[i:j]]

    def train(self, i=None, j=None):
        if i is None:
            if self.glob_desc:
                return self.data[self.tr,:]
            else:
                sel = []
                for i in self.tr:
                    sel.extend(
                        np.arange(
                            int(self.slices[i][0]),
                            int(self.slices[i][1])
                            )
                        )
                return self.data[sel,:]
        else:
            if self.glob_desc:
                if j is None:
                    return self.data[self.tr[i],:]
                else:
                    return self.data[self.tr[i:j],:]
            else:
                si,sj = self.slices[self.tr[i]]
                if sj >= self.data.shape[0]:
                    return self.data[si:,:]
                else:
                    return self.data[si:sj,:]

    def test(self, i=None, j=None):
        if i is None:
            if self.glob_desc:
                return self.data[self.tt,:]
            else:
                sel = []
                for i in self.tr:
                    sel.extend(
                        np.arange(
                            int(self.slices[i][0]),
                            int(self.slices[i][1])
                            )
                        )
                return self.data[sel,:]
        else:
            if self.glob_desc:
                if j is None:
                    return self.data[self.tt[i],:]
                else:
                    return self.data[self.tt[i:j],:]
            else:
                si,sj = self.slices[self.tt[i]]
                if sj >= self.data.shape[0]:
                    return self.data[si:,:]
                else:
                    return self.data[si:sj,:]
    
    def target(self, i=None, j=None):
        if i is None:
            return self.targets[self.tr]
        else:
            if j is None:
                return self.targets[self.tr[i]]
            else:
                return self.targets[self.tr[i:j]]
    
    def test_target(self, i=None, j=None):
        if i is None:
            return self.targets[self.tt]
        else:
            if j is None:
                return self.targets[self.tt[i]]
            else:
                return self.targets[self.tt[i:j]]

def mul_cols(data, vec, axis=1):
    if axis==1:
        return np.multiply(data, vec)
    else:
        return np.multiply(data.T, vec).T

def normalize(data):
    maxs = np.amax(data)
    mins = np.amin(data)
    scales = maxs-mins
    return mins, scales, (data-mins)/scales

def standardize(data):
    means = np.mean(data)
    stds = np.std(data)
    print("Mean: ",means," Std: ",stds)
    return means, stds, (data-means)/stds

def randomize_data(data):
    data = np.random.randn(data.shape[0],data.shape[1])
    return np.ascontiguousarray(data, dtype=np.float64)

def allones_data(data):
    data = np.ones(data.shape)
    return np.ascontiguousarray(data, dtype=np.float64)

def normalize_data(data, axis=1):
    if axis==1:
        scale_axis=0
    else:
        scale_axis=1
    maxs = np.amax(data, axis=scale_axis)
    mins = np.amin(data, axis=scale_axis)
    scales = maxs-mins
    scales[scales==0.] = 1.
    return mul_cols(data-mins, 1./scales, axis=axis)

def standardize_data(data, axis=1):
    if axis==1:
        scale_axis=0
    else:
        scale_axis=1
    means = np.mean(data, axis=scale_axis)
    stds = np.std(data, axis=scale_axis)
    scales = np.ones(data.shape[1])
    scales[stds != 0.] = 1./stds[stds != 0.]
    return mul_cols(data-means, 1./scales, axis=axis)

def scale_data(data, scale=1., axis=1):
    return mul_cols(data, np.array([scale for a in range(data.shape[axis])]), axis=axis)

def read_h5(filename):
    with h5py.File(filename, 'r') as hf:
        data = hf['dataset'][:]
    return np.array(data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generates SF descriptors and optionally apply CUR.")
    parser.add_argument("filename", 
            help="File path and name for the database to work.")
    parser.add_argument("targetfile", 
            help="File path and name for the target values.")
    parser.add_argument("-c", "--colnums", type=str, required=False, default=None, 
            help="CSV file name for the selected feature ids.")
    parser.add_argument("-s", "--splitlist", type=str, required=False, default=None, 
            help="file name for the train-test split list values.")
    parser.add_argument("-p", "--error_print", type=int, required=False, default=1, 
            help="'k' number for CUR decomposition.")
    parser.add_argument("-f", "--features", type=int, required=False, default=None, 
            help="'k' number for CUR decomposition.")
    parser.add_argument("-n", "--stopnum", type=int, required=False, default=1000, 
            help="'k' number for CUR decomposition.")
    parser.add_argument("-tol", "--tol", type=float, required=False, default=1e-4, 
            help="'k' number for CUR decomposition.")
    parser.add_argument("-gotol", "--gotol", type=float, required=False, default=30, 
            help="'k' number for CUR decomposition.")
    parser.add_argument("-lr", "--lr", type=float, required=False, default=1e-7, 
            help="'k' number for CUR decomposition.")
    parser.add_argument("-nrm", "--nrm", action="store_true",
            help="Normalise data.")
    parser.add_argument("-std", "--std", action="store_true",
            help="Standardise data.")
    parser.add_argument("-tnrm", "--tnrm", action="store_true",
            help="Normalise target values.")
    parser.add_argument("-tstd", "--tstd", action="store_true",
            help="Standardise target values.")
    parser.add_argument("-lsq", "--lsq", action="store_true",
            help="Use LSQ method with sklearn linear regression.")
    parser.add_argument("-lstsq", "--lstsq", action="store_true",
            help="Use LSQ method with numpy.linalg.lstsq.")
    parser.add_argument("-l2", "--l2", action="store_true",
            help="Use LSQ with L2 regularization.")
    parser.add_argument("-krr", "--krr", action="store_true",
            help="Use KRR method with sklearn linear kernel.")
    parser.add_argument("-br", "--br", action="store_true",
            help="Use Bayesian Regression method with sklearn.")
    parser.add_argument("-qr", "--qr", action="store_true",
            help="Use QR to solve Ax=b.")
    parser.add_argument("-alpha", "--alpha", type=float, required=False, default=0.001, 
            help="KRR parameter.")
    parser.add_argument("-gamma", "--gamma", type=float, required=False, default=None, 
            help="KRR parameter.")
    parser.add_argument("-degree", "--degree", type=int, required=False, default=50, 
            help="KRR parameter.")
    parser.add_argument("-kernel", "--kernel", type=str, required=False, default="linear", 
            help="KRR parameter.")
    parser.add_argument("-ridge", "--ridge", action="store_true",
            help="Use ridge method.")
    parser.add_argument("-maxiter", "--maxiter", type=int, required=False, default=None, 
            help="Ridge parameter.")
    parser.add_argument("-ridgetol", "--ridgetol", type=float, required=False, default=1e-3, 
            help="Ridge parameter.")
    parser.add_argument("-solver", "--solver", type=str, required=False, default='auto', 
            help="Ridge parameter.")
    parser.add_argument("-g", "--toglobal", action="store_true",
            help="Sum atomic descriptors to define global representation (linear).")
    parser.add_argument("-usemean", "--usemean", action="store_true",
            help="Sum atomic descriptors to define global representation (linear).")
    parser.add_argument("-noh5", "--noh5", action="store_true",
            help="H5 input data file.")
    parser.add_argument("-nonan", "--nonan", action="store_true",
            help="H5 input data file.")
    parser.add_argument("-x0", "--removezeros", action="store_true",
            help="Remove zero columns from input data file.")
    parser.add_argument("-rnd", "--rnd", action="store_true",
            help="Use random values instead of dataset.")
    parser.add_argument("-ones", "--ones", action="store_true",
            help="Use ones instead of dataset.")
    parser.add_argument("-targets1", "--targets1", action="store_true",
            help="Use only references with 1 in 3rd column in dataset.")
    parser.add_argument("-ref1", "--ref1", action="store_true",
            help="Use cohesive energies as targets (ref is isolated atom). (Default: ref is bulk)")
    args = parser.parse_args()

    print('Reading DATABASE FILE : ' + str(args.filename))
    if args.noh5:
        data = np.loadtxt(args.filename)
    else:
        data = read_h5(args.filename)
    print('  Database size : ' + ' x '.join([ str(s) for s in data.shape]))
    print('Reading targets from file : ' + str(args.targetfile))
    targets = np.loadtxt(args.targetfile)
    print(targets.shape)
    bulk_ref_1=-163.177966801875 
    if args.ref1:
        bulk_ref_1=-158.54496821
    bulk_ref_2=-381.2147714 / 64
    s2 = targets[:,2]>1.
    ntargets = np.zeros(len(targets))
    ntargets[~s2] = targets[~s2][:,1] - bulk_ref_1 * targets[~s2][:,0]
    ntargets[s2] = targets[s2][:,1] - bulk_ref_2 * targets[s2][:,0]
    targets[:,1] = ntargets

    if args.targets1:
        r1lines = targets[~s2][:,0].sum()
        ndata = data[0:int(r1lines),:]
        data = ndata
        targets = np.vstack([targets[~s2][:,0],targets[~s2][:,1],targets[~s2][:,2]]).T
        print("New data shape: ",data.shape)
        print("New target shape: ",targets.shape)
    print("Targets min: ",np.min(targets[:,1])," max: ",np.max(targets[:,1]))
    print("Targets min id: ",np.argmin(targets[:,1])," max id: ",np.argmax(targets[:,1]))
   
    colnums = None
    if args.colnums is not None:
        print('Reading file with selected features: ' + str(args.colnums))
        colnums = np.loadtxt(args.colnums, delimiter=',', dtype=np.int)
        print("Col nums shape:",colnums.shape)

    splitlist = None
    if args.splitlist is not None:
        print('Reading file with split list: ' + str(args.splitlist))
        splitlist = np.loadtxt(args.splitlist, dtype=np.int)
        print("Split list shape:",splitlist.shape)

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
            print(' New database size : ' + ' x '.join([ str(s) for s in data.shape]))

    print('----- Linear Regression for Dataset -----')
    if colnums is None:
        print('  N=' + str(data.shape[1]))
    else:
        print('  N=' + str(colnums.shape))
    normdata = 0
    if args.tnrm:
        normdata = 1
    elif args.tstd:
        normdata = 2
    if args.nrm:
        data = normalize_data(data, axis=1)
    if args.std:
        data = standardize_data(data, axis=1)
    if args.rnd:
        data = randomize_data(data)
    if args.ones:
        data = allones_data(data)
    mldat = MLDATA(data, targets, norm_data=normdata, split_list=splitlist, alpha=args.alpha,
                   toglobal=args.toglobal, nfeatures=args.features,use_qr=args.qr,
                   use_mean=args.usemean, use_l2=args.l2)
    if splitlist is None:
        np.savetxt("split_list.txt",np.hstack([np.array(len(mldat.tr)),mldat.tr,mldat.tt]),fmt="%i")
    
    weights, biases = LINREG(mldat, lr=float(args.lr), selected_features=colnums, use_lstsq=args.lstsq,
                             error_print=int(args.error_print), nonan=args.nonan, use_lsq=args.lsq, use_krr=args.krr, use_l2=args.l2,
                             stop_tol=float(args.tol), stop_num=int(args.stopnum), go_tol=args.gotol, gamma=args.gamma,
                             kernel=args.kernel, alpha=args.alpha, degree=args.degree, use_ridge=args.ridge, use_qr=args.qr,
                             use_br=args.br, ridgetol=args.ridgetol, maxiter=args.maxiter, solver=args.solver)
    np.savetxt("weights.txt",np.hstack([weights,biases]))

    sys.stdout.flush()

