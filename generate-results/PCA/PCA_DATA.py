#
# License is GPL-3 (2019)
# Author: Berk Onat
# This package id part of https://github.com/berkonat/descriptorzoo
# 
#
import os
import sys
import math
import argparse
import numpy as np
import copy
import h5py
import types
import time

import scipy.sparse
import scipy.linalg
import scipy.integrate

from numpy.linalg import pinv, svd
from scipy.linalg import pinv2
from sklearn.utils.extmath import randomized_svd
from sklearn.decomposition import PCA

class PCA_DATA(object):
    def __init__(self, data, dim=None, verbose=False, save=False, save_name=None):
        self.data = data
        self.data_shape = data.shape
        self.model = None
        if dim is not None:
            self.newdim = dim
        else:
            self.newdim = data.shape[1]
        self._verbose = verbose
        self._save = save
        if save_name is None:
            self.save_name = 'DATA_PCA_save'
        else:
            self.save_name = save_name
        self.nodata_files = True

    def verbose(self, verbose=None):
        rtn = False
        if verbose is not None:
            if verbose:
                rtn =  True
        elif self._verbose:
            rtn = True
        return rtn
    
    def save(self, save=None):
        rtn = False
        if save is not None:
            if save:
                rtn =  True
        elif self._save:
            rtn = True
        return rtn

    def write(self, fname, data, msg=None, verbose=None):
        if self.verbose(verbose=verbose):
            if msg is None:
                msg = 'Writting data into '
            print(msg+str(fname)+' file.')
            sys.stdout.flush()
        np.savetxt(fname, data) 
        if self.verbose(verbose=verbose):
            print('File is written.')
  
    def fit(self, verbose=None, save=True, save_name=None):
        if self.verbose(verbose=verbose):
            print('Calculating averages and shifting data...')
        self.dataAvg = np.mean(self.data, axis=0)
        self.data = self.data - self.dataAvg[:,None].T
        if self.save(save=save):
            if save_name is not None:
                outputname = save_name
            else:
                outputname = self.save_name
            self.write(outputname+'_AVERAGES', 
                       self.dataAvg[:,None], 
                       msg='Writing averages of columns for data into ',
                       verbose=verbose)
        if self.verbose(verbose=verbose):
            print('Fitting PCA with data...')
        self.model=PCA(n_components=self.newdim,copy=True)
        self.model.fit(self.data)
        if self.verbose(verbose=verbose):
            print('The explained variance ratios :')
            for ratio in self.model.explained_variance_ratio_:
                print(ratio)
            sys.stdout.flush()
        if self.save(save=save):
            self.write(outputname+'_VARRATIOS', 
                       self.model.explained_variance_ratio_, 
                       msg='Writing variance ratios to ',
                       verbose=verbose)
            self.write(outputname+'_VARIANCES',
                       self.model.explained_variance_,
                       msg='Writing variances to ',
                       verbose=verbose)
        if self.verbose(verbose=verbose):
            print('The total variance explained is {0}'.format(np.sum(self.model.explained_variance_ratio_)))
            sys.stdout.flush()

    def transform(self, save=None, verbose=None, save_name=None):
        if self.verbose(verbose=verbose):
            print('Applying PCA on Dataset.')
        self.data = self.model.transform(self.data) # apply PCA transform to data
        if self.verbose(verbose=verbose):
            print('Transformation is completed.')
        if save_name is not None:
            outputname = save_name
        else:
            outputname = self.save_name
        if self.nodata_files is False:
            self.write(outputname+'.gz', self.data, msg='Writing '+' x '.join([
                str(s) for s in np.shape(self.data)
                ])+' PCA transformed DATA to ', verbose=verbose)
        if self.save(save=save):
            self.write(outputname+'_COVMAT.gz', self.model.components_, msg='Writing '+' x '.join([
                    str(s) for s in np.shape(self.model.components_)
                    ])+' covariant matrix to ', verbose=verbose)

    def standardize(self, verbose=None, save=None, save_name=None):
        self.std_devs = np.std(self.data, axis=0)
        self.data = self.data / self.std_devs[:,None].T
        self.std_scale = np.mean(np.linalg.norm(self.data,axis=1))
        if self.verbose(verbose=verbose):
            print('After standardizing, the average L2 norm of data is', self.std_scale)
        if self.save(save=save):
            if save_name is not None:
                outputname = save_name
            else:
                outputname = self.save_name
            if self.nodata_files is False:
                self.write(outputname+'_STD.gz', self.data, msg='Writing '+' x '.join([
                    str(s) for s in np.shape(self.data)
                    ])+' Standardized PCA DATA to ', verbose=verbose)
            self.write(outputname+'_STD_DEVS', self.std_devs[:,None], 
                       msg='Writing standard deviations of columns for data into ',
                       verbose=verbose)

    def normalize(self, verbose=None, save=None, save_name=None):
        self.norm_vals = np.maximum(self.data) - np.minimum(self.data)
        self.data = self.data / self.norm_vals[:,None].T 
        self.norm_scale=np.mean(np.linalg.norm(self.data,axis=1))
        if self.verbose(verbose=verbose):
            print('After normalization, the average L2 norm of data is', self.norm_scale)
        if self.save(save=save):
            if save_name is not None:
                outputname = save_name
            else:
                outputname = self.save_name
            if self.nodata_files is False:
                self.write(outputname+'_NORM.gz', self.data, msg='Writing '+' x '.join([
                       str(s) for s in np.shape(self.data)
                       ])+' Normalized PCA DATA to ', verbose=verbose)
            self.write(outputname+'_NORM_VALS', self.norm_vals[:,None], 
                       msg='Writing scaling (max-min) of columns for data into ',
                       verbose=verbose)

    def importance(self, score=None, verbose=None, save=None, save_name=None, num_if=None):
        if score is None:
            #score = 0.99999 # 99.999%
            score = 1.0E-12 # 10^-10
        if num_if is None:
            num_if = 1
        # Following the codes and discussions here:
        # https://stackoverflow.com/questions/50796024/feature-variable-importance-after-a-pca-analysis/50845697
        self.initial_feature_ids = np.array(range(self.data_shape[1]))
        if score >= 1.:
            self.num_pcs= self.model.components_.shape[0]
        else:
            self.num_pcs = 0
            #self.target_ratio = 0.
            for pcs_ratio in self.model.explained_variance_ratio_:
                #self.target_ratio += pcs_ratio
                self.num_pcs += 1
                #if self.target_ratio >= score:
                if pcs_ratio <= score:
                    break
        self.most_important_pcs = np.array([np.array(np.abs(self.model.components_[i]).argsort()[::-1][:num_if]) for i in range(self.num_pcs)])
        self.most_important_features = [self.initial_feature_ids[self.most_important_pcs[i]] for i in range(self.num_pcs)]
        self.important_features = {'PC{}'.format(i+1): ','.join([str(s) for s in self.most_important_features[i]]) for i in range(self.num_pcs)}
        if self.verbose(verbose=verbose): 
            print('The important features after PCA analysis :')
            for k,v in self.important_features.items():
                print(k,v)
                sys.stdout.flush()
        if self.save(save=save):
            if save_name is not None:
                outputname = save_name
            else:
                outputname = self.save_name
            if self.verbose(verbose=verbose): 
                print('Writing important feature set for data into '+str(outputname)+'_IMPORTANT_FEATURES file.')
            with open(outputname+'_IMPORTANT_FEATURES', 'w') as fp:
                json.dump(self.important_features, fp, indent=4)

def read_h5(filename):
    with h5py.File(filename, 'r') as hf:
        data = hf['dataset'][:]
    return np.array(data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Apply PCA on DATA.")
    parser.add_argument("filename", 
            help="File path and name for the database to work.")
    parser.add_argument("-o", "--output", type=str, required=False, default='DATA_PCA_save',
            help="output file name for PCA transformed dataset.")
    parser.add_argument("-w", "--write_files", action="store_true",
            help="Write outputs to files. Default: False for all, only writes final PCA DATA.")
    parser.add_argument("-nd", "--nodata_files", action="store_true",
            help="Write outputs to files. Default: False for all, only writes final PCA DATA.")
    parser.add_argument("-d", "--dim", type=int, required=False, default=None,
            help="Number of selected dimensions for dimension reduction.")
    parser.add_argument("-if", "--important_features", action="store_true", 
            help="Select important features.")
    parser.add_argument("-in", "--num_important_features", type=int, required=False, default=1,
            help="Number of selected important features per PC.")
    parser.add_argument("-is", "--importance_score", type=float, required=False, default=1.0E-12,
            help="Select important features below this score over 1.0(100%).")
    parser.add_argument("-t", "--transform", action="store_true",
            help="Transform data with PCA after analyses.")
    parser.add_argument("-s", "--std", action="store_true",
            help="Standardize PCA applied data.")
    parser.add_argument("-n", "--normalize", action="store_true",
            help="Normalize PCA applied data.")
    parser.add_argument("-v", "--verbose", action="store_true",
            help="Verbose all logs and prints. Default: False")
    args = parser.parse_args()

    print('Reading DATABASE FILE : ' + str(args.filename))
    sys.stdout.flush()
    data = read_h5(args.filename)
    print('  Database size : ' + ' x '.join([ str(s) for s in data.shape]))
    sys.stdout.flush()

    transform = False
    if args.dim is not None:
        newdim = args.dim
        transform = True
    else:
        newdim = data.shape[1]
        if args.transform:
            transform = True
    if args.std:
        transform = True
    if args.normalize:
        transform = True
    if args.important_features:
        transform = True

    pcadata = PCA_DATA(data, dim=args.dim, verbose=args.verbose, save=args.write_files, save_name=args.output)
    if args.nodata_files:
        pcadata.nodata_files = args.nodata_files
    pcadata.fit()
    if transform:
        pcadata.transform()
        if args.important_features:
            pcadata.importance(score=args.importance_score, 
                               save=args.write_files, 
                               num_if=args.num_important_features)
        if args.std:
            pcadata.standardize(save=args.write_files)
        elif args.normalize:
            pcadata.normalize(save=args.write_files)
        if args.write_files:
            if args.nodata_files is False:
                pcadata.write(args.output+'_FINAL.gz', pcadata.data, msg='Writing '+' x '.join([
                    str(s) for s in np.shape(pcadata.data)
                    ])+' Final PCA DATA to ', verbose=args.verbose)

