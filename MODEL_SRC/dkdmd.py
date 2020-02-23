import sys
import pickle
import numpy as np

sys.path.insert(0,'../PREP_DATA_SRC')
sys.path.insert(0,'../MODEL_SRC')
sys.path.insert(0, '../../../')

from SKDMD.PREP_DATA_SRC.source_code.lib.utilities import timing
from SKDMD.MODEL_SRC.kdmd import KDMD

class DKDMD(KDMD):

    def __init__(self, config):
        super(DKDMD, self).__init__(config)
        self.type = 'd'
        self.model_dir = self.case_dir + '/' + self.type + '-kdmd-s' + str(config['sigma']) + '-r' + str(config['reduced_rank'])
        self.makedir(self.model_dir)

    def computeAhat(self, X, Xnext):
        Ahat = self.computeKernelArray(Xnext, X)
        return Ahat

    @timing
    def train(self, X, dt):
        # shift data by one, treat xnext as xdot
        self.X = X[:-1,:]
        self.Xdot = X[1:,:]

        # remember dt
        self.dt = dt

        # prepare scaler
        self.prepare_scaler(self.X)

        # compute Koopman tuples
        self.compute_Koopman_analysis()


    def save_model(self):
        # save kdmd model
        with open(self.model_dir + "/kdmd.model", "wb") as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    @timing
    def train_with_valid(self, X, Xdot, X_val, Xdot_val, criterion_threshold=0.05):

        self.criterion_threshold = criterion_threshold
        self.X = X
        self.Xdot = Xdot
        self.X_test = X_val
        self.Xdot_test = Xdot_val

        # prepare scaler
        self.prepare_scaler(self.X)

        # compute Koopman tuples
        self.compute_Koopman_analysis()

        # compute linear loss on training and testing data
        linear_error_train, linear_error_test, num_good_both = self.compute_linear_loss_on_testing_data(self.X_test, self.Xdot_test)

        return linear_error_train, linear_error_test, num_good_both


    def compute_linear_loss_on_testing_data(self, x_test, x_dot_test):

        # compute eigen: train and test
        eig_phi_train = self.compute_eigfun(self.X)
        eig_phi_test  = self.compute_eigfun(x_test)

        # compute sqrt(1/M sum phi_i^2 (x_k))
        normal_factor_train = np.sqrt(np.mean(np.abs(eig_phi_train)**2, axis=0))
        normal_factor_test  = np.sqrt(np.mean(np.abs(eig_phi_test )**2, axis=0))

        # compute deigen/dt: train and test
        next_step_eig_phi_train = self.compute_eigfun(self.Xdot)
        next_step_eig_phi_test  = self.compute_eigfun(x_dot_test)

        # compute linear dynamics loss (first half of it, and normalized)
        res_train = next_step_eig_phi_train - np.matmul(eig_phi_train, np.diag(self.Koopman['eigenvalues']))
        res_test  = next_step_eig_phi_test  - np.matmul(eig_phi_test,  np.diag(self.Koopman['eigenvalues']))

        # compute max error for all samples
        res_train = np.abs(res_train)
        res_test = np.abs(res_test)

        # try median or mean?
        # max_res_train_each_eigen = np.max(res_train, axis=0)
        # max_res_test_each_eigen = np.max(res_test, axis=0)
        max_res_train_each_eigen = np.mean(res_train, axis=0)
        max_res_test_each_eigen = np.mean(res_test, axis=0)


        # compute relative error: note that here it is a array/array operation
        rel_res_train = max_res_train_each_eigen / normal_factor_train
        rel_res_test  = max_res_test_each_eigen  / normal_factor_test

        index_good_train = np.where(rel_res_train < self.criterion_threshold)[0]
        index_good_test = np.where(rel_res_test < self.criterion_threshold)[0]

        # this is a good indicator, since it detects the number of modes that behaves well under a certain threshold...
        num_good_both = len(np.intersect1d(index_good_train, index_good_test))

        ## use mean of the relative error across all eigens as indicator
        # comment: it is a bad indicator because, it is using all the eigenmodes...so if one is bad, the whole thing is bad...
        linear_error_train = np.mean(rel_res_train)
        linear_error_test = np.mean(rel_res_test)

        return linear_error_train, linear_error_test, num_good_both

