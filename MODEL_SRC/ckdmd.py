import pickle
import sys
import numpy as np

sys.path.insert(0, '../../../')

from SKDMD.MODEL_SRC.kdmd import KDMD
from SKDMD.PREP_DATA_SRC.source_code.lib.utilities import timing


class CKDMD(KDMD):
    """
    Class for Kernel DMD

    with kernel as
    * Gaussian kernel
    * polynomial kernel
    * linear kernel DMD

    """

    def __init__(self, config):
        super(CKDMD, self).__init__(config)
        self.type = 'c'
        self.model_dir = self.case_dir + '/' + self.type + '-kdmd-s' +  str(config['sigma']) + '-r' + str(config['reduced_rank'])
        self.makedir(self.model_dir)

    def compute_deigphi_dt(self, x, xdot):

        # compute Ahat between x and self.X
        if self.kernel == 'linear':
            Ahat = self.computeKernelArray(xdot, self.X)
        elif self.kernel == 'gaussian':
            # dot_x^i_k * (x^i_k - x^j_k) scalar field from inner product
            Ahat_1 = np.tensordot(np.ones(self.X.shape[0]), xdot, axes=0)
            Z = np.tensordot(np.ones(self.X.shape[0]), x, axes=0)
            Z2 = np.tensordot(np.ones(x.shape[0]), self.X, axes=0)
            ZT = np.transpose(Z2,axes=(1,0,2))
            ZV = Z - ZT
            Ahat_2 = np.einsum('ijk,ijk->ji',Ahat_1,ZV)
            # elementwise multiplication with the last kernel thing
            newGhat = self.computeKernelArray(x, self.X)
            Ahat = Ahat_2 * newGhat * -2.0 / (self.sigma_gaussian**2)
        elif self.kernel == 'polynomial':
            Ahat_1 = np.matmul(xdot, np.transpose(self.X))
            newGhat = self.computeKernelArray(x, self.X)
            Ahat = self.power * np.power(newGhat, (self.power - 1)/self.power) * Ahat_1
        else:
            raise NotImplementedError("this kernel: " + str(self.kernel) + " is not implemented!")

        # then compute deigen_phi_dt
        deigen_phi_dt = np.matmul(np.matmul(np.matmul(Ahat, self.Q), self.inverse_sigma), self.Koopman['eigenvectorHat'])

        return deigen_phi_dt

    def computeAhat(self, X, Xdot):

        if self.kernel == 'linear':
            Ahat = np.matmul(Xdot, np.transpose(X))
        elif self.kernel == 'gaussian':
            # dot_x^i_k * (x^i_k - x^j_k) scalar field from inner product
            Ahat_1 = np.tensordot(np.ones(Xdot.shape[0]), Xdot, axes=0)
            Z = np.tensordot(np.ones(X.shape[0]), X, axes=0)
            ZT = np.transpose(Z,axes=(1,0,2))
            ZV = Z - ZT
            Ahat_2 = np.einsum('ijk,ijk->ji',Ahat_1,ZV)
            # elementwise multiplication with the last kernel thing
            Ahat = Ahat_2 * self.Ghat * -2.0 / (self.sigma_gaussian**2)
        elif self.kernel == 'polynomial':
            Ahat_1 = np.matmul(Xdot, np.transpose(X))
            Ahat = self.power * np.power(self.Ghat, (self.power - 1)/self.power) * Ahat_1
        else:
            raise NotImplementedError("this kernel: " + str(self.kernel) + " is not implemented!")

        return Ahat

    @timing
    def train_with_valid(self, X, Xdot, X_val, Xdot_val,criterion_threshold=0.05):

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
        linear_error_train, linear_error_test, num_good_both_train_valid = self.compute_linear_loss_on_testing_data(self.X_test, self.Xdot_test)

        return linear_error_train, linear_error_test, num_good_both_train_valid

    def compute_linear_loss_on_testing_data(self, x_test, x_dot_test):

        # compute eigen: train and test
        eig_phi_train = self.compute_eigfun(self.X)
        eig_phi_test = self.compute_eigfun(x_test)

        # compute sqrt(1/M sum phi_i^2 (x_k))
        normal_factor_train = np.sqrt(np.mean(np.abs(eig_phi_train)**2, axis=0))
        normal_factor_test = np.sqrt(np.mean(np.abs(eig_phi_test)**2, axis=0))

        # compute deigen/dt: train and test
        deig_dt_train = self.compute_deigphi_dt(self.X, self.Xdot)
        deig_dt_test = self.compute_deigphi_dt(x_test, x_dot_test)

        # compute linear dynamics loss (first half of it, and normalized)
        res_train = deig_dt_train - np.matmul(eig_phi_train, np.diag(self.Koopman['eigenvalues']))
        res_test = deig_dt_test - np.matmul(eig_phi_test, np.diag(self.Koopman['eigenvalues']))

        # compute max error for all samples
        res_train = np.abs(res_train)
        res_test = np.abs(res_test)

        # switch to median, makes more sense...for accounting the number of good functions...
        max_res_train_each_eigen = np.mean(res_train, axis=0)
        max_res_test_each_eigen = np.mean(res_test, axis=0)

        # compute relative error
        rel_res_train = max_res_train_each_eigen / normal_factor_train
        rel_res_test  = max_res_test_each_eigen  / normal_factor_test

        # compute the index of satified modes for both training and validation data
        index_good_train = np.where(rel_res_train < self.criterion_threshold)[0]
        index_good_test = np.where(rel_res_test < self.criterion_threshold)[0]

        # compute the intersection between the two set of indexes
        num_good_both = len(np.intersect1d(index_good_train, index_good_test))

        # compute average error across all eigens
        linear_error_train = np.mean(rel_res_train)
        linear_error_test = np.mean(rel_res_test)

        return linear_error_train, linear_error_test, num_good_both

    @timing
    def train(self, X, Xdot):
        """
        Given X and Xdot, training for Koopman eigenfunctions, eigenvalues, eigenvectors, and Koopman modes

        :type X: np.ndarray
        :param X: state of the system

        :type Xdot: np.ndarray
        :param Xdot: time derivative of the state of the system

        """

        self.X = X
        self.Xdot = Xdot
        # prepare scaler
        self.prepare_scaler(self.X)
        # compute Koopman tuples
        self.compute_Koopman_analysis()

    def save_model(self):
        # save kdmd model
        with open(self.model_dir + "/kdmd.model", "wb") as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

