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


