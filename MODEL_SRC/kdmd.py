import scipy.linalg as SLA
from scipy.sparse.linalg import eigs, eigsh
import numpy as np
# from matplotlib import pyplot as plt
import sys
sys.path.insert(0, '../../../')

from scipy.spatial.distance import cdist
from SKDMD.MODEL_SRC.dmd import DMD
from decimal import Decimal

sys.dont_write_bytecode = True

class KDMD(DMD):

    def computeElementwisePower(self, x):
        """
        compute power kernel elementwise
        :type x: np.ndarray or float, int
        :param x:
        :return:
        """
        return np.power( (1 + x), self.power)

    def computeKernelArray(self, X, Y):

        if self.kernel == 'linear':

            KXY = np.matmul(X, np.transpose(Y))

        elif self.kernel == 'gaussian':

            pairwise_dists = cdist(X, Y, 'euclidean')
            KXY = np.exp(-np.square(pairwise_dists / self.sigma_gaussian))

        elif self.kernel == 'polynomial':

            KXY = np.matmul(X, np.transpose(Y))
            KXY = self.computeElementwisePower(KXY)

        else:
            raise NotImplementedError()

        return KXY

    def computeGhat(self, X):
        """
        compute Ghat in KDMD
        """

        Ghat = self.computeKernelArray(X,X)

        return Ghat

    def check_symmetric(self, x, tol=1e-8):
        """
        check if the matrix is symmetric
        """
        return np.allclose(x, x.transpose(), atol=tol)

    def computeAhat(self, X, Xdot):
        raise NotImplementedError("KDMD class itself alone doesn't contain the method to compute Ahat!")

    def compute_Koopman_analysis(self):
        """
        compute Koopman analysis, with dict as **self.Koopman** with

        - modes
        - Kmatrix
        - eigenvectorHat
        - eigenvalues

        """

        if self.FLAG['normalize']:
            eta = self.transform_to_eta(self.X)
            etaDot = self.transform_to_etadot(self.Xdot)

        # compute G_hat

        if self.FLAG['normalize']:
            # compute normalized eta/etaDot
            self.Ghat = self.computeGhat(eta)
        else:
            self.Ghat = self.computeGhat(self.X)

        # print("Ghat = ",self.Ghat)

        # check if matrix is hermitiain, but note that it might not be P.S.D.
        assert self.check_symmetric(self.Ghat) == True, 'Ghat is not symmetric!'

        ## SAVE full specturm

        # compute Q, sigma
        M = self.X.shape[0]

        ## new implementation
        self.sigma, self.Q = eigsh(self.Ghat, k=self.reduced_rank, which='LM',maxiter=1e5) # search largest
        # reduced_rank components
        
        print("current reduced rank = ", self.reduced_rank)
        print('current condition number = ', '%.2E' % Decimal(np.max(self.sigma)/np.min(self.sigma)), ' for rank = ', self.reduced_rank)

        # if condition number exceeds...
        if abs(np.max(self.sigma)/np.min(self.sigma)) > self.cut_off_cond:
            ## check to find cut off index
            condition_numb_array = abs(np.max(self.sigma) / self.sigma)
            condition_numb_array = condition_numb_array[::-1]
            index_array = condition_numb_array < self.cut_off_cond
            new_num_reduced_rank = np.where(index_array == False)[0][0]
            self.reduced_rank = new_num_reduced_rank

            ## redo implementation
            self.sigma, self.Q = eigsh(self.Ghat, k=self.reduced_rank,maxiter=1e4) # search largest reduced_rank components 
   
            print('updated condition number = ', '%.2E' % Decimal(np.max(self.sigma)/np.min(self.sigma)), ' for rank = ', self.reduced_rank)

        ## old implementation
        full_sv_squared, _ = SLA.eigh(self.Ghat)
        full_sv_squared = np.abs(full_sv_squared)  ## it can be non-posiitive...

        np.savez(self.model_dir + '/sv_squared.npz', full_sv_squared=full_sv_squared)

        # make sure our K matrix will be in a reduced size!
        assert self.sigma.size == self.reduced_rank, "sigma size = " + str(self.sigma.size) + "while reduced rank = " + str(self.reduced_rank)

        # Square root out ....
        self.sigma = np.diag(np.sqrt(np.abs(self.sigma)))  ## prevent non-positive issues
        # Ghat = Q diag(sigma)^2 Q^T

        # compute inverse of self.sigma
        self.inverse_sigma = np.diag(1.0/np.diag(self.sigma))

        # print 'suggested cut off: ',

        # compute A_hat
        if self.FLAG['normalize']:
            self.Ahat = self.computeAhat(eta, etaDot)
        else:
            self.Ahat = self.computeAhat(self.X, self.Xdot)

        ## new implementation
        r1 = np.matmul(self.inverse_sigma, self.Q.T)
        self.Khat = np.matmul(np.matmul(r1, self.Ahat), r1.T)

        # compute linear error
        b1 = np.matmul(self.Q, self.sigma)
        b2 = np.matmul(np.matmul(b1, self.Khat), b1.T)

        linear_error_train = np.linalg.norm(b2 - self.Ahat)

        if not self.FLAG['cv_search_hyp']:
            print('single step linear Koopman residual error on training data = ', linear_error_train)

        # summary koopman
        self.Koopman = {}
        self.Koopman['Kmatrix'] = self.Khat  # indeed it was Khat
        self.Koopman['eigenvalues'], self.Koopman['eigenvectorHat'] = np.linalg.eig(self.Khat)

        ## new implementation
        Phi_X = np.matmul(np.matmul(self.Q, self.sigma),self.Koopman['eigenvectorHat'])

        if self.FLAG['normalize']:
            result = SLA.lstsq(Phi_X, eta)
            self.Koopman['modes'] = np.matmul(result[0], np.diag(self.scaler.scale_))  ## it can be complex
        else:
            result = SLA.lstsq(Phi_X, self.X)
            self.Koopman['modes'] = result[0]  ## it can be complex

        if not self.FLAG['cv_search_hyp']:
            print('residuals of KDMD reconstruction = ', np.linalg.norm(np.matmul(Phi_X, result[0]) - self.X))

        # misc
        self.numKoopmanModes = self.Koopman['Kmatrix'].shape[0]


    def compute_eigfun(self, X_input_matrix, index_modes_select=None):
        """compute Koopman eigenfunction for KDMD"""

        if self.FLAG['normalize']:
            eta_input_matrix = self.transform_to_eta(X_input_matrix)
            eta_X = self.transform_to_eta(self.X)
            KTestXX = self.computeKernelArray(eta_input_matrix, eta_X)
        else:
            KTestXX = self.computeKernelArray(X_input_matrix, self.X)

        # X = SLA.lstsq(self.sigma.T, np.matmul(KTestXX, self.Q).T)[0].T
        C = np.matmul(np.matmul(KTestXX, self.Q), self.inverse_sigma)

        if type(index_modes_select) == type(None):
            phi_eigen = np.matmul(C, self.Koopman['eigenvectorHat'])
        else:
            phi_eigen = np.matmul(C, self.Koopman['eigenvectorHat'][:, index_modes_select])

        return phi_eigen



