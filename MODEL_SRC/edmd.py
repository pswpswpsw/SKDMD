import scipy.linalg as SLA
import numpy as np
import pickle
import sys

sys.path.insert(0, '../../../')
sys.dont_write_bytecode = True

from SKDMD.MODEL_SRC.dmd import DMD
from SKDMD.PREP_DATA_SRC.source_code.lib.utilities import timing
from scipy.special import hermitenorm


class EDMD(DMD):
    """
    Class for Extended DMD with dictionary as

    - Hermite with each component up to r-order (Williams, 2015)
    - Scalable EDMD (DeGennaro, 2019)

    .. note::
        1. In details, there are two sub-classes for continuous and discrete version
        2. This class is written using Continuous as a template, which means discrete version implied but not explicitly expressed
        in the following function names

    """

    def __init__(self, config):
        super(EDMD, self).__init__(config)

    def set_rff_z(self):
        # rff shared Z
        num_features = int(self.rff_number_features/2)
        self.rff_z = np.random.normal(0, 1.0/self.rff_sigma_gaussian, (self.X.shape[1], num_features)) # note that second argument is scale, so it is sigma^2
        print('set up RFF random feature successful!')

    def gen_dict_feature(self, X):
        """
        generate nonlinear feature given data + nonlinear transformation given
        X = | x_1,.....,x_n @ time_step = 1|
            | x_1,.....,x_n @ time_step = M|
            X.shape = (num_samples, num_components)

        .. note::
            Computational time:
                * The computational time scales with the number of features mainly, which is determined by number of components and polynomial order

        :type X: np.ndarray
        :param X: input data with shape = (num_samples, num_components)

        :return: generated_feature_array with shape (num_samples, num_features)
        :rtype: np.ndarray
        """

        num_sample, num_components = X.shape
        generated_feature_array = None

        if self.dict == 'hermite':
            # normalized hermite polynomial
            ## compute feature list = [[H0(x1).. H0(xn)],...,[HN(x1).. HN(xn)] ]
            feature_list = []
            for order in range(self.hermite_order + 1):
                phi_i = hermitenorm(order)
                phi_i_X = np.polyval(phi_i, X)
                feature_list.append(phi_i_X)

            # create feature array from feature list
            generated_feature_array = self.gen_cross_component_features(
                feature_list=feature_list,
                num_sample=num_sample,
                num_components=num_components
            )

        elif self.dict == 'rff_gaussian':
            # isotropic gaussian kernel exp(-||x||^2/2\sigma)
            generated_feature_array = self.gen_rff_features(X=X)

        elif self.dict == 'nystrom':
            pass

        else:
            raise NotImplementedError("we haven't implemented that!")

        return generated_feature_array

    def gen_rff_features(self, X):
        # the corresponding distribution to sample z from is N(0, sigma^2 I_n)
        # get the big inner product matrix
        Q = np.matmul(X,self.rff_z)

        # get the fourier features
        Fcos = np.cos(Q)
        Fsin = np.sin(Q)

        # stacking the get the total features
        F = np.hstack([Fcos, Fsin])
        assert F.shape == (X.shape[0], self.rff_number_features)

        return F

    def gen_grad_dict_dot_f(self, Xdot, X):
        raise NotImplementedError("you should call it from class cedmd/kdmd! ")

    @staticmethod
    def gen_cross_component_features(feature_list, num_sample, num_components):
        """
        compute cross feature matrix from list
        each row is a flattened feature array

        :type feature_list: list
        :param feature_list: polynomial feature list = [[H0(x1).. H0(xn)],...,[HN(x1).. HN(xn)]]

        :type num_sample: int
        :param num_sample: number of samples in generating the features

        :type num_components: int
        :param num_components: number components in state, i.e., D.O.F.

        :return: generated_feature_array
        :rtype: np.ndarray
        """

        # for each sample, generate cross feature with each compoent up to r-order
        feature_cross_list = []
        for row_sample in range(num_sample):
            feature_list_each_sample = [feature[row_sample, :] for feature in feature_list]
            feature_array_each_sample = np.vstack(feature_list_each_sample)

            feature_row_vector_each_sample = feature_array_each_sample[:, 0].transpose()
            for col_component in range(1, num_components):
                feature_row_vector_each_sample = np.outer(feature_row_vector_each_sample,
                                                          feature_array_each_sample[:, col_component]).flatten()
            feature_cross_list.append(feature_row_vector_each_sample)

        generated_feature_array = np.array(feature_cross_list)
        return generated_feature_array

    def compute_Koopman_analysis(self):
        """compute Koopman eigenvalues, eigenvectors, modes on training data"""

        if self.FLAG['normalize']:
            print('normalized!')
            # compute normalized eta/etaDot
            eta = self.transform_to_eta(self.X)
            etaDot = self.transform_to_etadot(self.Xdot)
            self.Phi_X_i_sample = self.gen_dict_feature(eta)
        else:
            self.Phi_X_i_sample = self.gen_dict_feature(self.X)

        if self.svd_reg:
            ## reg 2, simply SVD on Phi_X....but since it is EDMD, cost will not be very high in the size of feature numbers..
            # so I just mapping to original system
            phi_x_u, phi_x_s, phi_x_vh = np.linalg.svd(self.Phi_X_i_sample, full_matrices=False)

            # condition number
            print('current condition number = ', np.linalg.cond(self.Phi_X_i_sample)/10**12, ' x 10^12')

            # manual truncation based on self.reduced_rank
            phi_x_s[self.reduced_rank:] = 0
            # reconstruct with SVD
            Phi_X_i_sample_svd = phi_x_u @ np.diag(phi_x_s) @ phi_x_vh
            # save singular value spectrum
            _, full_sv, _ = np.linalg.svd(self.Phi_X_i_sample, full_matrices=False)
            np.savez(self.model_dir + '/sv.npz',full_sv=full_sv)

        else:
            # no svd regularization
            Phi_X_i_sample_svd = self.Phi_X_i_sample

        # compute G and A
        if self.FLAG['normalize']:
            Phi_Xdot_i_sample = self.gen_grad_dict_dot_f(etaDot, eta)
        else:
            Phi_Xdot_i_sample = self.gen_grad_dict_dot_f(self.Xdot, self.X)

        print('176:')

        self.num_sample = self.X.shape[0]

        # don't use the textbook solution.. just least square...
        # meanG = np.matmul(Phi_X_i_sample_svd.T, Phi_X_i_sample_svd)/self.num_sample
        # meanA = np.matmul(Phi_X_i_sample_svd.T, Phi_Xdot_i_sample)/self.num_sample

        # save Phi_Y
        self.Phi_Xdot_i_sample = Phi_Xdot_i_sample

        # The size of Koopman matrix depends on the DOF and hermite order
        self.Koopman = {}
        self.Koopman['Kmatrix'] = SLA.lstsq(Phi_X_i_sample_svd, Phi_Xdot_i_sample)[0]
        self.Koopman['eigenvalues'], self.Koopman['eigenvectors'] = np.linalg.eig(self.Koopman['Kmatrix'])

        print('192:')


        # compute residual matrix:
        self.residual_matrix = Phi_Xdot_i_sample - np.matmul(Phi_X_i_sample_svd, self.Koopman['Kmatrix'])
        linear_error_train = np.linalg.norm(self.residual_matrix) / self.residual_matrix.shape[0]

        if not self.FLAG['cv_search_hyp']:
            print('avg.per.samples; avg over all eigenfunctions.. single step linear Koopman residual error on training data = ', linear_error_train)

        ## compute Koopman modes
        # note that Koopman modes are basically telling you the field of each eigenvalues
        # Koopman modes is like the DMD modes..
        psi_X = np.matmul(Phi_X_i_sample_svd, self.Koopman['eigenvectors'])

        if self.FLAG['normalize']:
            self.Koopman['modes'] = SLA.lstsq(psi_X, eta)[0]
            self.Koopman['modes'] = np.matmul(self.Koopman['modes'], np.diag(self.scaler.scale_))  ## it can be complex
        else:
            self.Koopman['modes'] = SLA.lstsq(psi_X, self.X)[0]

        # misc
        self.numKoopmanModes = self.Koopman['Kmatrix'].shape[0]

    def compute_eigfun(self, X_input_matrix, index_selected_modes=None):
        """
        compute Koopman eigenfunction based on input,
        note that input X matrix must be (num_sample_size, num_component) shape

        :param X_input_matrix: input matrix of state with shape (num_sample, num_components)
        :type X_input_matrix: np.ndarray

        :return: Koopman eigenfunction evaluated at X_input_matrix
        :rtype: np.ndarray
        """

        if self.FLAG['normalize']:
            eta_input_matrix = self.transform_to_eta(X_input_matrix)
            Phi_X = self.gen_dict_feature(X=eta_input_matrix)
        else:
            Phi_X = self.gen_dict_feature(X=X_input_matrix)

        if type(index_selected_modes) == type(None):
            result = np.matmul(Phi_X, self.Koopman['eigenvectors'])
        else:
            result = np.matmul(Phi_X, self.Koopman['eigenvectors'][:,index_selected_modes])

        return result

    @timing
    def train(self, X, Xdot, svd_reg=True, dt=None):
        """
        :type X: np.ndarray
        :param X: training derivatives matrix with shape (num_sample, num_components)

        :type Xdot: np.ndarray
        :param Xdot: training state matrix with shape (num_sample, num_components)
        """

        self.X = X
        self.Xdot = Xdot
        self.dt = dt
        self.svd_reg = svd_reg

        # add rff feature generation
        if 'rff' in self.dict:
            self.set_rff_z()

        self.num_sample = self.X.shape[0]
        # self.model_dir = self.model_dir + '/edmd'

        if self.FLAG['normalize']:
            # prepare scaler
            self.prepare_scaler(self.X)

        # compute Koopman matrix and Koopman eigenvalue, eigenvectors
        self.compute_Koopman_analysis()

    def save_model(self):
        # save model, koopman dictionary, as an object, into directory
        # NOTE THAT koopman dictionary is already in the object
        with open(self.model_dir + "/edmd.model", "wb") as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    def compute_deigphi_dt(self, x, xdot, index_selected_modes=None):
        raise NotImplementedError("you should call it from cedmd")

    @timing
    def train_with_valid(self, X, Xdot, X_val, Xdot_val, criterion_threshold=0.05):

        self.criterion_threshold = criterion_threshold
        self.X = X
        self.Xdot = Xdot
        self.X_test = X_val
        self.Xdot_test = Xdot_val

        if 'rff' in self.dict:
            # setting RFF random features
            self.set_rff_z()  #

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
        normal_factor_train = np.sqrt(np.mean(np.abs(eig_phi_train) ** 2, axis=0))
        normal_factor_test = np.sqrt(np.mean(np.abs(eig_phi_test) ** 2, axis=0))

        if self.type == 'c':
            # compute deigen/dt: train and test
            deig_dt_train = self.compute_deigphi_dt(self.X, self.Xdot)
            deig_dt_test = self.compute_deigphi_dt(x_test, x_dot_test)
        elif self.type == 'd':
            # compute the discrete case: train and test
            deig_dt_train = self.compute_eigfun(self.Xdot)
            deig_dt_test = self.compute_eigfun(x_dot_test)
        else:
            raise NotImplementedError("you should call cedmd or dedmd!")

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
        rel_res_test = max_res_test_each_eigen / normal_factor_test

        # compute the index of satified modes for both training and validation data
        index_good_train = np.where(rel_res_train < self.criterion_threshold)[0]
        index_good_test = np.where(rel_res_test < self.criterion_threshold)[0]

        # compute the intersection between the two set of indexes
        num_good_both = len(np.intersect1d(index_good_train, index_good_test))

        # compute average error across all eigens
        linear_error_train = np.mean(rel_res_train)
        linear_error_test = np.mean(rel_res_test)

        return linear_error_train, linear_error_test, num_good_both

