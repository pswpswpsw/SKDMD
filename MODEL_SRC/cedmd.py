import scipy.linalg as SLA
import numpy as np
import pickle
import sys

sys.path.insert(0, '../../../')
sys.dont_write_bytecode = True

from SKDMD.MODEL_SRC.dmd import DMD
from SKDMD.PREP_DATA_SRC.source_code.lib.utilities import timing
from scipy.special import hermitenorm


class CEDMD(DMD):
    """
    Class for Extended DMD with dictionary as

    - Hermite with each component up to r-order (Williams, 2015)

    .. note::
        Important things to remember
            * this class does not assume a subset of non-trainable, which means, the reconstructing problem is not of our concern, we will not deal with recontructing problem.

            * With that being said, Koopman modes are still computed by 2014 Willams paper in kernel-based EDMD, which is simply a pseudo-inverse

    """

    def __init__(self, config):
        super(CEDMD, self).__init__(config)
        self.type = 'c'
        self.model_dir = self.case_dir  + '/' + self.type + '-edmd-h' + str(config['hermite_order']) + '-r' + str(config['reduced_rank'])
        self.makedir(self.model_dir)

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

        return generated_feature_array

    def gen_grad_dict_dot_f(self, Xdot, X):
        """
        compute the gradient of phi dot product with f

        :type Xdot: np.ndarray
        :param Xdot: time derivative of state

        :return: generated_gradPhi_dot_f_array
        :rtype: np.ndarray
        """

        num_sample, num_components = Xdot.shape

        if self.dict == 'hermite':
            # normalized hermite polynomial
            ## compute [ [d[H0(x1).. H0(xn)]/dx1,...,d[HN(x1).. HN(xn)]/dx1 ],
            #            ...
            #            [d[H0(x1).. H0(xn)]/dxn,...,d[HN(x1).. HN(xn)]/dxn ] ]
            generated_feature_array_list = []
            feature_list_ddx_list = []
            for i_component in range(num_components):
                feature_list = []
                for order in range(self.hermite_order + 1):
                    phi_i = hermitenorm(order)
                    phi_i_dx = np.poly1d.deriv(phi_i)
                    phi_i_X = np.polyval(phi_i, X)
                    # update i_component with the derivative one
                    phi_i_X[:, i_component] = np.polyval(phi_i_dx, X[:, i_component])
                    feature_list.append(phi_i_X)
                feature_list_ddx_list.append(feature_list)

                # generate feature array from feature list for each i_component
                generated_feature_array = self.gen_cross_component_features(
                    feature_list=feature_list,
                    num_sample=num_sample,
                    num_components=num_components
                )

                # dot product f with the gradient
                Xdot_i_component = Xdot[:, i_component]
                Xdot_i_matrix = np.diag(Xdot_i_component)
                generated_feature_array_list.append(np.matmul(Xdot_i_matrix, generated_feature_array))

            # summing up the dot product for each component
            generated_gradPhi_dot_f_array = np.sum(generated_feature_array_list, axis=0)

        else:
            raise NotImplementedError("the type of " + self.dict + " is not implemented yet!")

        return generated_gradPhi_dot_f_array

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

        ## reg 2, simply SVD on Phi_X....but since it is EDMD, cost will not be very high in the size of feature numbers..
        # so I just mapping to original system
        phi_x_u, phi_x_s, phi_x_vh = np.linalg.svd(self.Phi_X_i_sample, full_matrices=False)
        # manual truncation based on self.reduced_rank
        phi_x_s[self.reduced_rank:] = 0
        # reconstruct with SVD
        Phi_X_i_sample_svd = np.matmul(np.matmul(phi_x_u, np.diag(phi_x_s)), phi_x_vh)
        # save singular value spectrum
        _, full_sv, _ = np.linalg.svd(self.Phi_X_i_sample)
        np.savez(self.model_dir + '/sv.npz',full_sv=full_sv)

        # compute G and A
        if self.FLAG['normalize']:
            Phi_Xdot_i_sample = self.gen_grad_dict_dot_f(etaDot, eta)
        else:
            Phi_Xdot_i_sample = self.gen_grad_dict_dot_f(self.Xdot, self.X)

        meanG = np.matmul(Phi_X_i_sample_svd.T, Phi_X_i_sample_svd)/self.num_sample
        meanA = np.matmul(Phi_X_i_sample_svd.T, Phi_Xdot_i_sample)/self.num_sample

        # save Phi_Y
        self.Phi_Xdot_i_sample = Phi_Xdot_i_sample

        # The size of Koopman matrix depends on the DOF and hermite order
        self.Koopman = {}
        self.Koopman['Kmatrix'] = SLA.lstsq(meanG, meanA)[0]
        self.Koopman['eigenvalues'], self.Koopman['eigenvectors'] = np.linalg.eig(self.Koopman['Kmatrix'])

        # compute residual matrix:
        self.residual_matrix = Phi_Xdot_i_sample - np.matmul(Phi_X_i_sample_svd, self.Koopman['Kmatrix'])

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
    def train(self, X, Xdot):
        """
        :type X: np.ndarray
        :param X: training derivatives matrix with shape (num_sample, num_components)

        :type Xdot: np.ndarray
        :param Xdot: training state matrix with shape (num_sample, num_components)
        """

        self.X = X
        self.Xdot = Xdot
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



