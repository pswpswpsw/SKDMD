import numpy as np
import sys

sys.path.insert(0, '../../../')
sys.dont_write_bytecode = True

from SKDMD.MODEL_SRC.edmd import EDMD
from scipy.special import hermitenorm


class CEDMD(EDMD):
    """
    Class for Continuous Extended DMD with dictionary as

    .. note::

    """
    def __init__(self, config):

        super(CEDMD, self).__init__(config)

        self.type = 'c'

        if self.dict == 'hermite':
            self.model_dir = self.case_dir  + '/' + self.type + '-edmd-h' + str(config['hermite_order']) + '-r' + str(config['reduced_rank'])
        elif self.dict == 'rff_gaussian':
            self.model_dir = self.case_dir + '/' + self.type + '-edmd-rff-' + str(self.rff_number_features) + \
                             '-gaussian_sigma-'+ str(self.rff_sigma_gaussian) + '-rank-' + str(config['reduced_rank'])
        elif self.dict == 'nystrom':
            pass
        else:
            raise NotImplementedError('this functionality has not been implemented!!!')
        self.makedir(self.model_dir)

    # def get_rff_features_grad_with_gxT(self, x, gxT):
    #     # x is supposed to be shape (1, N_sysdim)
    #     # gxT is supposed to be shape = (N_input, N_sysdim)
    #     return self.gen_rff_features_dot(Xdot=gxT, X=x)

    def gen_rff_features_dot(self, Xdot, X):
        Q = np.matmul(X, self.rff_z)
        M = np.hstack([ -np.sin(Q), np.cos(Q) ]) # since it is the grad...so cos -> -sin..
        R = np.matmul(Xdot, self.rff_z)
        R = np.hstack([R, R])
        Fdot = R*M # elementwise multiplication
        return Fdot

    def compute_deigphi_dt(self, x, xdot, index_selected_modes=None):

        if self.FLAG['normalize']:
            eta_input_matrix = self.transform_to_eta(x)
            etaDot_input_matrix = self.transform_to_eta(xdot)
            xdot_input = etaDot_input_matrix
            x_input = eta_input_matrix
        else:
            xdot_input = xdot
            x_input = x

        if type(index_selected_modes) == type(None):
            deigphi_dt = np.matmul(self.gen_grad_dict_dot_f(xdot_input, x_input), self.Koopman['eigenvectors'])
        else:
            deigphi_dt = np.matmul(self.gen_grad_dict_dot_f(xdot_input, x_input), self.Koopman['eigenvectors'][:,index_selected_modes])

        return deigphi_dt

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

        elif self.dict == 'rff_gaussian':
            generated_gradPhi_dot_f_array = self.gen_rff_features_dot(Xdot,X)

        elif self.dict == 'nystrom':
            pass

        else:
            raise NotImplementedError("the type of " + self.dict + " is not implemented yet!")

        return generated_gradPhi_dot_f_array


