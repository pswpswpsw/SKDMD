import numpy as np
import pickle


class ClassBaseDMD(object):
    """ Base class for ClassModel using EDMD and KDMD """
    def __init__(self, model_path):
        """
        constructor that read model from model_path using pickle

        .. note::
            one would need to *import pickle*

        :type model_path: str
        :param model_path: path for the model
        """
        self.model_path = model_path
        self.model = pickle.load(open(model_path, "rb"))  ## load the model, which is saved with `pickle`
        self.loadKoopman()
        self.normalize = self.model.FLAG['normalize']
        self.scaler = self.model.scaler

        if self.normalize:
            print('normalization is enabled.')

    def get_KoopmanModes(self):
        raise NotImplementedError()

    def loadKoopman(self):
        raise NotImplementedError()


class ClassModelEDMD(ClassBaseDMD):
    """ Class for extended Dynamic mode decomposition (EDMD) model """
    def loadKoopman(self):
        """
        load Koopman matrix, modes and eigenvectors
        """
        self.linearEvolving = self.model.Koopman['Kmatrix']
        self.KoopmanModes = self.model.Koopman['modes']
        self.KoopmanEigenV = self.model.Koopman['eigenvectors']
        # self.residual_matrix = self.model.residual_matrix
        self.Phi_X_i_sample = self.model.Phi_X_i_sample
        self.linearEvolvingEigen = np.diag(self.model.Koopman['eigenvalues'])

        if self.model.type == 'd':
            self.dt = self.model.dt
        elif self.model.type == 'c':
            self.dt = None
        else:
            raise NotImplementedError

    def get_KoopmanModes(self):
        return self.KoopmanModes

    def get_linearEvolving(self):
        return self.linearEvolving

    def get_linearEvolvingEigen(self, selected_modes_index=None):

        if type(selected_modes_index) == type(None):
            result = self.linearEvolvingEigen
        else:
            result = np.diag(np.diag(self.linearEvolvingEigen)[selected_modes_index])

        return result

    def computePhi(self, x):
        """
        compute Phi transformation

        :type x: np.ndarray
        :param x: state
        :return: Phi(x)
        :rtype: np.ndarray
        """

        if self.normalize:
            input = self.model.transform_to_eta(x)
        else:
            input = x
        return self.model.gen_dict_feature(input)

    def compute_eigen_grad_with_gxT(self, gxT, x, index_selected_modes=None):
        return self.model.compute_deigphi_dt(x, gxT, index_selected_modes)

    def computeEigenPhi(self, x, index_selected_modes=None):
        return self.model.compute_eigfun(x, index_selected_modes)

    def reconstruct(self, Phi):
        """
        reconstruct state from Phi

        :type Phi: np.ndarray
        :param Phi: Phi(x)
        :return: state that reconstructed using Koopman modes
        :rtype: np.ndarray
        """

        # first compute compute Koopman eigenfunctions
        eig_func_at_t = np.matmul(Phi, self.KoopmanEigenV)
        # then use Koopman modes to mapping back
        result = np.matmul(eig_func_at_t, self.KoopmanModes)

        if self.normalize:
            result = result + self.scaler.mean_

        return np.real(result)

    def reconstructFromEigenPhi(self, EigenPhi, index_selected_modes=None, kept_koopman_modes=None):

        # first update koopman modes if necessary
        if type(index_selected_modes) == type(None):
            koopman_modes = self.KoopmanModes
        else:
            koopman_modes = kept_koopman_modes

        result = np.matmul(EigenPhi, koopman_modes)

        # then scaling if necessary
        if self.normalize:
            result = result + self.scaler.mean_

        return np.real(result)

    def get_reconstruction_transformation(self, index_selected_modes=None, kept_koopman_modes=None):
        assert self.normalize == False
        if type(index_selected_modes) == type(None):
            koopman_modes = self.KoopmanModes
        else:
            koopman_modes = kept_koopman_modes
        return koopman_modes


class ClassModelKDMD(ClassBaseDMD):
    """
    Class for kernel Dynamic mode decomposition (KDMD) model

    .. note::
        the difference between the implementation of KDMD and EDMD is, in EDMD, computePhi means compute features
        while in KDMD, computePhi means compute eigenfunctions.

    """

    def loadKoopman(self):
        """
        loading Koopman eigenvalues and Koopman modes

        .. note::
            Koopman operator is the diag of Koopman eigenvalues, because in KDMD, I mapped everything into eigenfunction space

        """
        self.linearEvolving = np.diag(self.model.Koopman['eigenvalues'])
        self.KoopmanModes = self.model.Koopman['modes']
        # self.residual_matrix = self.model.residual_matrix
        self.linearEvolvingEigen = self.linearEvolving

        if self.model.type == 'd':
            self.dt = self.model.dt
        elif self.model.type == 'c':
            self.dt = None
        else:
            raise NotImplementedError

    def get_KoopmanModes(self):
        return self.KoopmanModes

    def get_linearEvolving(self):
        """note that it is only used for aposterior evaluation"""
        return self.get_linearEvolvingEigen()

    def get_linearEvolvingEigen(self, selected_modes_index=None):

        if type(selected_modes_index) == type(None):
            result = self.linearEvolvingEigen
        else:
            result = np.diag(np.diag(self.linearEvolvingEigen)[selected_modes_index])
        return result

    def computePhi(self, x, index_modes_select):
        """
        compute Phi transformation

        :type x: np.ndarray
        :param x: state
        :return: Phi(x)
        :rtype: np.ndarray
        """

        return self.model.compute_eigfun(x, index_modes_select)

    def computeEigenPhi(self, x, index_modes_select=None):
        return self.computePhi(x, index_modes_select)

    def reconstructFromEigenPhi(self, EigenPhi, index_selected_modes=None, kept_koopman_modes=None):

        # first update koopman modes if necessary
        if type(index_selected_modes) == type(None):
            koopman_modes = self.KoopmanModes
        else:
            koopman_modes = kept_koopman_modes

        # scaling if necessary
        result = np.matmul(EigenPhi, koopman_modes)

        if self.normalize:
            result = result + self.scaler.mean_

        return np.real(result)


