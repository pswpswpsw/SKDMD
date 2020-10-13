import sys
import numpy as np

sys.path.insert(0, '../../../')

from SKDMD.PREP_DATA_SRC.source_code.lib.utilities import mkdir
from sklearn.preprocessing import StandardScaler

class DMD(object):
    """ Base class for Extended DMD and Kernel DMD"""

    def __init__(self, config):

        # model type

        self.type = None

        # KDMD attributes

        self.kernel              = config.get('kernel', None)
        self.sigma_gaussian      = config.get('sigma', None)

        # EDMD attributes

        self.dict                = config.get('dict', None)
        self.power      		 = config.get('power', None) # also can be used by KDMD
        self.hermite_order       = config.get('hermite_order', None)

        ## EDMD-RFF attributes

        self.rff_number_features = config.get('n_rff_features', None)
        self.rff_sigma_gaussian  = config.get('rff_gaussian_sigma', None)

        # misc attributes

        self.case_dir            = './' + config.get('model_case', None)
        self.phase_space_range   = config.get('phase_space_range', None)
        self.reduced_rank        = config.get('reduced_rank', 2)
        self.numKoopmanModes = None
        self.Koopman = None
        self.X = None
        self.Xdot = None

        self.Xdot_test = None
        self.X_test = None

        self.FLAG = {'normalize': False,  #whether to normalize the data
                     'cv_search_hyp': config.get('cv_search_hyp', False)}

        self.cut_off_cond = 2e14
        self.criterion_threshold = 0.05

        self.model_dir = None

        self.scaler = StandardScaler()
        self.dt = None

    def prepare_scaler(self, X):
        self.scaler.fit(X)

    def transform_to_eta(self, X):
        return self.scaler.transform(X)

    def transform_to_etadot(self, Xdot):
        return np.matmul(Xdot, np.diag(1./self.scaler.scale_))

    def transform_to_x(self, eta):
        return self.scaler.inverse_transform(eta)

    @staticmethod
    def makedir(dir):
        mkdir(dir)

    def compute_eigfun(self, X_input_matrix, index_selected_modes=None):
        raise NotImplementedError("computing Koopman eigenfunction is not implemented yet")

    def plot_Koopman_eigen_contour(self):
        """plot Koopman eigenfunction contour"""

        # only draw Koopman eigenfunctions for 2D phase space
        if self.X.shape[1] == 2:

            # set Koopman eigenvalues
            D_real = np.real(self.Koopman['eigenvalues'])
            D_imag = np.imag(self.Koopman['eigenvalues'])

            # distribute sampling point in physical space
            x1_min, x1_max = self.phase_space_range[0]
            x2_min, x2_max = self.phase_space_range[1]

            ndraw = 100
            ndrawj = ndraw * 1j
            x1_, x2_ = np.mgrid[x1_min:x1_max:ndrawj, x2_min:x2_max:ndrawj]

            sample_x = x1_.reshape(-1, 1)
            sample_xdot = x2_.reshape(-1, 1)

            x_sample = np.hstack((sample_x, sample_xdot))  # make it (?,2) size

            assert x_sample.shape == (ndraw ** 2, 2), "sampling shape is wrong, check visualizing eigenfunction!"

            # compute Koopman eigenfunction
            phi_eigen_array = self.compute_eigfun(X_input_matrix=x_sample)

            np.savez(self.model_dir + '/koopman_eigenfunctions.npz',
                     numKoopmanModes=self.numKoopmanModes,
                     phi_eigen_array=phi_eigen_array,
                     ndraw=ndraw,
                     D_real=D_real,
                     D_imag=D_imag,
                     x1_=x1_,
                     x2_=x2_
                     )
        else:
            print('nominal dimension of the system not 2D, eigenfunction cannot be plotted in contour!')


