import numpy as np
import sys

sys.path.insert(0, '../../../')
sys.dont_write_bytecode = True

from SKDMD.MODEL_SRC.edmd import EDMD
from SKDMD.PREP_DATA_SRC.source_code.lib.utilities import timing

class DEDMD(EDMD):
    """
    Class for Discrete Extended DMD with dictionary as

    .. note::

    """
    def __init__(self, config):
        super(DEDMD, self).__init__(config)
        self.type = 'd'

        if self.dict == 'hermite':
            self.model_dir = self.case_dir + '/' + self.type + '-edmd-h' + str(config['hermite_order']) + '-r' + str(config['reduced_rank'])

        elif self.dict == 'rff_gaussian':
            self.model_dir = self.case_dir + '/' + self.type + '-edmd-rff-' + str(self.rff_number_features) + \
                             '-gaussian_sigma-' + str(self.rff_sigma_gaussian) + '-rank-' + str(config['reduced_rank'])

        elif self.dict == 'rff_gaussian_state':
            self.model_dir = self.case_dir + '/' + self.type + '-edmd-rff-' + str(self.rff_number_features) + \
                             '-gaussian_state_sigma-' + str(self.rff_sigma_gaussian) + '-rank-' + str(config['reduced_rank'])

        elif self.dict == 'nystrom':
            pass

        else:
            raise NotImplementedError('this functionality has not been implemented!!!')

        self.makedir(self.model_dir)

    def gen_grad_dict_dot_f(self, Xdot, X):
        """
        compute the discrete version of gradient of phi dot product with f
        which is simply features at Xnext..

        :type Xdot: np.ndarray
        :param Xdot: next time state

        :type X: np.ndarray
        :param: X: current time state (won't be used)

        :return: phi at next time state
        :rtype: np.ndarray
        """

        return self.gen_dict_feature(Xdot)
