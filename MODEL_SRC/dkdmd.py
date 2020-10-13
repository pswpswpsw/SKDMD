import sys
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

