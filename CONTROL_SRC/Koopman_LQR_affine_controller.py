import sys
import pickle
import numpy as np
import re
from scipy.linalg import logm
from scipy.io import savemat
from scipy.linalg import solve_continuous_are as SCARE
from scipy.linalg import solve_discrete_are as SDARE
from control import ctrb

sys.dont_write_bytecode = True
sys.path.insert(0, '../../')
sys.path.append('../../EVAL_SRC')

from main_apo import ClassModelEDMD

# from SKDMD.EVAL_SRC.main_apo import ClassModelEDMD
# from SKDMD.EVAL_SRC.main_apo import ClassApoEval

class ClassVanillaKoopmanLQR_Controller(object):

    def __init__(self, model_path):
        model_path_file = model_path + 'edmd.model'
        self.model = ClassModelEDMD(model_path=model_path_file)
        # get Koopman linear representation
        # phi_n+1 = A * phi_n + Bu * u
        self.A = self.get_A()
        self.Bu = self.get_Bu()
        # x_n = B * phi_n
        self.B = self.get_B()

    def set_Q_and_R(self, Q, R):
        self.Q = Q
        self.Qphi = self.B.T @ self.Q @ self.B
        self.R = R
        self.Rinv = np.linalg.inv(self.R)

    def get_A(self):
        return self.model.linearEvolving.T

    def get_B(self):
        KoopmanEigenV = self.model.KoopmanEigenV
        KoopmanModes = self.model.KoopmanModes
        Brec = np.real(KoopmanEigenV @ KoopmanModes)
        # note, here I didn't consider normalization, which is rare in EDMD.
        return Brec.T

    def get_Bu(self):
        return self.model.model.Koopman['Bmatrix'].T

    def get_phi(self, x):
        return self.model.computePhi(x.T).T

    def _check_if_controllable(self):
        C = ctrb(self.A, self.Bu)
        print('num. unstable = ', np.sum(np.abs(np.linalg.eig(self.A)[0])>1))
        rank = np.linalg.matrix_rank(C)
        # assert rank == self.A.shape[0]
        print('ctrb rank = ',rank)
        print('required rank = ',self.A.shape[0])

    def solve_are(self):
        # test if controllable.
        self._check_if_controllable()
        # then solve SCARE
        ## debug save as mat
        # savemat('abqr.mat',{'A':self.A,'B':self.Bu,'Q':self.Qphi,'R':self.R})

        ## solve discrete-time ARE...
        self.P = SDARE(self.A, self.Bu, self.Qphi, self.R)

        # finally get u_lqr following formula
        pre_factor = -1.0 * self.Rinv @ self.Bu.T
        self.pre_factor = pre_factor @ self.P

        print('solved ! Prefactor = \n', self.pre_factor)

    def feedback(self, x, x_ref):
        # compute phi_ref
        phi_ref = self.get_phi(x_ref.reshape(-1,1))
        phi = self.get_phi(x.reshape(-1,1))
        u_lqr = np.matmul(self.pre_factor, phi - phi_ref)
        u_lqr = u_lqr.ravel()
        return u_lqr






class Class_SDLQR_Controller(object):

    def __init__(self, model_dir, best_index):
        eval_model_pickle_path = '../EXAMPLES/' + model_dir + 'apoeval.model'
        self.eval_model = pickle.load(open(eval_model_pickle_path, "rb"))

        # manage to get the true directory of that model folder
        self.best_index = best_index
        self.eval_model_save_dir = '../EXAMPLES/' + re.split("/", model_dir)[0] + self.eval_model.save_dir[1:]

        self.cont_A_matrix = None
        self.cont_A_matrix_real = None
        self.Q = None
        self.Qinv = None

        # initialize the best case
        self._initialize_best_index_data()

        # check conjugacy
        self.FLAG = self._check_conjugacy()

    def _initialize_best_index_data(self):
        self.eval_model.load_best_index_data(self.best_index, self.eval_model_save_dir)

    def get_cont_A_matrix(self):
        A = self.eval_model.sparse_compute_Lambda_at_best_index()
        if self.eval_model.type == 'c':
            A_matrix = A
        elif self.eval_model.type == 'd':
            A_matrix = logm(A)*1.0/self.eval_model.dt
        else:
            raise NotImplementedError
        self.cont_A_matrix = A_matrix
        return A_matrix

    def _check_conjugacy(self):
        self.get_cont_A_matrix()
        A_row = np.diag(self.cont_A_matrix)
        FLAG = True

        for i in range(len(A_row)):
            if np.imag(A_row[i]) == 0:
                pass
            else:
                try:
                    FLAG = np.conjugate(A_row[i]) == A_row[i+1]
                except:
                    FLAG = False

        if FLAG:
            print('pass! conjugacy satisfy for best index = ', self.best_index)
        else:
            print('failure! choose another best index instead of ', self.best_index)

        return FLAG

    def perform_Koopman_Canonical_Transformation(self):
        assert self.FLAG == True
        # Goal find Q:
        # phi_dot = phi * A
        # phi_hat = phi * Q
        # phi_hat_dot = phi_hat * Q = phi * A * Q ---> find D SUCH THAT = phi * Q * D
        # obviously: D = Q^{-1} A * Q

        Q = np.zeros(self.cont_A_matrix.shape)
        j=0
        while j < self.cont_A_matrix.shape[0]:
            if np.imag(self.cont_A_matrix[j, j]) == 0:
                Q[j,j]=1
                j=j+1
            else:
                Q[j,j]   = 0.5
                Q[j,j+1] = -0.5
                Q[j+1,j] = 0.5
                Q[j+1,j] = 0.5
                j=j+2

        self.Q = Q
        self.Qinv = np.linalg.inv(self.Q)
        # get real -ized  A matrix
        self.cont_A_matrix_real = np.matmul(self.Qinv, np.matmul(self.cont_A_matrix,Q))
        assert np.linalg.norm(np.imag(self.cont_A_matrix_real)) < 1e-10
        self.cont_A_matrix_real = np.real(self.cont_A_matrix_real)

    def lifting(self, state):
        eigenfunction_value = self.eval_model.sparse_compute_eigen_at_best_index(state)
        realized_eigenfunction_value = np.matmul(eigenfunction_value,self.Q)

        # since it is realized...
        realized_eigenfunction_value = np.real(realized_eigenfunction_value)

        return realized_eigenfunction_value

    def get_cont_A_matrix_real(self):
        return self.cont_A_matrix_real

    def unlifting(self, realized_eigenfunction_value):
        eigenfunction_value = np.matmul(realized_eigenfunction_value, self.Qinv)
        state_re = self.eval_model.sparse_compute_reconstruct_at_best_index(eigenfunction_value)
        return state_re

    def get_unlifting_transformation_matrix(self):
        # here B should be complex.. since it is the matrix from original eigen-decomposd complex koopman...
        B = self.eval_model.sparse_compute_transformation_matrix_at_best_index()
        B = np.matmul(self.Qinv, B)
        # since it is realized...
        B = np.real(B)
        return B

    def get_actuator_aux_state_dependent_matrix(self, gxT, x):
        state_stack = np.empty((gxT.shape[0], x.shape[1]))
        for i in range(gxT.shape[0]):
            state_stack[i,:] = x
        return self.eval_model.sparse_compute_actuator_aux_state_dependent_matrix_at_best_index(gxT, state_stack)

    def _test_get_eigen_linear_system(self, state_ic, tspan):
        eigenfunction_value = self.eval_model.sparse_compute_eigen_at_best_index(state_ic)
        A = self.eval_model.sparse_compute_Lambda_at_best_index()
        state_ic_re = self.eval_model.sparse_compute_reconstruct_at_best_index(eigenfunction_value)
        B = self.eval_model.sparse_compute_transformation_matrix_at_best_index()


        return eigenfunction_value, A, state_ic_re,B


if __name__ == '__main__':

    # N_TEST = 1 # unit test: simple lusch problem with pruning on the EDMD
    N_TEST = 2 # unit test: 8d system with vanilla scalable EDMD

    if N_TEST == 1:

        model_dir = 'lusch/2d_simple_lusch2017_noise_level_0/c-edmd-h5-r36/'
        controller = Class_SDLQR_Controller(model_dir, best_index=30)

        state_ic = np.array([[0.2, 0.2]])
        tspan = np.linspace(0, 1, 100)

        # have access to eigenstate (initial), diagonal A matrix (complex), and reconstructed state for sanity check..
        eigen, A, state_ic_re, B = controller._test_get_eigen_linear_system(state_ic, tspan) # return numpy object


        print('eigen = ',eigen)
        print('eigen shape = ',eigen.shape)
        print('A = ',np.diag(A))
        print('A shape = ',A.shape)
        print('state_ic_re = ',state_ic_re)


        # realized the system
        controller.perform_Koopman_Canonical_Transformation()
        eigen_ic = controller.lifting(state_ic)
        A_real = controller.get_cont_A_matrix_real()
        state_ic_re = controller.unlifting(eigen_ic)

        print('')
        print('eigen realized = ', eigen_ic)
        print('A realized = ', A_real)
        print('state recovered = ', state_ic_re)

        ## validate the reconstructed matrix
        B = controller.get_unlifting_transformation_matrix()
        error = np.linalg.norm(state_ic - np.matmul(eigen_ic, B))
        print('reconstructed with manually matrix = ',np.matmul(eigen_ic, B))
        print('error should be small ',error)

        # final test .. just as a simple case for the gxT
        # controller.get_actuator_aux_state_dependent_matrix(gxT, x)
        gxT = np.asarray([[1.0, 1.0], [2.0,2.0]])
        print('gxT shape = ',gxT.shape)
        BuT = controller.get_actuator_aux_state_dependent_matrix(gxT, state_ic)
        print('BuT shape = ',BuT.shape)

        # final final test...solving LQR..

    elif N_TEST == 2:

        # this model is 500 features
        model_dir = '../EXAMPLES/8d_fjr_with_random_input/8d_flex_mani_with_random_input_noise_level_0/' \
                    'd-edmd-rff-40-gaussian_sigma-100.0-rank-40/'

        # this model is state inclusive, with 58
        # model_dir = '../EXAMPLES/8d_fjr_with_random_input/8d_flex_mani_with_random_input_noise_level_0/' \
        #             'd-edmd-rff-200-gaussian_state_sigma-150-rank-208/'
        #

        # this is a EDMD model with first order polynomial but comination among 8 components
        # model_dir = '../EXAMPLES/8d_fjr_with_random_input/8d_flex_mani_with_random_input_noise_level_0/' \
        #             'd-edmd-h1-r512/'

        controller = ClassVanillaKoopmanLQR_Controller(model_dir)

        Q = np.eye(8) # 8-system
        R = np.eye(2) # 2-input
        xref = np.zeros((8,1))

        controller.set_Q_and_R(Q, R)
        controller.solve_are()
        # x = np.ones((8,1))
        x = xref
        control_input = controller.feedback(x,xref)

        print('shape of control input = ')
        print(control_input.shape)
        print('value of control input = ')
        print(control_input)

    ## end of controller

    ## todo: write the controlled dynamics and see how it works.



