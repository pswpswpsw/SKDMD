"""
Evaluate model performance in a posteriori

three steps

* mapping state IC into feature space IC
* evolving feature space IC linearly using np.linalg.expm
* use either DL nonlinear reconstruction or simply linear Koopman modes for EDMD, KDMD.

.. note::
    we do not use any recursive interation, simply because transforming back and tranforming into the phi and x space will cause some accumulation error

"""

import sys
import joblib

sys.dont_write_bytecode = True
sys.path.append('../../MODEL_SRC/')


import numpy as np
from dmd import *
from lib.lib_model_interface import *
from scipy.linalg import lstsq
from numpy.linalg import matrix_power
from scipy.linalg import expm
from lib_analytic_model import F_simple_2d_system, F_duffing_2d_system
from scipy.integrate import ode
from sklearn.linear_model import enet_path


N_CPU = joblib.cpu_count()

def F_simple_2d_system_interface(t, y):
    """
    Governing equation for 2D simple system of Lusch 2017 paper: for Python RK4 purpose

    :type t: float
    :param t: time

    :type y: np.ndarray
    :param y: state of the system
    :return: derivative of the state
    :rtype: np.ndarray
    """
    input = np.array(y)
    return F_simple_2d_system(input)


def F_2d_duffing_system_interface(t, y):
    """
    Governing equation for 2D duffing system using Otto 2017 paper: for Python RK4 purpose

    :type t: float
    :param t: time

    :type y: np.ndarray
    :param y: state of the system
    :return: derivative of the state
    :rtype: np.ndarray
    """
    input = np.array(y)
    return F_duffing_2d_system(input)


class ClassApoEval(object):
    """
    Class for a posteriori evaluation

    * put model and model_name, case_name in the constructor

    """

    def __init__(self, model, model_name, case_name,
                 normalize_phi_tilde=False,
                 max_iter=1e4,
                 alpha_range=np.logspace(-16, 1, 50),
                 selected_modes_flag=True,
                 scaling_factor=1,
                 truncation_threshold=5e-3):
        """
        initialize with model

        * model must have these methods & property
            * model.computePhi
            * model.linearEvolving
            * model.reconstruct

        :type model: classobj
        :param model: Koopman model for prediction

        :type model_name: str
        :param model_name: name of the model

        :type case_name: str
        :param case_name: name of the problem case
        """
        self.model = model
        self.model_name = model_name
        self.case_name = case_name
        self.normalize_phi_tilde = normalize_phi_tilde
        self.max_iter = max_iter
        self.alpha_range = alpha_range
        self.selected_modes_flag = selected_modes_flag
        self.top_k_index_modes_select = None
        self.scaling_factor=scaling_factor
        self.truncation_threshold = truncation_threshold # 1e-2 for 2D fixed point case, 1e-3 for other cases
        self.dt = self.model.dt

        if self.model_name[0] == 'd':
            self.type = 'd'
        elif self.model_name[0] == 'c':
            self.type = 'c'
        else:
            raise NotImplementedError("check your model name! it must start with 'd' or 'c'!")

        self.save_dir = './' + case_name + '/' + model_name + '/'
        mkdir(self.save_dir)

        # just for KoopmanLQR controller
        self.selected_index = None
        self.selected_koopman_modes = None

    def compute_eigen(self, state, selected_modes_index=None):
        ## warning: this is only reserved for LQR controller class
        eigen_phi_state = self.model.computeEigenPhi(state, selected_modes_index)
        return eigen_phi_state

    def compute_eigen_grad_with_gxT(self, gxT, x, selected_modes_index=None):
        return self.model.compute_eigen_grad_with_gxT(gxT, x, selected_modes_index)

    def get_Lambda_matrix(self, selected_modes_index):
        ## warning: this is only reserved for LQR controller class
        Lambda = self.model.get_linearEvolvingEigen(selected_modes_index)
        return Lambda

    def get_reconstructed_state(self, phi, selected_modes_index, selected_modes):
        ## warning: this is only reserved for LQR controller class
        state = self.model.reconstructFromEigenPhi(phi, selected_modes_index, selected_modes)
        return state

    def get_reconstruction_transformation(self, selected_modes_index, selected_modes):
        transformation_matrix = self.model.get_reconstruction_transformation(selected_modes_index, selected_modes)
        return transformation_matrix



    def predict(self, tspan, ic, selected_modes_flag=False, selected_modes_index=None, selected_modes=None):
        """
        future predicting for the state of dynamical system

        :type tspan: np.ndarray
        :param tspan: time array for prediction

        :type ic: np.ndarray
        :param ic: initial condition of the state

        :return: array of state in the future
        :rtype: np.ndarray
        """

        state = ic
        state_list_initial = [state]

        # direct evaluate state at time t using analytic solution of linear system, not using recursive for each delta t

        if selected_modes_flag:
            # selected_modes_index = self.top_k_index_modes_select
            print("number of final selected modes", len(selected_modes_index))
        else:
            selected_modes_index = None
            # print('top k modes index = ', self.top_k_index_modes_select + 1)
            # print('num = ', self.top_k_index_modes_select.size)

        # new implementation, only forward in eigenspace
        eigen_phi_state = self.model.computeEigenPhi(state, selected_modes_index)

        if self.type == 'c':

            print('continuous model')

            def state_pred(index_time):
                phi_state_at_t = eigen_phi_state @ expm((tspan[index_time] - tspan[0]) * self.get_Lambda_matrix(selected_modes_index))
                return self.get_reconstructed_state(phi_state_at_t, selected_modes_index, selected_modes)

            # parallel implementation
            print('num cpu = ', N_CPU)
            state_list = joblib.Parallel(n_jobs=N_CPU)(joblib.delayed(state_pred)(index_time) for index_time in range(1, tspan.size))

            state_list = state_list_initial + state_list  ## combine with initial condition

        else:

            print('discrete model')

            tmp_1 = self.get_Lambda_matrix(selected_modes_index)
            tmp_1_d = np.diag(tmp_1)

            def state_pred(index_time):
                # old version
                # phi_state_at_t = np.matmul(eigen_phi_state, matrix_power(self.get_Lambda_matrix(selected_modes_index), index_time))
                # new version
                phi_state_at_t = eigen_phi_state @ np.diag(tmp_1_d**index_time)

                return self.get_reconstructed_state(phi_state_at_t, selected_modes_index, selected_modes)

            # parallel implementation
            print('num cpu = ', N_CPU)
            state_list = joblib.Parallel(n_jobs=N_CPU)(
                joblib.delayed(state_pred)(index_time) for index_time in range(1, tspan.size))

            state_list = state_list_initial + state_list

        return np.vstack(state_list)

    def compute_save_svd_effect_on_phi_edmd(self):

        # obtain eigenvectors & phi on training data
        ev = self.model.KoopmanEigenV
        phi = self.model.Phi_X_i_sample

        # 1. Phi V_i: before SVD
        phi_ev = np.matmul(phi, ev)

        # 2. compute SVD for phi
        u, s, vh = np.linalg.svd(phi, full_matrices=False)

        phi_r_ev_list = []
        for r in range(1, np.linalg.matrix_rank(phi) + 1):
            ur = u[:, :r]
            phi_r = np.matmul(np.matmul(ur, ur.T), phi)
            phi_r_ev = np.matmul(phi_r, ev)
            phi_r_ev_list.append(phi_r_ev)

        # 3. save data into file
        np.savez(self.save_dir + 'compute_save_svd_effect_on_phi_edmd.npz', phi=phi, ev=ev, phi_r_ev_list=phi_r_ev_list,
                 phi_ev=phi_ev)

    def compute_kou_index(self, true_tj, true_eigenTj):

        # true_tj = M, N
        # true_eigenT = M,L
        # B = L, N

        # 1. compute koopman modes
        B = np.linalg.lstsq(true_eigenTj, true_tj)[0]

        # 2. normalize koopman modes
        # B_normalized = np.matmul(np.linalg.inv(np.diag(np.linalg.norm(B, axis=1))), B)

        # 3. recompute the eigenfunctions
        true_eigenTj_new = np.matmul(true_eigenTj, np.diag(np.linalg.norm(B, axis=1)))

        # 4. compute the sum of absolute value
        abs_sum_of_eigen_function = np.sum(abs(true_eigenTj_new), axis=0)

        index_chosen = np.argsort(abs_sum_of_eigen_function)[::-1]  # best modes comes first
        abs_sum_of_index_chosen = np.sort(abs_sum_of_eigen_function)[::-1]

        # 5. bouns: compute the normalized residual remaining
        relative_rec_error_from_top_i_rank_1_sum_list = []
        for i in range(len(index_chosen)):
            top_i_index = index_chosen[:i + 1]
            X_rec_from_top_i = np.matmul(true_eigenTj[:, top_i_index], B[top_i_index, :])
            relative_rec_error_from_top_i_rank_1_sum_list.append(
                np.linalg.norm(X_rec_from_top_i - true_tj) / np.linalg.norm(true_tj))

        return index_chosen, abs_sum_of_index_chosen, np.array(relative_rec_error_from_top_i_rank_1_sum_list)

    def compute_accuracy_and_aposterior_eigentj(self, true_tsnap, true_tj, num_user_defined):

        # print('266: ', datetime.now())
        true_eigenTj = self.model.computeEigenPhi(true_tj)

        ## 1. compute max normalized error

        # normalizing constant for the eigenmodes evolution error
        NC = np.sqrt(np.mean(np.abs(true_eigenTj) ** 2, axis=0))

        # define function that evaluates relative error at time t
        if self.type == 'c':

            def error_examiner_at_single_time(index_time):
                delta_time = true_tsnap[index_time] - true_tsnap[0]
                # linearly evolving eigenmodes into the future
                eigenTj_at_t = true_eigenTj[0, :] @ expm(delta_time * self.model.get_linearEvolvingEigen())
                # save varphi error
                eigenError_at_t = true_eigenTj[index_time, :] - eigenTj_at_t
                # save varphi error / linear evolving
                relative_error_at_t = np.abs(eigenError_at_t) / NC  # / (np.abs(eigenTj_at_t) + 1e-6)
                return relative_error_at_t, eigenTj_at_t

        elif self.type == 'd':

            # diagonalize A matrix first
            ev_A, et_A = np.linalg.eig(self.model.get_linearEvolvingEigen())
            assert np.linalg.norm(np.linalg.pinv(et_A) @ np.diag(ev_A) @ et_A - self.model.get_linearEvolvingEigen()) < 1e-8

            tmp_1 = true_eigenTj[0, :] @ np.linalg.pinv(et_A)

            def error_examiner_at_single_time(index_time):
                # linearly evolving eigenmodes into the future
                # old version
                # eigenTj_at_t = np.matmul(true_eigenTj[0, :], matrix_power(self.model.get_linearEvolvingEigen(), index_time))
                # new version.. speed up a lot!!
                eigenTj_at_t = tmp_1 @ np.diag(ev_A**index_time) @ et_A

                # save varphi error
                eigenError_at_t = true_eigenTj[index_time, :] - eigenTj_at_t
                # save varphi error / linear evolving
                relative_error_at_t = np.abs(eigenError_at_t) / NC  # / (np.abs(eigenTj_at_t) + 1e-6)
                return relative_error_at_t, eigenTj_at_t

        else:
            raise NotImplementedError()

        # print('301: ', datetime.now())

        # parallel coimputing the normalized relative error
        r = joblib.Parallel(n_jobs=N_CPU)(joblib.delayed(error_examiner_at_single_time)(index_time) for index_time in range(true_tsnap.size))

        # print('306: ', datetime.now())


        normalized_relative_error, aposteriori_eigen = zip(*r)
        normalized_relative_error = np.array(normalized_relative_error)
        aposteriori_eigen = np.array(aposteriori_eigen)


        ## 2. order the error from small to large
        max_normalized_relative_error = np.max(normalized_relative_error, axis=0)
        # mean_normalized_relative_error = np.mean(normalized_relative_error, axis=0)

        small_to_large_error_eigen_index = np.argsort(max_normalized_relative_error)

        ## 3. selected top few modes, get index
        selected_best_k_accurate_modes_index = small_to_large_error_eigen_index[:num_user_defined]

        ## 4. sweep for reconstruction and accuracy truncation
        norm_true_tj = np.linalg.norm(true_tj)

        ## call reconstruction function
        small_to_large_error_eigen_index_kou, abs_sum_of_index_chosen_kou, error_reconstruct_state_list, selected_best_k_aposteriori_eigen = \
            self.compute_reconstruction_with_top_few_eigenmodes(num_user_defined, aposteriori_eigen, small_to_large_error_eigen_index,
                                                           true_tj, norm_true_tj, normalized_relative_error)


        # print('349: ', datetime.now())
        return normalized_relative_error, max_normalized_relative_error, small_to_large_error_eigen_index, small_to_large_error_eigen_index_kou, \
               abs_sum_of_index_chosen_kou, error_reconstruct_state_list, selected_best_k_accurate_modes_index, selected_best_k_aposteriori_eigen, aposteriori_eigen, norm_true_tj

    def compute_reconstruction_with_top_few_eigenmodes(self, num_user_defined, aposteriori_eigen, small_to_large_error_eigen_index, true_tj, norm_true_tj,
                                                       normalized_relative_error):

        def reconstruct_top_selected_eigenmodes(i):

            # new: we use a posteriori eigen for reconstruct the modes
            top_i_selected_eigenTj = aposteriori_eigen[:, small_to_large_error_eigen_index[:i]]
            B = lstsq(top_i_selected_eigenTj, true_tj)[0]
            error_reconstruct_state = top_i_selected_eigenTj @ B - true_tj
            error_reconstruct_state = np.linalg.norm(error_reconstruct_state) / norm_true_tj

            return top_i_selected_eigenTj, error_reconstruct_state

        # print('357: ', datetime.now())

        # set the maximal evaluation best features to be 500.. note that I can still consider features more than 500..but
        # best ones are only 500... this is due to computational complexity issues....
        max_num_feature_evaluate = min(500, normalized_relative_error.shape[1] + 1)
        if normalized_relative_error.shape[1] + 1 >= 500: print('too much feature to evaluate, cut to 500!')

        r = joblib.Parallel(n_jobs=N_CPU)(joblib.delayed(reconstruct_top_selected_eigenmodes)(i) for i in range(1, max_num_feature_evaluate))
        top_i_selected_eigenTj_list, error_reconstruct_state_list = zip(*r)


        # print('361: ', datetime.now())

        selected_best_k_aposteriori_eigen = top_i_selected_eigenTj_list[:num_user_defined][-1]

        ## 5. compute kou's criterion (very fast)
        # disable that kou criterion just for efficiency issues..
        # small_to_large_error_eigen_index_kou, abs_sum_of_index_chosen_kou, _ = self.compute_kou_index(aposteriori_eigen, true_eigenTj)
        small_to_large_error_eigen_index_kou, abs_sum_of_index_chosen_kou = [], []

        return small_to_large_error_eigen_index_kou, abs_sum_of_index_chosen_kou, error_reconstruct_state_list, selected_best_k_aposteriori_eigen

    def order_modes_with_accuracy_and_aposterior_eigentj(self, true_tsnap, true_tj, num_user_defined):

        normalized_relative_error, max_normalized_relative_error, small_to_large_error_eigen_index, small_to_large_error_eigen_index_kou, \
        abs_sum_of_index_chosen_kou, error_reconstruct_state_list, selected_best_k_accurate_modes_index, selected_best_k_aposteriori_eigen, aposteriori_eigen, norm_true_tj = \
            self.compute_accuracy_and_aposterior_eigentj(true_tsnap, true_tj, num_user_defined)

        # finally, save them for future plot
        np.savez(self.save_dir + 'ComputeSaveNormalizedEigenError_fig1.npz',
                 nre=normalized_relative_error,
                 tt=true_tsnap,
                 le=self.model.linearEvolvingEigen)


        ## SAVE everything related to rec. vs acc. plot and kou's criterion
        np.savez(self.save_dir + 'ComputeSaveNormalizedEigenError_fig2.npz',
                 mre=max_normalized_relative_error,
                 stli=small_to_large_error_eigen_index,
                 stli_kou=small_to_large_error_eigen_index_kou,
                 abs_sum_kou=abs_sum_of_index_chosen_kou,
                 ersl=error_reconstruct_state_list)

        ## SAVE just for legacy purpose
        np.savez(self.save_dir + 'ComputeSaveNormalizedEigenError_fig3.npz', tkm_index_list=range(num_user_defined))

        return selected_best_k_accurate_modes_index, selected_best_k_aposteriori_eigen

    def order_modes_with_accuracy_and_aposterior_eigentj_multiple_trj(self, true_tsnap_list, true_tj_list, num_user_defined):

        for i_trj in range(len(true_tsnap_list)):
            true_tsnap = true_tsnap_list[i_trj]
            true_tj = true_tj_list[i_trj]

            normalized_relative_error, max_normalized_relative_error, small_to_large_error_eigen_index, small_to_large_error_eigen_index_kou, \
            abs_sum_of_index_chosen_kou, error_reconstruct_state_list, selected_best_k_accurate_modes_index, selected_best_k_aposteriori_eigen, aposteriori_eigen, norm_true_tj = \
                self.compute_accuracy_and_aposterior_eigentj(true_tsnap, true_tj, num_user_defined)

            ## save
            # finally, save them for future plot
            sub_folder = self.save_dir + str(i_trj) + '/'
            mkdir(sub_folder)
            np.savez(sub_folder + 'ComputeSaveNormalizedEigenError_fig1.npz',
                     nre=normalized_relative_error,
                     tt=true_tsnap,
                     le=self.model.linearEvolvingEigen)

            ## SAVE everything related to rec. vs acc. plot and kou's criterion
            np.savez(sub_folder + 'ComputeSaveNormalizedEigenError_fig2.npz',
                     mre=max_normalized_relative_error,
                     stli=small_to_large_error_eigen_index,
                     stli_kou=small_to_large_error_eigen_index_kou,
                     abs_sum_kou=abs_sum_of_index_chosen_kou,
                     ersl=error_reconstruct_state_list)

            ## SAVE just for legacy purpose
            np.savez(sub_folder + 'ComputeSaveNormalizedEigenError_fig3.npz', tkm_index_list=range(num_user_defined))

        ## read and compute the `selected_best_k_accurate_modes_index` by taking the mean from all the data

        ## debug
        sys.exit()

        return selected_best_k_accurate_modes_index

    def sweep_sparse_reconstruction_for_modes_selection(self, true_tj, top_k_selected_eigenTj, top_k_index):

        ## 1. get relative index selected from sweeping multi-task elastic net
        path_selected_relative_index_array_list = self.compute_save_multi_task_elastic_net_path(phi_tilde=top_k_selected_eigenTj,
                                                                                                X=true_tj,
                                                                                                max_iter=self.max_iter)
        selected_index_array_list = []
        for index_array in path_selected_relative_index_array_list:
            selected_index_array_list.append(top_k_index[index_array])

        ## 2. compute Koopman modes & koopman decomposed field for each case
        selected_koopman_modes_list = []
        decomposed_koopman_field_list = []
        for index_array in path_selected_relative_index_array_list:
            if len(index_array) > 0:
                B = lstsq(top_k_selected_eigenTj[:, index_array], true_tj)[0]
                field = np.einsum("ij,jk->jik",top_k_selected_eigenTj[:, index_array],B)
            else:
                B = np.array([])
                field=np.array([])
            selected_koopman_modes_list.append(B)
            decomposed_koopman_field_list.append(field)

        ## save them into each sweep directory
        for i, alpha in enumerate(self.alpha_range[::-1]):
            alpha_dir_eval = self.save_dir + 'sweep/sweep_alpha_' + str(alpha)
            np.savez(alpha_dir_eval + '/selected_index_and_koopman_modes.npz',
                     selected_index=selected_index_array_list[i],
                     selected_eigenvals=np.diag(self.model.linearEvolvingEigen)[selected_index_array_list[i]],
                     selected_koopman_modes=selected_koopman_modes_list[i],
                     decomposed_koopman_field=decomposed_koopman_field_list[i]*self.scaling_factor)

        return

    def sweep_sparse_prediction_comparison(self, true_tsnap, true_trajectory):

        # 1. read alpha_enet from `fig_data`
        fig_data = np.load(self.save_dir + 'MultiTaskElasticNet_result.npz')
        alphas_enet = fig_data['alphas_enet']

        # 2. read koopman modes from each sweep directory
        for ii, alpha in enumerate(alphas_enet):

            print("=================================================")
            print("now sweep alpha = ", alpha, " index = ", ii)

            alpha_dir_eval = self.save_dir + 'sweep/sweep_alpha_' + str(alpha)
            data = np.load(alpha_dir_eval + '/selected_index_and_koopman_modes.npz')
            selected_index = data['selected_index']
            selected_koopman_modes = data['selected_koopman_modes']

            if len(selected_index) > 0:

                # predict new trajectory based on true trajectory
                pred_trajectory = self.model_pred_save_trajectory_given_true_trajectory(true_tsnap,
                                                                                        true_trajectory,
                                                                                        self.selected_modes_flag,
                                                                                        selected_index,
                                                                                        selected_koopman_modes,
                                                                                        alpha_dir_eval)
    def load_best_index_data(self, best_index, save_dir):
        ## Warning: this function is only reserved for Koopman LQR class

        # 1. read alpha_enet from `fig_data`
        fig_data = np.load(save_dir + 'MultiTaskElasticNet_result.npz')
        alphas_enet = fig_data['alphas_enet']

        # 2. read the selected modes from disk: the sweep folder
        alpha = alphas_enet[best_index]
        alpha_dir_eval = save_dir + 'sweep/sweep_alpha_' + str(alpha)
        data = np.load(alpha_dir_eval + '/selected_index_and_koopman_modes.npz')
        selected_index = data['selected_index']
        selected_koopman_modes = data['selected_koopman_modes']

        ## save them as public shared variables
        self.selected_index = selected_index
        self.selected_koopman_modes = selected_koopman_modes

    def sparse_compute_eigen_at_best_index(self, state):
        assert type(self.selected_index) != type(None)
        assert type(self.selected_koopman_modes) != type(None)
        # then make prediction, return it
        eigen = self.compute_eigen(state, selected_modes_index=self.selected_index)
        return eigen

    def sparse_compute_actuator_aux_state_dependent_matrix_at_best_index(self, gxT, x):
        assert type(self.selected_index) != type(None)
        assert type(self.selected_koopman_modes) != type(None)
        # then get the transformation matrix!
        BuT = self.compute_eigen_grad_with_gxT(gxT, x, self.selected_index)
        return BuT

    def sparse_compute_Lambda_at_best_index(self):
        assert type(self.selected_index) != type(None)
        assert type(self.selected_koopman_modes) != type(None)
        # then make prediction, return it
        Lambda = self.get_Lambda_matrix(self.selected_index)
        return Lambda

    def sparse_compute_reconstruct_at_best_index(self, phi):
        assert type(self.selected_index) != type(None)
        assert type(self.selected_koopman_modes) != type(None)
        # then make prediction, return it
        state = self.get_reconstructed_state(phi, self.selected_index, self.selected_koopman_modes)
        return state

    def sparse_compute_transformation_matrix_at_best_index(self):
        assert type(self.selected_index) != type(None)
        assert type(self.selected_koopman_modes) != type(None)

        # then get the transformation matrix!
        transformation_matrix = self.get_reconstruction_transformation(self.selected_index, self.selected_koopman_modes)

        return transformation_matrix

    def ComputeSaveNormalizedEigenError(self, true_tsnap, true_tj, true_eigenTj, relative_error=True, top_k_modes=10):
        """only called by EDMD/KDMD"""

        ## fig 1. eigenmodes evolution error vs time
        # note that we will compute a time series of normalized eigenerror, given true eigenTj
        # pred_eigenTj = []
        # error_eigen = []
        # normalized_relative_error = []

        # normalizing constant for the eigenmodes evolution error
        NC = np.sqrt(np.mean(np.abs(true_eigenTj) ** 2, axis=0))

        if self.type == 'c':

            def error_examiner_at_single_time(index_time):
                delta_time = true_tsnap[index_time] - true_tsnap[0]
                # linearly evolving eigenmodes into the future
                eigenTj_at_t = true_eigenTj[0, :] @ expm(delta_time * self.model.get_linearEvolvingEigen())
                # save varphi error
                eigenError_at_t = true_eigenTj[index_time, :] - eigenTj_at_t
                # save varphi error / linear evolving
                if relative_error:
                    relative_error_at_t = np.abs(eigenError_at_t) / NC  # / (np.abs(eigenTj_at_t) + 1e-6)
                else:
                    relative_error_at_t = np.abs(eigenError_at_t)
                return relative_error_at_t, eigenTj_at_t

        elif self.type == 'd':



            def error_examiner_at_single_time(index_time):
                # linearly evolving eigenmodes into the future
                eigenTj_at_t = np.matmul(true_eigenTj[0, :],
                                         matrix_power(self.model.get_linearEvolvingEigen(), index_time))
                # save varphi error
                eigenError_at_t = true_eigenTj[index_time, :] - eigenTj_at_t
                # save varphi error / linear evolving
                if relative_error:
                    relative_error_at_t = np.abs(eigenError_at_t) / NC  # / (np.abs(eigenTj_at_t) + 1e-6)
                else:
                    relative_error_at_t = np.abs(eigenError_at_t)
                return relative_error_at_t, eigenTj_at_t

        else:

            raise NotImplementedError()

        ## parallel coimputing the normalized relative error
        r = joblib.Parallel(n_jobs=N_CPU)(joblib.delayed(error_examiner_at_single_time)(index_time) for index_time in range(true_tsnap.size))

        normalized_relative_error, aposteriori_eigen = zip(*r)
        normalized_relative_error = np.array(normalized_relative_error)
        aposteriori_eigen = np.array(aposteriori_eigen)

        ## SAVE
        # 1. normalized_relative_error
        # 2. true_tsnap
        # 3. self.model.linearEvolvingEigen
        np.savez(self.save_dir + 'ComputeSaveNormalizedEigenError_fig1.npz',
                 nre=normalized_relative_error,
                 tt=true_tsnap,
                 le=self.model.linearEvolvingEigen)

        # plot error ordered plot in max error
        # also plot reconstruction error from SMALL to LARGE
        max_normalized_relative_error = np.max(normalized_relative_error, axis=0)
        small_to_large_error_eigen_index = np.argsort(max_normalized_relative_error)

        # implement kou's criterion - before error analysis
        # 1. -- using instanenous observaiton
        # small_to_large_error_eigen_index_kou, abs_sum_of_index_chosen_kou = self.compute_kou_index(true_tj,
        #                                                                                            true_eigenTj)
        # 2. -- using evoluted observation
        small_to_large_error_eigen_index_kou, \
        abs_sum_of_index_chosen_kou, \
        relative_rec_error_from_top_i_rank_1_sum = self.compute_kou_index(aposteriori_eigen, true_eigenTj)

        print('index chosen by Kou  = ', small_to_large_error_eigen_index_kou + 1)
        print('abs energy of Kou = ', abs_sum_of_index_chosen_kou)

        # record the top_k_selected_eigenTj, true_tj
        # top_i_selected_eigenTj_list = []

        # reconstruction error with selected eigenmodes: using pseudo-inverse
        # error_reconstruct_state_list = []

        def reconstruct_top_selected_eigenmodes(i):

            ## new: we use a posteriori eigen for reconstruct the modes
            top_i_selected_eigenTj = aposteriori_eigen[:, small_to_large_error_eigen_index[:i]]

            ## previous
            # top_i_selected_eigenTj = true_eigenTj[:, small_to_large_error_eigen_index[:i]]

            # normalize the eigenfunction so it is easier for ElasticNet to find good sparse solutions.

            # top_k_selected_eigenTj = top_k_selected_eigenTj / np.linalg.norm(top_k_selected_eigenTj, axis=0)

            # use lstsq to solve B: min || Phi * B - X ||
            # note that, the B here is complex, due to Phi is eigendecomposition.

            B = lstsq(top_i_selected_eigenTj, true_tj)[0]
            error_reconstruct_state = np.matmul(top_i_selected_eigenTj, B) - true_tj
            error_reconstruct_state = np.linalg.norm(error_reconstruct_state) / np.linalg.norm(true_tj)

            return top_i_selected_eigenTj, error_reconstruct_state

        r = joblib.Parallel(n_jobs=N_CPU)(joblib.delayed(reconstruct_top_selected_eigenmodes)(i) for i in
                                          range(1, normalized_relative_error.shape[1] + 1))

        top_i_selected_eigenTj_list, error_reconstruct_state_list = zip(*r)

        # for i in range(1, normalized_relative_error.shape[1] + 1): # loop over top-i selected eigen modes
        #
        #     top_k_selected_eigenTj = true_eigenTj[:, small_to_large_error_eigen_index[:i]]
        #
        #     # normalize the eigenfunction so it is easier for ElasticNet to find good sparse solutions.
        #     top_k_selected_eigenTj = top_k_selected_eigenTj/np.linalg.norm(top_k_selected_eigenTj, axis=0)
        #
        #     top_i_selected_eigenTj_list.append(top_k_selected_eigenTj)
        #
        #     # use lstsq to solve B: min || Phi * B - X ||
        #     # note that, the B here is complex, due to Phi is eigendecomposition.
        #     B = lstsq(top_k_selected_eigenTj, true_tj)[0]
        #
        #     error_reconstruct_state = np.matmul(top_k_selected_eigenTj, B) - true_tj
        #     error_reconstruct_state = np.linalg.norm(error_reconstruct_state)/np.linalg.norm(true_tj)
        #     error_reconstruct_state_list.append(error_reconstruct_state)

        ## post-processing! sparse regression using elastic-net

        # we augment phi_tilde into fully real formulism.
        top_k_selected_eigenTj = top_i_selected_eigenTj_list[:top_k_modes][-1]
        ## this eigenTj is model trained with training data
        # but evaluated at validation data, and it performs linearly well.

        bool_index_kept, kept_koopman_modes = self.compute_save_multi_task_elastic_net_path(
            phi_tilde=top_k_selected_eigenTj,
            X=true_tj,
            max_iter=self.max_iter)

        # finally update the top modes selected
        self.top_k_index_modes_select = small_to_large_error_eigen_index[:top_k_modes]
        # print ("small to large error index = ", self.top_k_index_modes_select)
        self.top_k_index_modes_select = self.top_k_index_modes_select[bool_index_kept]

        assert len(self.top_k_index_modes_select) == kept_koopman_modes.shape[0]

        self.kept_koopman_modes = kept_koopman_modes

        ## SAVE
        # 1. max_normalized_relative_error
        # 2. small_to_large_error_eigen_index
        # 3. error_reconstruct_state_list
        np.savez(self.save_dir + 'ComputeSaveNormalizedEigenError_fig2.npz',
                 mre=max_normalized_relative_error,
                 stli=small_to_large_error_eigen_index,
                 stli_kou=small_to_large_error_eigen_index_kou,
                 abs_sum_kou=abs_sum_of_index_chosen_kou,
                 ersl=error_reconstruct_state_list,
                 iestli=bool_index_kept)

        ## SAVE
        # 1. top_k_modes
        np.savez(self.save_dir + 'ComputeSaveNormalizedEigenError_fig3.npz', tkm_index_list=range(top_k_modes))

        ## SAVE
        # 1. for sparse regression later in post-processing
        # -- save top K modes first, then do sparse regression to find the best parameter
        # top_i_selected_eigenTj_dict = {'phi_tilde': top_k_selected_eigenTj,
        #                                'X': true_tj,
        #                                'small_to_large_error_eigen_index': small_to_large_error_eigen_index[:top_k_modes]}
        # with open(self.save_dir + 'top_i_selected_list_and_true_tj.pkl', 'wb') as f:
        #     pickle.dump(top_i_selected_eigenTj_dict, f, pickle.HIGHEST_PROTOCOL)

        assert true_eigenTj.shape == aposteriori_eigen.shape

        ## SAVE:
        # we save the predicted eigen trajectory, in both a priori (like DMD) and a posteriori fashion.
        # np.savez(self.save_dir + 'KoopmanDecompOnTestData.npz',
        #          apr_eigen=true_eigenTj[:, self.top_k_index_modes_select],
        #          apo_eigen=aposteriori_eigen[:, self.top_k_index_modes_select], koopman_modes=kept_koopman_modes)

        return true_eigenTj[:, self.top_k_index_modes_select], \
               aposteriori_eigen[:, self.top_k_index_modes_select], kept_koopman_modes
        ## this is a prior decomposition of validation trajectory data.

    def compute_save_multi_task_elastic_net_path(self,phi_tilde,X,max_iter=1e5,tol=1e-12,l1_ratio=0.99):
        """ computing multi task elastic net and save the coefficient of the path """

        num_alpha = len(self.alpha_range)

        ## 1. normalize the features by making modal amplititute to 1 for all features
        if self.normalize_phi_tilde:

            print("EDMD/KDMD dictionary features are normalized....")
            scaling_transform = np.diag(1. / np.abs(phi_tilde[0, :]))
            inverse_scaling = np.linalg.inv(scaling_transform)
            assert np.linalg.norm(scaling_transform @ inverse_scaling - np.eye(scaling_transform.shape[0])) < 1e-6
            phi_tilde_scaled = phi_tilde @ scaling_transform
            print('norm = ', np.linalg.norm(phi_tilde_scaled, axis=0))
            print(phi_tilde_scaled.shape)
            print(phi_tilde_scaled[0,:])

        else:

            phi_tilde_scaled = phi_tilde

        ## 2. augmenting the complex AX=B problem into a AX=B problem with real entries
        ##    since current package only support real number array

        # -- note: must run in sequential....to have good convergence property as shown in my github page.

        a = np.hstack([np.real(phi_tilde_scaled), -np.imag(phi_tilde_scaled)])
        b = np.hstack([np.imag(phi_tilde_scaled), np.real(phi_tilde_scaled)])
        phi_tilde_aug = np.vstack([a, b])
        X_aug = np.vstack([X, np.zeros(X.shape)])
        num_data = X.shape[0]
        num_target = X.shape[1]
        alphas_enet, coefs_enet_aug, _ = enet_path(phi_tilde_aug,
                                                   X_aug,
                                                   max_iter=max_iter,
                                                   tol=tol,
                                                   alphas=self.alpha_range,
                                                   l1_ratio=l1_ratio,
                                                   fit_intercept=False,
                                                   check_input=True,
                                                   verbose=0)
        num_total_eigen_func = int(coefs_enet_aug.shape[1] / 2)

        # get the real and image part from the complex solution
        coefs_enet_real = coefs_enet_aug[:, :num_total_eigen_func, :]
        coefs_enet_imag = coefs_enet_aug[:, num_total_eigen_func:, :]
        assert coefs_enet_imag.shape == coefs_enet_real.shape

        # combine them into complex arrary for final results!
        coefs_enet_comp = coefs_enet_real + 1j * coefs_enet_imag

        ## 2.5 remove feature that is smaller than 1e-3 of the max. because most often,
        for i_alpha in range(coefs_enet_comp.shape[2]):
            for i_target in range(coefs_enet_comp.shape[0]):
                coef_cutoff_value = self.truncation_threshold * np.max(abs(coefs_enet_comp[i_target, :, i_alpha]))
                index_remove = abs(coefs_enet_comp[i_target, :, i_alpha]) < coef_cutoff_value
                coefs_enet_comp[i_target, index_remove, i_alpha] = 0 +0j

        ## 2.7 given features selected, do LS-refit to remove the bias of any kind of regularization
        for i_alpha in range(coefs_enet_comp.shape[2]):
            bool_non_zero = np.linalg.norm(coefs_enet_comp[:,:,i_alpha],axis=0) > 0
            phi_tilde_scaled_reduced = phi_tilde_scaled[:, bool_non_zero]
            coef_enet_comp_reduced_i_alpha = np.linalg.lstsq(phi_tilde_scaled_reduced, X)[0]
            coefs_enet_comp[:,bool_non_zero,i_alpha] = coef_enet_comp_reduced_i_alpha.T
            coefs_enet_comp[:,np.invert(bool_non_zero),i_alpha] = 0

        ## 3. compute residual for parameter sweep. so I can draw the trade off plot between num. non-zero vs rec. resdiual

        # convert complex array into mag.
        coefs_enet_mag = np.abs(coefs_enet_comp)

        def compute_residual_list(i):
            residual = np.linalg.norm(X - np.matmul(phi_tilde_scaled, coefs_enet_comp[:, :, i].T)[:num_data])  # computed
            # the augmented but only compare first half rows.
            residual = residual / np.linalg.norm(X)  # normalize the residual in the same fashion... L2 norm of X
            return residual

        residual_array = np.array(joblib.Parallel(n_jobs=N_CPU)(joblib.delayed(compute_residual_list)(i) for i in range(num_alpha)))

        # finally, compute the complex koopman modes on those kept eigenfunctions
        if self.normalize_phi_tilde:
            phi_tilde_scaled = phi_tilde_scaled @ inverse_scaling

        # save data for trade-off plot
        np.savez(self.save_dir + 'MultiTaskElasticNet_result.npz',
                 alphas_enet=alphas_enet,
                 coefs_enet=coefs_enet_mag,
                 coefs_enet_comp=coefs_enet_comp,
                 residual_array=residual_array
                 )

        ## 4. sweep for each alpha

        # 1. for each alpha, we get the non-zero index of selected eigentj
        print("evaluation module... start alpha sweeping...")
        mkdir(self.save_dir + 'sweep')

        sweep_index_list = []
        for ii, alpha in enumerate(alphas_enet):

            print("current alpha = ", alpha, " index = ",ii)
            alpha_dir_eval = self.save_dir + 'sweep/sweep_alpha_' + str(alpha)
            mkdir(alpha_dir_eval)

            # compute selected index
            non_zero_index_bool_array = np.linalg.norm(coefs_enet_comp[:, :, ii], axis=0) > 0
            sweep_index_list.append(non_zero_index_bool_array)

        return sweep_index_list

    def compute_eigen_trj(self, trj):
        # obtain the eigenfunctions values along the trajectory
        return self.model.computeEigenPhi(trj[:, :])

    def model_pred_save_trajectory_given_true_trajectory(self, true_tsnap,
                                                         true_trajectory,
                                                         selected_modes_flag,
                                                         selected_modes_index,
                                                         selected_modes,
                                                         save_path):

        # compute prediction
        pred_trajectory = self.predict(tspan=true_tsnap,
                                       ic=true_trajectory[0:1, :],
                                       selected_modes_flag=selected_modes_flag,
                                       selected_modes_index=selected_modes_index,
                                       selected_modes=selected_modes)

        ## SAVE
        # 1. true_trajectory.shape[1]
        np.savez(save_path + '/save_trj_comparison.npz',
                 num_components=true_trajectory.shape[1],
                 tt=true_tsnap,
                 ttrj=true_trajectory*self.scaling_factor,
                 ptrj=pred_trajectory*self.scaling_factor)

        return pred_trajectory

    def computeTrueTrajectory(self, F, ic, tsnap):
        """
        compute true trajectory using scipy.intergrate.ode, with 'dopri5' algorithm

        :type F: function
        :param F: governing equation of a ODE system

        :type ic: np.ndarray
        :param ic: initial condition of the state

        :type tsnap: np.ndarray
        :param tsnap: time array

        :return: array of true trajectory
        :rtype: np.ndarray
        """

        r = ode(F).set_integrator('dopri5')
        ic_list = ic.tolist()[0]
        r.set_initial_value(ic_list, tsnap[0])

        list_trueTraj = []
        list_trueTraj.append(r.y)
        for index_time in range(tsnap.size - 1):
            # print 'index =', index_time, tsnap[index_time + 1] - tsnap[index_time]
            y_current = r.integrate(r.t + tsnap[index_time + 1] - tsnap[index_time])
            list_trueTraj.append(y_current)

        trajectory = np.vstack(list_trueTraj)

        return trajectory

    def save_trajectory(self, trajectory, filename):
        np.savez(self.save_dir + filename + '.npz', trj=trajectory)

    def save_model(self):
        # saving the a posteriori evaluation model
        with open(self.save_dir + "/apoeval.model", "wb") as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)