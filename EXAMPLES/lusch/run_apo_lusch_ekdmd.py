import numpy as np
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, '../../EVAL_SRC')

# from SKDMD.EVAL_SRC.main_apo import ClassModelEDMD
# from SKDMD.EVAL_SRC.main_apo import ClassModelKDMD
# from SKDMD.EVAL_SRC.main_apo import ClassApoEval
# from SKDMD.EVAL_SRC.main_apo import F_simple_2d_system_interface

from main_apo import ClassModelEDMD
from main_apo import ClassModelKDMD
from main_apo import ClassApoEval
from main_apo import F_simple_2d_system_interface

# Options

selected_modes_flag = True

truncation_threshold = 1e-2


def main(case, noise_level, type, model_list, num_cut_of_plt, normalize_phi_tilde, max_iter, alpha_range):
    # setup case name
    case_name = case + '_noise_level_' + str(noise_level)

    # validation data: set an initial condition
    state_valid = np.array([[0.4, 0.4]])  # -- range from x1 = [-0.5, 0.5], x2 = [-0.5, 0.5]
    # set length of the trajectory
    tspan_valid = np.linspace(0, 30, 800)

    # testing data
    state_test = np.array([[-0.3, -0.3]])
    tspan_test = np.linspace(0, 40, 600)

    if type == 'EDMD':

        #############################
        ## EDMD
        #############################

        for model_name in model_list:
            print('=========================')
            print('')
            print('current model = ', model_name)
            print('')

            model_path_edmd = './' + case_name + '/' + model_name + '/edmd.model'
            model_edmd = ClassModelEDMD(model_path=model_path_edmd)
            eval_model = ClassApoEval(model_edmd, model_name, case_name,
                                      normalize_phi_tilde=normalize_phi_tilde,
                                      max_iter=max_iter,
                                      alpha_range=alpha_range,
                                      selected_modes_flag=selected_modes_flag,
                                      truncation_threshold=truncation_threshold)

            # generate validation data
            valid_true_trajectory = eval_model.computeTrueTrajectory(F=F_simple_2d_system_interface,
                                                                     ic=state_valid,
                                                                     tsnap=tspan_valid)
            # generate testing data
            test_true_trajectory = eval_model.computeTrueTrajectory(F=F_simple_2d_system_interface,
                                                                    ic=state_test,
                                                                    tsnap=tspan_test)

            # save trajectory
            eval_model.save_trajectory(valid_true_trajectory, "valid_trajectory")
            eval_model.save_trajectory(test_true_trajectory, "test_trajectory")

            if selected_modes_flag:

                # get best k Koopman modes
                best_k_accurate_mode_index, \
                best_k_aposteriori_eigen = eval_model.order_modes_with_accuracy_and_aposterior_eigentj(true_tsnap=tspan_valid,
                                                                                                       true_tj=valid_true_trajectory,
                                                                                                       num_user_defined=num_cut_of_plt)
                # multi-task elastic net sweep for sparse reconstruction
                eval_model.sweep_sparse_reconstruction_for_modes_selection(true_tj=valid_true_trajectory,
                                                                           top_k_selected_eigenTj=best_k_aposteriori_eigen,
                                                                           top_k_index=best_k_accurate_mode_index)

                # sweep prediction for different modes.
                eval_model.sweep_sparse_prediction_comparison(true_tsnap=tspan_test,
                                                              true_trajectory=test_true_trajectory)
            else:

                eval_model.model_pred_save_trajectory_given_true_trajectory(tspan_test, test_true_trajectory,
                                                                            selected_modes_flag, None, None, eval_model.save_dir)

    elif type == 'KDMD':

        #############################
        ## KDMD
        #############################

        for model_name in model_list:
            print('=========================')
            print('')
            print('current model = ', model_name)
            print('')

            model_path_kdmd = './' + case_name + '/' + model_name + '/kdmd.model'
            model_kdmd = ClassModelKDMD(model_path=model_path_kdmd)
            eval_model = ClassApoEval(model_kdmd, model_name, case_name, normalize_phi_tilde,
                                      max_iter=max_iter,
                                      alpha_range=alpha_range)

            # generate validation data
            valid_true_trajectory = eval_model.computeTrueTrajectory(F=F_simple_2d_system_interface,
                                                                     ic=state_valid,
                                                                     tsnap=tspan_valid)
            # generate testing data
            test_true_trajectory = eval_model.computeTrueTrajectory(F=F_simple_2d_system_interface,
                                                                    ic=state_test,
                                                                    tsnap=tspan_test)

            # save trajectory
            eval_model.save_trajectory(valid_true_trajectory, "valid_trajectory")
            eval_model.save_trajectory(test_true_trajectory, "test_trajectory")

            if selected_modes_flag:

                # get best k Koopman modes
                best_k_accurate_mode_index, best_k_aposteriori_eigen = eval_model.order_modes_with_accuracy_and_aposterior_eigentj(true_tsnap=tspan_valid,
                                                                                                                                   true_tj=valid_true_trajectory,
                                                                                                                                   num_user_defined=num_cut_of_plt)
                # multi-task elastic net sweep for sparse reconstruction
                eval_model.sweep_sparse_reconstruction_for_modes_selection(true_tj=valid_true_trajectory,
                                                                           top_k_selected_eigenTj=best_k_aposteriori_eigen,
                                                                           top_k_index=best_k_accurate_mode_index)

                # sweep prediction for different modes.
                eval_model.sweep_sparse_prediction_comparison(true_tsnap=tspan_test,
                                                              true_trajectory=test_true_trajectory)

            else:

                eval_model.model_pred_save_trajectory_given_true_trajectory(tspan_test, test_true_trajectory,
                                                                            selected_modes_flag, None, None, eval_model.save_dir)

    else:

        raise NotImplementedError()

    # save the a posteriori model
    eval_model.save_model()


if __name__ == '__main__':
    case = '2d_simple_lusch2017'
    noise_level = 0

    ## EDMD case
    # type = 'EDMD'
    # model_list = ['c-edmd-h5-r36']
    # normalize_phi_tilde = True  # for KDMD normalizing is performing good, but not for EDMD...

    ## to show SV truncation doesn't help
    # type = 'EDMD'
    # model_list = ['c-edmd-h5-r35','c-edmd-h5-r31', 'c-edmd-h5-r26','c-edmd-h5-r21','c-edmd-h5-r16']

    ## KDMD case
    type = 'KDMD'
    normalize_phi_tilde = True
    model_list = ['c-kdmd-s2-r36']

    # model_list = ['c-kdmd-s2-r35','c-kdmd-s2-r31', 'c-kdmd-s2-r26','c-kdmd-s2-r21','c-kdmd-s2-r16']

    num_cut_of_plt = 10
    alpha_range = np.logspace(-16, -1, 50)
    main(case, noise_level, type, model_list, num_cut_of_plt, normalize_phi_tilde, max_iter=1e5, alpha_range=alpha_range)
