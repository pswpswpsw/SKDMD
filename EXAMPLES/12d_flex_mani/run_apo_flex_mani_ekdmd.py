import numpy as np
import sys

sys.dont_write_bytecode = True
sys.path.append('../../EVAL_SRC')

from main_apo import ClassModelEDMD
from main_apo import ClassApoEval


# Options

selected_modes_flag = True
truncation_threshold = 1e-3

def main(case, noise_level, model_list, num_cut_of_plt, normalize_phi_tilde, max_iter, alpha_range):

    # setup case name
    case_name = case + '_noise_level_' + str(noise_level)

    ## prepare data
    # validation trajectory for mode selection
    valid_data = np.load('./12d_flex_mani_noise_level_0/501_validData.npz')
    valid_true_trajectory = valid_data['Xtrain']
    valid_tspan = valid_data['tspan']

    # test trajectory for true prediction
    test_data = np.load('./12d_flex_mani_noise_level_0/501_testData.npz')
    test_true_trajectory = test_data['Xtrain']
    test_tspan = test_data['tspan']

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

        if selected_modes_flag:

            # get best k Koopman modes
            best_k_accurate_mode_index, \
            best_k_aposteriori_eigen = eval_model.order_modes_with_accuracy_and_aposterior_eigentj(true_tsnap=valid_tspan,
                                                                                                   true_tj=valid_true_trajectory,
                                                                                                   num_user_defined=num_cut_of_plt)
            # multi-task elastic net sweep for sparse reconstruction
            eval_model.sweep_sparse_reconstruction_for_modes_selection(true_tj=valid_true_trajectory,
                                                                       top_k_selected_eigenTj=best_k_aposteriori_eigen,
                                                                       top_k_index=best_k_accurate_mode_index)

            # sweep prediction for different modes.
            eval_model.sweep_sparse_prediction_comparison(true_tsnap=test_tspan,
                                                          true_trajectory=test_true_trajectory)
        else:
            eval_model.model_pred_save_trajectory_given_true_trajectory(test_tspan, test_true_trajectory,
                                                                        selected_modes_flag, None, None, eval_model.save_dir)

        # save the a posteriori model
        eval_model.save_model()


if __name__ == '__main__':

    case = '12d_flex_mani'
    noise_level = 0

    ## EDMD case
    type = 'EDMD'
    model_list = ['d-edmd-rff-490-gaussian_sigma-110-rank-490'] # 100 number of features.

    normalize_phi_tilde = True  # for KDMD normalizing is performing good, but not for EDMD...

    num_cut_of_plt = 140 ## 10: just for testing
    alpha_range = np.logspace(-16, 0, 50) # new
     # alpha_range = np.logspace(-6, 2, 50) # new
    main(case, noise_level, model_list, num_cut_of_plt, normalize_phi_tilde,
         max_iter=1e2, alpha_range=alpha_range)
