import numpy as np
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, '../../../')

from SKDMD.PPS_SRC.postprocess import ClassEDMDPPS

selected_modes_flag = True

def main(case, noise_level, model_name_list, plot_data=False):

    case_name = case + '_noise_level_' + str(noise_level)

    for model_name in model_name_list:

        pps_dir = './' + case_name + '/' + model_name + '/fig'
        eval_dir = './' + case_name + '/' +  model_name
        model_dir = eval_dir

        pps_edmd = ClassEDMDPPS(pps_dir=pps_dir,
                                eval_dir=eval_dir,
                                model_dir=model_dir,
                                params={'relative_error': True},
                                draw_eigen_function_plot=False
                                )


        if selected_modes_flag:

            # plot eigenmodes evaluation
            pps_edmd.pps_eigenmodes_eval(y_scale_linear_error=[1e-4,1e-0],
                                         y_scale_recon_error=[0.01, 0.5])

            # sweep alpha: eigenvalue, eigenfunction and component trj
            pps_edmd.pps_sweep_alpha()
        else:
            pps_edmd.pps_component_trj()


if __name__ == '__main__':

    case = '12d_flex_mani'
    noise_level = 0

    # case 1. EDMD
    type = 'EDMD'
    model_list = ['d-edmd-rff-490-gaussian_sigma-110-rank-490']

    main(case, noise_level, model_list, plot_data=False)
