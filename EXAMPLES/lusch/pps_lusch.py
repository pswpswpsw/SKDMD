import numpy as np
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, '../../../')

from SKDMD.PPS_SRC.postprocess import ClassEDMDPPS
from SKDMD.PPS_SRC.postprocess import ClassKDMDPPS


selected_modes_flag = True

def main(case, noise_level, model_name_list, type, plot_data=False):

    case_name = case + '_noise_level_' + str(noise_level)

    for model_name in model_name_list:

        pps_dir = './' + case_name + '/' + model_name + '/fig'
        eval_dir = './' + case_name + '/' +  model_name
        model_dir = eval_dir

        if type == 'EDMD':
            pps_edmd = ClassEDMDPPS(pps_dir=pps_dir,
                                    eval_dir=eval_dir,
                                    model_dir=model_dir,
                                    params={'relative_error': True}
                                    )


            if selected_modes_flag:

                # plot eigenmodes evaluation
                pps_edmd.pps_eigenmodes_eval()

                # sweep alpha: eigenvalue, eigenfunction and component trj
                pps_edmd.pps_sweep_alpha()
            else:
                pps_edmd.pps_component_trj()

            # plot singular value
            pps_edmd.pps_singular_value()

            # plot data distribution
            if plot_data:
                data_path = './' + case_name + '/1600_trainData.npz'
                pps_edmd.pps_2d_data_dist(data_path=data_path)

            ## only for 2d simple lusch model, one can analyze the effect of SVD on phi, given ground true phi is known..
            # pps_edmd.pps_2d_simple_lusch_effect_svd_on_phi()

            # pps_edmd.pps_lsq_spectrum()

        elif type == 'KDMD':

            #################################################3
            # 2. postprocessing for KDMD model + apo eval
            pps_kdmd = ClassKDMDPPS(pps_dir=pps_dir,
                                    eval_dir=eval_dir,
                                    model_dir=model_dir,
                                    params={'relative_error': True}
                                    )

            if selected_modes_flag:

                # plot eigenmodes evaluation
                pps_kdmd.pps_eigenmodes_eval()

                # sweep alpha: eigenvalue, eigenfunction and component trj
                pps_kdmd.pps_sweep_alpha()
            else:
                pps_kdmd.pps_component_trj()

            # plot singular value
            pps_kdmd.pps_singular_value()

            if plot_data:
                data_path = './' + case_name + '/1600_trainData.npz'
                pps_kdmd.pps_2d_data_dist(data_path=data_path)

        else:

            raise NotImplementedError()


if __name__ == '__main__':

    case = '2d_simple_lusch2017'
    noise_level = 0

    # # case 1. EDMD
    # type = 'EDMD'
    # model_list = ['c-edmd-h5-r36']

    # # case 2. EDMD with SV truncation
    # type = 'EDMD'
    # model_list = ['c-edmd-h5-r35','c-edmd-h5-r31', 'c-edmd-h5-r26','c-edmd-h5-r21','c-edmd-h5-r16']

    # case 3. KDMD
    type = 'KDMD'
    scale_list  = [2]
    # rank_list = [35,31,26,21,16]
    rank_list = [36]
    model_list = ['c-kdmd-s' + str(scale) + '-r' + str(rank) for scale in scale_list for rank in rank_list]

    main(case, noise_level, model_list, type, plot_data=False)
