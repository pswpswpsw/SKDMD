import numpy as np
import gc
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, '../../../')

from SKDMD.MODEL_SRC.cedmd import CEDMD
from SKDMD.MODEL_SRC.ckdmd import CKDMD


def main(case, noise_level, type, rank_list):
    # Case description
    case_name = case + '_noise_level_' + str(noise_level)
    phase_space_range = [[-.5, .5], [-.5, .5]]
    total_numer_data = 1600

    if type == 'EDMD':

        # EDMD Options

        polynomial_type = 'hermite'
        hermite_order = 5
        edmd_rank_list = rank_list  # [10, 11, 12, 18, 19, 20, 25, 30, 36]

        for rank in edmd_rank_list:
            config = {
                'model_case': case_name,
                'dict': polynomial_type,
                'hermite_order': hermite_order,
                'phase_space_range': phase_space_range,
                'reduced_rank': rank  # 10
            }

            case_dir = './' + config['model_case']
            data = np.load(case_dir + '/' + str(total_numer_data) + '_trainData.npz')

            X = data['Xtrain']  # np.random.rand(50,2)
            Xdot = data['XdotTrain']  # np.random.rand(50, 2)

            print('...EDMD....', ' hermite order = ', config['hermite_order'], ' reduced rank = ', config[
                'reduced_rank'])
            # mkdir(directory=case_dir + '/c-edmd-h' + str(config['hermite_order']) + '-r' + str(config[
            # 'reduced_rank']))
            edmdLearner = CEDMD(config)
            edmdLearner.train(X=X, Xdot=Xdot)
            edmdLearner.save_model()
            edmdLearner.plot_Koopman_eigen_contour()

            print('')
            # collect
            gc.collect()

    elif type == 'KDMD':

        # KDMD Options

        sigma_list = [2]
        kdmd_rank_list = rank_list  # [80, 100, 150] # [10, 25, 30]
        kernel = 'gaussian'

        for sigma in sigma_list:
            for rank in kdmd_rank_list:
                config = {
                    'model_case': case_name,
                    'kernel': kernel,
                    'sigma': sigma,
                    'phase_space_range': phase_space_range,
                    'reduced_rank': rank
                }
                case_dir = './' + config['model_case']
                data = np.load(case_dir + '/' + str(total_numer_data) + '_trainData.npz')

                X = data['Xtrain']  # np.random.rand(50,2)
                Xdot = data['XdotTrain']  # np.random.rand(50, 2)

                print('')
                print('...KDMD....', ' sigma = ', config['sigma'], ' reduced_rank = ', config['reduced_rank'])
                # mkdir(directory=case_dir + '/kdmd-s' + str(config['sigma']) + '-r' + str(config['reduced_rank']))
                kdmdLearner = CKDMD(config)
                kdmdLearner.train(X=X, Xdot=Xdot)
                kdmdLearner.save_model()
                kdmdLearner.plot_Koopman_eigen_contour()
                gc.collect()

    else:

        raise NotImplementedError()


if __name__ == '__main__':
    case = '2d_simple_lusch2017'
    noise_level = 0

    # type = 'EDMD'
    # rank_list = [36]
    # rank_list = [35, 31, 26, 21, 16]

    type = 'KDMD'
    rank_list = [36]
    # rank_list = [35, 31, 26, 21, 16, 10]

    main(case, noise_level, type, rank_list)
