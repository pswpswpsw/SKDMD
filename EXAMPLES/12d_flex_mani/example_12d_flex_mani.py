import numpy as np
import gc
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, '../../../')

from SKDMD.MODEL_SRC.dedmd import DEDMD


def main(case, noise_level, num_feature_list, sigma_list):
    # Case description
    case_name = case + '_noise_level_' + str(noise_level)
    phase_space_range = [[-.5, .5], [-.5, .5]]
    total_numer_data = 501

    # EDMD Options

    for nf in num_feature_list:
        for sigma in sigma_list:

            config = {
                'model_case': case_name,
                'dict': 'rff_gaussian',
                'n_rff_features': nf,
                'rff_gaussian_sigma': sigma,
                'phase_space_range': phase_space_range,
                'reduced_rank': nf,  # since nf < num_total_data, this simples means no SVD truncation
            }

            case_dir = './' + config['model_case']
            data = np.load(case_dir + '/' + str(total_numer_data) + '_trainData.npz')

            X = data['Xtrain'][:-1,:]  # np.random.rand(50,2)
            Xdot = data['Xtrain'][1:,:]  # np.random.rand(50, 2)
            dt = data['dt']

            print('...EDMD....', ' RFF sigma = ', config['rff_gaussian_sigma'],
                  ' number of features = ', config['n_rff_features'])
            # mkdir(directory=case_dir + '/c-edmd-h' + str(config['hermite_order']) + '-r' + str(config[
            # 'reduced_rank']))
            edmdLearner = DEDMD(config)
            edmdLearner.train(X=X, Xdot=Xdot, dt=dt)
            edmdLearner.save_model()
            edmdLearner.plot_Koopman_eigen_contour()

            print('')
            # collect
            gc.collect()



if __name__ == '__main__':
    case = '12d_flex_mani'
    noise_level = 0

    type = 'EDMD'
    num_feature_list = [490]
    sigma_list = [110]

    main(case, noise_level, num_feature_list, sigma_list)
