import numpy as np
import gc
import sys
sys.dont_write_bytecode = True
sys.path.append('../../MODEL_SRC')

from dkdmd import DKDMD

def main(case, noise_level, sigma_list, rank_list):

    # Case description
    case_name = case + '_noise_level_' + str(noise_level)

    ## original data
    total_number_train_data = 297

    # KDMD Options
    kdmd_rank_list = rank_list

    kernel = 'gaussian'
    # kernel = 'polynomial'

    for sigma in sigma_list:
        for rank in kdmd_rank_list:
            config = {
                'model_case'     : case_name,
                'kernel'         : kernel,
                'sigma'          : sigma,
                'reduced_rank'   : rank
            }
            case_dir = './' + config['model_case']
            data = np.load(case_dir + '/' + str(total_number_train_data) + '_trainData.npz')

            # feed with trajectory data
            X = data['Xtrain']
            tspan = data['tspan']

            print('')
            print('...KDMD....', ' sigma = ', config['sigma'], ' reduced_rank = ', config['reduced_rank'])

            # start modeling with KDMD
            kdmdLearner = DKDMD(config)
            kdmdLearner.train(X=X, dt=tspan[1]-tspan[0])
            kdmdLearner.save_model()
            gc.collect()

if __name__ == '__main__':

    case = '20d_cylinder'
    noise_level = 0

    type = 'KDMD'
    sigma_list = [3]
    rank_list = [180] # [400]

    main(case,noise_level,sigma_list, rank_list)

