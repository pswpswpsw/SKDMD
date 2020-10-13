import numpy as np
from scipy.io import loadmat
import sys
sys.dont_write_bytecode = True
import errno
from matplotlib import pyplot as plt
import os

plt.style.use('siads')


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def main():

    # start with some Re
    n_com = 20
    data_mat = loadmat('../../../bin_data_from_nick/Re100svd_20_data.mat')
    data = data_mat.get('POD_COEF').T
    tspan = data_mat.get('total_time').ravel()
    scaling_factor = data_mat.get('largest_std')

    print('...reading file... size = ', data.shape)

    # split trajectory full data into three parts
    end_train_index = 700

    # training data
    train_data = data[0::3,:]
    train_tspan = tspan[0::3]

    # validation data
    valid_data = data[1::3,:] # odd but to the last
    valid_tspan = tspan[1::3]

    # test data
    test_data = data[2::3,:] # odd but to the last
    test_tspan = tspan[2::3]


    # plot even data to determine the final number in training index
    mkdir_p('./raw_data')
    for i in range(n_com):
        plt.figure(figsize=(20, 6))
        plt.plot(train_tspan, train_data[:,i],'k^-',label='train')
        plt.plot(valid_tspan, valid_data[:,i],'go-',label='valid')
        plt.plot(test_tspan, test_data[:,i],'b*-',label='test')
        plt.legend(loc='best')
        plt.xlabel('time')
        plt.ylabel('POD coef.')
        plt.savefig('./raw_data/' + str(i+1) + '_train_test_valid.png')
        plt.close()

    # mkdir for major path
    path = '20d_cylinder_noise_level_0'
    mkdir_p(path)

    # save the one for running training
    filename =  str(train_data.shape[0]) + '_trainData.npz'
    np.savez(path + '/' + filename, Xtrain=train_data, tspan=train_tspan, scaling_factor=scaling_factor)
    print('training data size = ', train_data.shape)

    # save data for validation
    filename =  str(valid_data.shape[0]) + '_validData.npz'
    np.savez(path + '/' + filename, Xtrain=valid_data, tspan=valid_tspan, scaling_factor=scaling_factor)
    print('valid data size = ', valid_data.shape)

    # save data for testing
    filename =  str(test_data.shape[0]) + '_testData.npz'
    np.savez(path + '/' + filename, Xtrain=test_data, tspan=test_tspan, scaling_factor=scaling_factor)
    print('test data size = ', test_data.shape)

    # make a dir for hyp
    hyp_path = './' +  path + '/hyp/' + path+ '/'
    mkdir_p(hyp_path)

    # save the train data set for hyp
    filename = str(train_data.shape[0]) + '_trainData.npz'
    np.savez(hyp_path + filename, Xtrain=train_data, tspan=train_tspan, scaling_factor=scaling_factor)

    pass


if __name__=='__main__':
    main()
