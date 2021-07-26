from matplotlib import pyplot as plt
import numpy as np

plt.style.use('siads')


def main():
    # read training data

    train_data = np.load('./2d_simple_lusch2017_noise_level_0/1600_trainData.npz')
    valid_data = np.load('./2d_simple_lusch2017_noise_level_0/c-edmd-h5-r35/valid_trajectory.npz')
    test_data = np.load('./2d_simple_lusch2017_noise_level_0/c-edmd-h5-r35/test_trajectory.npz')

    plt.figure(figsize=(8, 8))

    plt.scatter(train_data['Xtrain'][:, 0], train_data['Xtrain'][:, 1], s=14,label='train')

    plt.scatter(valid_data['trj'][:, 0], valid_data['trj'][:, 1], s=20,label='validation')

    plt.scatter(test_data['trj'][:, 0], test_data['trj'][:, 1], s=20,label='test')

    plt.xlim([-0.6,0.6])
    plt.ylim([-0.6,0.6])

    plt.xlabel(r'$x_1$',fontsize = 32)
    plt.ylabel(r'$x_2$',fontsize = 32)

    lgd = plt.legend(bbox_to_anchor=(1, 0.5))
    plt.savefig('data_distribution.png',
                bbox_inches='tight',
                bbox_extra_artists=(lgd,)
                )
    plt.close()


if __name__ == "__main__":
    main()
