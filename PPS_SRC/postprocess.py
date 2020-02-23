## postprocessing

import numpy as np
import sys
import pickle
import joblib
import matplotlib.ticker as ticker

sys.dont_write_bytecode = True
sys.path.insert(0, '../../')

from matplotlib import pyplot as plt
from SKDMD.PREP_DATA_SRC.source_code.lib.utilities import mkdir
from scipy.io import loadmat


# plt.style.use('siads')

plt.locator_params(axis='y', nbins=6)
plt.locator_params(axis='x', nbins=10)

FIG_SIZE = (8,8)
N_CPU = joblib.cpu_count()

def fmt(x, pos):
    a, b = '{:.2e}'.format(x).split('e')
    b = int(b)
    return r'${} \times 10^{{{}}}$'.format(a, b)

class ClassKoopmanPPS(object):

    def __init__(self, pps_dir, eval_dir, model_dir, params, draw_eigen_function_plot=True, compare_against_spdmd=False):

        self.pps_dir = pps_dir
        self.eval_dir = eval_dir
        self.model_dir = model_dir
        mkdir(self.pps_dir)
        self.params = params
        self.type = None
        self.dt = None
        self.index_selected_in_full = None
        self.draw_eigen_function_plot = draw_eigen_function_plot
        self.compare_against_spdmd = compare_against_spdmd

    def pps_eigenfunction(self):
        raise NotImplementedError("Postprocessing for eigenfunction need to be implemented!")

    def plot_eigenfunction_given_index_and_path(self, index_of_eigens, save_path):
        raise NotImplementedError("Postprocessing for eigenfunction need to be implemented!")

    def pps_eigenvalues(self):
        raise NotImplementedError("Postprocessing for eigenvalues need to be implemented!")

    def pps_eigenmodes_eval(self):
        raise NotImplementedError("Postprocessing for eigenmodes evaluation need to be implemented!")

    def pps_scatter_plot_eigenvalue(self, index_selected_array, path_to_save,zoomed_X_Y_max, case_specific_frequency_dict):
        raise NotImplementedError("Postprocessing for plotting scatter eigenvalues evaluation need to be implemented!")

    def pps_sweep_alpha(self, zoomed_X_Y_max=None, case_specific_frequency_dict={'draw_st':False}):

        fig_data = np.load(self.eval_dir + '/MultiTaskElasticNet_result.npz')
        alphas_enet = fig_data['alphas_enet']
        coefs_enet_comp = fig_data['coefs_enet_comp']

        # loop over all alpha
        for ii, alpha in enumerate(alphas_enet):
            print("current alpha = ", alpha, " index = ",ii)

            # 1 make directory
            alpha_dir_eval = self.eval_dir + '/sweep/sweep_alpha_' + str(alpha)
            alpha_dir = self.pps_dir + '/sweep_alpha_' + str(alpha)
            mkdir(alpha_dir)

            # 2 compute the index of selected modes
            non_zero_index_bool_array = np.linalg.norm(coefs_enet_comp[:, :, ii],axis=0) > 0
            further_selected_index_array = self.index_selected_in_full[non_zero_index_bool_array]

            print("number of non-zero coef = ", len(further_selected_index_array))

            if len(further_selected_index_array) > 0:

                # 3 draw the eigenvalue selected plot
                self.pps_scatter_plot_eigenvalue(index_selected_array=further_selected_index_array,
                                                 path_to_save=alpha_dir,
                                                 zoomed_X_Y_max=zoomed_X_Y_max,
                                                 case_specific_frequency_dict=case_specific_frequency_dict)

                # 4. load and draw trajectory comparison
                # load npz data from eval directory for parameter sweep
                fig_data = np.load(alpha_dir_eval + '/save_trj_comparison.npz')
                true_trajectory = fig_data['ttrj']
                pred_trajectory = fig_data['ptrj']
                true_tsnap = fig_data['tt']
                self.pps_plot_trajectory_given_pred(true_tsnap=true_tsnap,
                                                    pred_trajectory=pred_trajectory,
                                                    true_trajectory=true_trajectory,
                                                    path_to_save_fig=alpha_dir)
                # 5. load and draw eigenfunction plot
                if self.draw_eigen_function_plot:
                    self.plot_eigenfunction_given_index_and_path(index_of_eigens=further_selected_index_array, save_path=alpha_dir)
        return

    def pps_plot_trajectory_given_pred(self, true_tsnap, pred_trajectory, true_trajectory, path_to_save_fig):

        # # plot
        num_components = true_trajectory.shape[1]
        for i_comp in range(num_components):
            # plt.figure(figsize=FIG_SIZE)
            plt.figure(figsize=(8, 4))
            plt.plot(true_tsnap, true_trajectory[:, i_comp], 'k-', label='true')
            plt.plot(true_tsnap, pred_trajectory[:, i_comp], 'r--',label='pred')
            plt.xlabel('time',fontsize = 32)
            plt.ylabel(r'$x_{' + str(i_comp + 1) + '}$',fontsize = 32)
            lgd = plt.legend(bbox_to_anchor=(1, 0.5))
            plt.savefig(path_to_save_fig + '/component_' + str(i_comp + 1) + '.png',  bbox_extra_artists=(lgd,),
                        bbox_inches='tight')
            plt.close()

        if self.compare_against_spdmd:

            # load data from matlab solution of SPDMD
            data_dmd = loadmat('/home/shaowu/Documents/2016_PHD/PROJECTS/2019_aerospace/spdmd/20_pred.mat')
            # data_full_dmd = loadmat('/home/shaowu/Documents/2016_PHD/PROJECTS/2019_aerospace/spdmd/200_pred.mat')
            data_spdmd = loadmat('/home/shaowu/Documents/2016_PHD/PROJECTS/2019_aerospace/spdmd/200_sp_pred.mat')

            dmd_pred = data_dmd['x_list'].T
            # full_dmd_pred = data_full_dmd['x_list'].T
            sp_dmd_pred   = data_spdmd['x_list_2'].T[:,:-1]

            # plot them others
            for i_comp in range(num_components):
                # plt.figure(figsize=FIG_SIZE)
                plt.figure(figsize=(8,4))
                plt.plot(true_tsnap[:-1], true_trajectory[:-1, i_comp], 'k-', label='true')
                plt.plot(true_tsnap[:-1], pred_trajectory[:-1, i_comp], 'r--', label='spKDMD')
                plt.plot(true_tsnap[1:], dmd_pred[1:, i_comp], 'c--', label='DMD r=20')
                # plt.plot(true_tsnap[:-1], full_dmd_pred[:-1, i_comp], 'g--', label='full DMD (r=200)')
                plt.plot(true_tsnap[:-1], sp_dmd_pred[:, i_comp], 'b--', label='spDMD r=200')

                plt.xlabel('time', fontsize=32)
                plt.ylabel(r'$x_{' + str(i_comp + 1) + '}$', fontsize=32)
                lgd = plt.legend(bbox_to_anchor=(1, 0.5))
                plt.savefig(path_to_save_fig + '/vs_spdmd_component_' + str(i_comp + 1) + '.png', bbox_extra_artists=(lgd,),
                            bbox_inches='tight')
                plt.close()

    def pps_component_trj(self):

        ## LOAD SaveCompareWithTruth
        fig_data = np.load(self.eval_dir + '/save_trj_comparison.npz')
        true_trajectory = fig_data['ttrj']
        pred_trajectory = fig_data['ptrj']
        true_tsnap = fig_data['tt']

        self.pps_plot_trajectory_given_pred(true_tsnap=true_tsnap,
                                            pred_trajectory=pred_trajectory,
                                            true_trajectory=true_trajectory,
                                            path_to_save_fig=self.pps_dir)


    def pps_2d_data_dist(self, data_path):

        # only obtain the phase space locations
        data2D = np.load(data_path)['Xtrain']

        plt.figure(figsize=FIG_SIZE)
        plt.scatter(data2D[:,0], data2D[:,1], s=0.1 ,c='k')
        plt.xlabel(r'$x_1$',fontsize = 32)
        plt.ylabel(r'$x_2$',fontsize = 32)
        plt.savefig(
            self.pps_dir + '/trainPhaseDist.png',
            bbox_inches='tight'
        )
        plt.close()

    def get_cmap(self, n, name='rainbow'):
        '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
        RGB color; the keyword argument name must be a standard mpl colormap name.'''
        return plt.cm.get_cmap(name, n)


class ClassDictPPS(ClassKoopmanPPS):

    def pps_singular_value(self):
        raise NotImplementedError()

    def pps_eigenmodes_eval(self):

        ## LOAD ComputeSaveNormalizedEigenError: fig1
        fig1_data = np.load(self.eval_dir + '/ComputeSaveNormalizedEigenError_fig1.npz')
        normalized_relative_error = fig1_data['nre']
        true_tsnap = fig1_data['tt']
        linearEvolvingEigen = fig1_data['le']
        relative_error = self.params['relative_error']
        ## draw ComputeSaveNormalizedEigenError: fig1
        # plot normalized relative error for each eigenmodes
        plt.figure(figsize=FIG_SIZE)
        cmap = self.get_cmap(normalized_relative_error.shape[1])
        for i in range(normalized_relative_error.shape[1]):
            plt.plot(true_tsnap, normalized_relative_error[:, i], '-', c=cmap(i), label= str(i + 1) + 'th-eigenvalue: ' + "{0:.3f}".format(linearEvolvingEigen[i,i]))
        plt.xlabel('time',fontsize = 32)
        if relative_error:
            plt.ylabel('normalized error',fontsize = 32)
        else:
            plt.ylabel('error',fontsize = 32)
        plt.yscale('log')
        lgd = plt.legend(bbox_to_anchor=(1, 0.5))
        plt.savefig(self.pps_dir + '/normalized_relative_eigen_error.png',
                    bbox_extra_artists=(lgd,),
                    bbox_inches='tight')
        plt.close()

        ## LOAD ComputeSaveNormalizedEigenError: fig2
        fig2_data = np.load(self.eval_dir + '/ComputeSaveNormalizedEigenError_fig2.npz')
        mean_normalized_relative_error = fig2_data['mre']
        small_to_large_error_eigen_index = fig2_data['stli']
        small_to_large_error_eigen_index_kou = fig2_data['stli_kou']
        abs_sum_kou = fig2_data['abs_sum_kou']
        error_reconstruct_state_list = fig2_data['ersl']
        # bool_index_further = fig2_data['iestli'] ## it is removed..

        self.small_to_large_error_eigen_index = small_to_large_error_eigen_index
        # self.bool_index_further = bool_index_further

        ## draw ComputeSaveNormalizedEigenError: fig2
        # error ordered
        fig= plt.figure(figsize=FIG_SIZE)
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twinx()
        ax1.plot(range(1, normalized_relative_error.shape[1] + 1),
                 mean_normalized_relative_error[small_to_large_error_eigen_index],
                 'b-^', label='max relative eigenfunction error')
        ax1.set_xlabel(r'number of selected eigenmodes $\hat{L}$',fontsize = 32)
        # ax1.legend(bbox_to_anchor=(1, 0.5))
        ax1.set_yscale('log')
        if relative_error:
            ax1.set_ylabel('max linear evolving normalized error', color='b',fontsize = 32)
        else:
            ax1.set_ylabel('max error', color='b',fontsize = 32)
        # plot error from reconstruction state from eigenfunction values
        ax2.plot(range(1, normalized_relative_error.shape[1] + 1),
                 error_reconstruct_state_list,'r-o',
                 label='reconstruction normalized error')
        if relative_error:
            ax2.set_ylabel('reconstruction normalized error', color='r',fontsize = 32)
        else:
            ax2.set_ylabel('reconstruction error', color='r',fontsize = 32)
        # ax2.set_ylim([-1,20])
        ax2.set_yscale('log')

        # set up ticks
        yticks = ticker.LogLocator()
        ax1.yaxis.set_major_locator(yticks)
        ax2.yaxis.set_major_locator(yticks)

        # ax2.legend(bbox_to_anchor=(1, 0.5))
        plt.savefig(self.pps_dir + '/reConstr_decay_normalized_relative_eigen_error.png',
                    bbox_inches='tight')
        plt.close()

        ## LOAD ComputeSaveNormalizedEigenError: fig3
        fig3_data = np.load(self.eval_dir + '/ComputeSaveNormalizedEigenError_fig3.npz')
        top_k_modes_list = fig3_data['tkm_index_list']

        self.top_k_modes_list = top_k_modes_list

        # print out kou's result
        print('as a comparison: index chosen by Kou criterion: ')
        print(small_to_large_error_eigen_index_kou + 1)
        print('corresponding abs sum:')
        print(abs_sum_kou)

        self.index_selected_in_full = self.small_to_large_error_eigen_index[:self.top_k_modes_list[-1] + 1]
        # self.index_selected_in_full = self.small_to_large_error_eigen_index

        ## draw ComputeSaveNormalizedEigenError: fig3
        # fig. 3: plot normalized relative error for top K smallest error eigenmodes
        plt.figure(figsize=FIG_SIZE)
        cmap = self.get_cmap(len(top_k_modes_list))
        for i in top_k_modes_list:
            i_s = small_to_large_error_eigen_index[i]
            plt.plot(true_tsnap, normalized_relative_error[:, i_s], '-', c=cmap(i),
                     label= str(i_s + 1) + 'th-eigenvalue: ' + "{0:.3f}".format(linearEvolvingEigen[i_s,i_s]))
            # print eigenvectors
            # print 'no.  eigen vectors ', i_s+1
            # print self.model.KoopmanEigenV[:, i_s]
        plt.xlabel('time',fontsize = 32)
        if relative_error:
            plt.ylabel('normalized error',fontsize = 32)
        else:
            plt.ylabel('error',fontsize = 32)
        plt.yscale('log')
        lgd = plt.legend(bbox_to_anchor=(1, 0.5))
        plt.savefig(self.pps_dir +  '/top_' + str(len(top_k_modes_list)) + '_normalized_relative_eigen_error.png',
                    bbox_extra_artists=(lgd,),
                    bbox_inches='tight')
        plt.close()

        # load MTENET
        fig4_data = np.load(self.eval_dir + '/MultiTaskElasticNet_result.npz')
        alphas_enet = fig4_data['alphas_enet']
        coefs_enet = fig4_data['coefs_enet']
        residual_array = fig4_data['residual_array']

        # coefficients vs alpha & number non-zero

        num_target_components = coefs_enet.shape[0]
        alphas_enet_log_negative = -np.log10(alphas_enet)

        # print("coef_enet real= ", np.real(coefs_enet))
        # print("coef_enet imag= ", np.imag(coefs_enet))

        for i_component in range(num_target_components):

            plt.figure(figsize=FIG_SIZE)
            cmap = self.get_cmap(len(top_k_modes_list))
            for i in top_k_modes_list:
                i_s = small_to_large_error_eigen_index[i]
                plt.plot(alphas_enet_log_negative, abs(coefs_enet[i_component,i,:]), '-*', c=cmap(i),
                         label = 'No. ' + str(i + 1) + ', index = ' + str(i_s+1))
            max_val = np.max(abs(coefs_enet[i_component, :, -1]))
            min_val = np.min(abs(coefs_enet[i_component, :, -1]))

            diss = (max_val - min_val)/2
            mean = (max_val + min_val)/2

            plt.xlabel(r'-$\log_{10}(\alpha)$',fontsize = 32)
            plt.ylabel('abs of coefficients',fontsize = 32)
            plt.ylim([mean - diss*1.05, mean + diss*3])

            lgd = plt.legend(bbox_to_anchor=(1, 0.5))
            plt.savefig(self.pps_dir + '/multi-elastic-net-coef-' + str(i_component+1) + '.png',
                        bbox_extra_artists=(lgd,),
                        bbox_inches='tight')
            plt.close()

            # total number of non-zero terms1

            plt.figure(figsize=FIG_SIZE)
            num_non_zeros = [len((coefs_enet[i_component, abs(coefs_enet[i_component, :, ii]) >0*np.max(abs(coefs_enet[i_component,:,ii])), ii]))
                             for ii in range(coefs_enet.shape[2])]
            plt.plot(alphas_enet_log_negative, num_non_zeros , 'k^-')
            plt.xlabel(r'-$\log_{10}(\alpha)$',fontsize = 32)
            plt.ylabel('number of selected features',fontsize = 32)
            lgd = plt.legend(bbox_to_anchor=(1, 0.5))
            plt.savefig(self.pps_dir + '/multi-elastic-net-coef-non-zeros-' + str(i_component+1) + '.png',
                        bbox_extra_artists=(lgd,),
                        bbox_inches='tight')
            plt.close()


        num_non_zero_all_alpha = []

        for ii in range(coefs_enet.shape[2]):

            non_zero_index_per_alpha = []
            for i_component in range(num_target_components):
                # non_zero_index_per_alpha_per_target = abs(coefs_enet[i_component, :, ii]) > 0
                non_zero_index_per_alpha_per_target = abs(coefs_enet[i_component, :, ii]) > 0*np.max(abs(coefs_enet[i_component, :, ii]))
                non_zero_index_per_alpha.append(non_zero_index_per_alpha_per_target)
            non_zero_index_per_alpha_all_targets = np.logical_or.reduce(non_zero_index_per_alpha)
            num_non_zero_all_alpha.append(np.sum(non_zero_index_per_alpha_all_targets))

        num_non_zero_all_alpha = np.array(num_non_zero_all_alpha)

        # total residual vs alpha AND number of non-zero modes vs alpha

        fig=plt.figure(figsize=FIG_SIZE)
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twinx()
        ax1.plot(alphas_enet_log_negative, residual_array, 'k*-')
        ax1.set_xlabel(r'-$\log_{10}(\alpha)$',fontsize = 32)
        ax1.set_ylabel('normalized reconstruction MSE',color='k',fontsize = 32)
        # ax1.set_yscale('log')

        ax2.plot(alphas_enet_log_negative, num_non_zero_all_alpha,'r*-')
        ax2.set_ylabel('number of selected features',color='r',fontsize = 32)
        lgd = plt.legend(bbox_to_anchor=(1, 0.5))
        plt.savefig(self.pps_dir + '/multi-elastic-net-mse.png',
                    bbox_extra_artists=(lgd,),
                    bbox_inches='tight')
        plt.close()

    def pps_scatter_plot_eigenvalue(self, index_selected_array,
                                    path_to_save,
                                    zoomed_X_Y_max=None,
                                    mag_contrib_all_kmd=None,
                                    case_specific_frequency_dict={'draw_st':False}):
        ## read eigenvalues from model
        ev = self.model.Koopman['eigenvalues']

        ## finally I decided to draw discrete time in discrete time
        # if self.model.type == 'd':
        #     ev = np.log(ev) / self.dt

        D_real = np.real(ev)
        D_imag = np.imag(ev)
        # 1+. eigenvalue distribution
        plt.figure(figsize=FIG_SIZE)
        plt.grid()
        # draw all of the eigenvalue in the rank truncated case, i.e., original version of reduced KDMD
        plt.scatter(D_real, D_imag, c='b', label='full')
        # draw the one selected
        plt.scatter(D_real[index_selected_array], D_imag[index_selected_array], c='r', label='selected')

        # draw circle
        theta = np.linspace(0, 2*np.pi ,200)
        plt.plot(np.cos(theta), np.sin(theta), 'k-',alpha=0.2)

        plt.xlabel(r'Real($\lambda$)',fontsize = 32)
        plt.ylabel(r'Imag($\lambda$)',fontsize = 32)
        lgd = plt.legend(bbox_to_anchor=(1, 0.5))

        # if zoomed version
        if type(zoomed_X_Y_max) != type(None):

            plt.savefig(path_to_save + '/koopmanEigVal.png', bbox_inches='tight', bbox_extra_artists=(lgd,))

            if case_specific_frequency_dict['draw_st']:

                ## here we add St-lines-plot
                theta_for_st = np.linspace(0, np.pi/2, 50)
                length_array = np.linspace(0, 1.0, 2)

                max_num_tot = case_specific_frequency_dict['max_num_st_lines']

                tot = 0
                for i, theta_line in enumerate(theta_for_st):
                    plt.plot(length_array*np.cos(theta_line), length_array*np.sin(theta_line), 'k--', alpha=0.3)
                    if tot < max_num_tot:
                        if i % 5 == 0:
                            tot += 1

                            St_sample = case_specific_frequency_dict['St_sample']

                            # cylinder case
                            # dt = 0.6
                            # U_infty = 1
                            # D = 2
                            # St_sample = D/(dt*U_infty)
                            St = St_sample * theta_line/(2*np.pi)

                            # ship case

                            #
                            if i == 0:
                                s = 'St=0'
                            else:
                                s = 'St='+"{0:.2f}".format(St)
                            plt.plot(length_array * np.cos(theta_line), length_array * np.sin(theta_line), 'k-')
                            plt.text(1.005*length_array[-1]*np.cos(theta_line), 1.005*length_array[-1]*np.sin(theta_line), s,
                                     rotation=theta_line/np.pi*180*0.3 )

                if case_specific_frequency_dict['characteristic_st_list'] != None:
                    for st_char_color_pair in case_specific_frequency_dict['characteristic_st_list']:

                        st_char = st_char_color_pair[0]

                        theta_st = st_char / St_sample * 2 * np.pi
                        plt.plot(length_array * np.cos(theta_st), length_array * np.sin(theta_st), st_char_color_pair[1])

                        # Std = 0.3012
                        # Stl = 0.1506
                        # # # Re 100
                        # # Std = 0.3386
                        # # Stl = 0.1694
                        # # Re 130
                        # # Std = 0.3634
                        # # Stl = 0.1816

            plt.xlim(zoomed_X_Y_max[0])
            plt.ylim(zoomed_X_Y_max[1])

            plt.savefig(path_to_save + '/koopmanEigVal_zoomed.png', bbox_inches='tight', bbox_extra_artists=(lgd,))
        else:
            plt.savefig(path_to_save + '/koopmanEigVal.png', bbox_inches='tight', bbox_extra_artists=(lgd,))

        # last, plot the reference plot for eigenvalue and its number
        for index in index_selected_array:
            plt.text(D_real[index], D_imag[index], str(index))
        plt.savefig(path_to_save + '/koopmanEigVal_numbered.png', bbox_inches='tight', bbox_extra_artists=(lgd,))

        plt.close()

        if self.compare_against_spdmd:
            # plot against spDMD result
            plt.figure(figsize=FIG_SIZE)
            plt.grid()
            # draw all of the eigenvalue in the rank truncated case, i.e., original version of reduced KDMD
            # plt.scatter(D_real, D_imag, s=30, marker='3',c='b', label='full KDMD')
            # draw the one selected
            plt.scatter(D_real[index_selected_array], D_imag[index_selected_array], s=150, marker='o',c='r', label='spKDMD',edgecolors='k')

            # load data from matlab solution of SPDMD
            data_dmd = loadmat('/home/shaowu/Documents/2016_PHD/PROJECTS/2019_aerospace/spdmd/20_pred.mat')
            # data_full_dmd = loadmat('/home/shaowu/Documents/2016_PHD/PROJECTS/2019_aerospace/spdmd/200_pred.mat')
            data_spdmd = loadmat('/home/shaowu/Documents/2016_PHD/PROJECTS/2019_aerospace/spdmd/200_sp_pred.mat')

            # normal DMD
            D_DMD_real = np.real(np.exp(data_dmd['Edmd']*self.dt))
            D_DMD_imag = np.imag(np.exp(data_dmd['Edmd']*self.dt))

            # draw full DMD
            # D_full_DMD_real = np.real(np.exp(data_full_dmd['Edmd']*self.dt))
            # D_full_DMD_imag = np.imag(np.exp(data_full_dmd['Edmd']*self.dt))

            # draw spDMD
            # D_sp_DMD_real = np.real(np.exp(data_spdmd['Edmd_select']*self.dt))
            # D_sp_DMD_imag = np.imag(np.exp(data_spdmd['Edmd_select']*self.dt))
            D_sp_DMD_real = np.real(np.exp(data_spdmd['Edmd_select']*self.dt))
            D_sp_DMD_imag = np.imag(np.exp(data_spdmd['Edmd_select']*self.dt))


            plt.scatter(D_DMD_real, D_DMD_imag, s=80, marker='d', c='yellow', label='DMD r=20',edgecolors='k')
            # plt.scatter(D_full_DMD_real, D_full_DMD_imag, s=60,marker='s', c='lime', label='full DMD (r=200)',edgecolors='k')
            plt.scatter(D_sp_DMD_real, D_sp_DMD_imag, s=50,marker='^',c='b', label='spDMD r=200',edgecolors='k')

            # draw circle
            theta = np.linspace(0, 2 * np.pi, 200)
            plt.plot(np.cos(theta), np.sin(theta), 'k-',alpha=0.2)

            plt.xlabel(r'Real($\lambda$)', fontsize=32)
            plt.ylabel(r'Imag($\lambda$)', fontsize=32)
            lgd = plt.legend(bbox_to_anchor=(1, 0.5))

            # if zoomed version
            if type(zoomed_X_Y_max) != type(None):

                plt.savefig(path_to_save + '/vs_spdmd_koopmanEigVal.png', bbox_inches='tight', bbox_extra_artists=(lgd,))

                if case_specific_frequency_dict['draw_st']:

                    ## here we add St-lines-plot
                    theta_for_st = np.linspace(0, np.pi / 2, 50)
                    length_array = np.linspace(0, 1.0, 2)

                    max_num_tot = case_specific_frequency_dict['max_num_st_lines']

                    tot = 0
                    for i, theta_line in enumerate(theta_for_st):
                        plt.plot(length_array * np.cos(theta_line), length_array * np.sin(theta_line), 'k--', alpha=0.3)
                        if tot < max_num_tot:
                            if i % 5 == 0:
                                tot += 1

                                St_sample = case_specific_frequency_dict['St_sample']

                                # cylinder case
                                # dt = 0.6
                                # U_infty = 1
                                # D = 2
                                # St_sample = D/(dt*U_infty)
                                St = St_sample * theta_line / (2 * np.pi)

                                # ship case

                                #
                                if i == 0:
                                    s = 'St=0'
                                else:
                                    s = 'St=' + "{0:.2f}".format(St)
                                plt.plot(length_array * np.cos(theta_line), length_array * np.sin(theta_line), 'k-')
                                plt.text(1.005 * length_array[-1] * np.cos(theta_line), 1.005 * length_array[-1] * np.sin(theta_line), s,
                                         rotation=theta_line / np.pi * 180 * 0.3)

                    for st_char_color_pair in case_specific_frequency_dict['characteristic_st_list']:
                        st_char = st_char_color_pair[0]

                        theta_st = st_char / St_sample * 2 * np.pi
                        plt.plot(length_array * np.cos(theta_st), length_array * np.sin(theta_st), st_char_color_pair[1])


                plt.xlim(zoomed_X_Y_max[0])
                plt.ylim(zoomed_X_Y_max[1])
                plt.savefig(path_to_save + '/vs_spdmd_koopmanEigVal_zoomed.png', bbox_inches='tight', bbox_extra_artists=(lgd,))
            else:
                plt.savefig(path_to_save + '/vs_spdmd_koopmanEigVal.png', bbox_inches='tight', bbox_extra_artists=(lgd,))
            plt.close()

        # print the final number of koopmans
        print("final number of modes = ", len(D_real[index_selected_array]))

        # if we would plot the mag. percent. of each mode
        if type(mag_contrib_all_kmd) != type(None):
            plt.figure(figsize=FIG_SIZE)
            plt.plot(np.arange(1, len(mag_contrib_all_kmd)+1), mag_contrib_all_kmd)
            plt.xticks(np.arange(1, len(mag_contrib_all_kmd)+1))
            plt.xlabel('index of eigenvalue',fontsize = 32)
            plt.ylabel('mag. percent. of each eigen-modes',fontsize = 32)
            plt.close()

            np.savez(self.pps_dir + 'eigenvalue_circle.npz',
                     ev_real=D_real[index_selected_array],
                     ev_imag=D_imag[index_selected_array],
                     mag_percent=mag_contrib_all_kmd)


    def pps_eigenvalues(self, zoomed_X_Y_max=None, mag_contrib_all_kmd=None,case_specific_frequency_dict={'draw_st':False}):

        # simply draw the scatter plot of eigenvalue
        self.pps_scatter_plot_eigenvalue(index_selected_array=self.index_selected_in_full, path_to_save=self.pps_dir,
                                         zoomed_X_Y_max=zoomed_X_Y_max,mag_contrib_all_kmd=mag_contrib_all_kmd,
                                         case_specific_frequency_dict=case_specific_frequency_dict)
        return

    def plot_eigenfunction_given_index_and_path(self, index_of_eigens, save_path):

        ## load eigenfunction data
        fig_data = np.load(self.model_dir + '/koopman_eigenfunctions.npz')

        numKoopmanModes = fig_data['numKoopmanModes']
        phi_eigen_array = fig_data['phi_eigen_array']
        ndraw = fig_data['ndraw']
        D_real = fig_data['D_real']
        D_imag = fig_data['D_imag']
        x1_ = fig_data['x1_']
        x2_ = fig_data['x2_']

        for ikoopman in index_of_eigens:
            # R_i = R[:, ikoopman:ikoopman + 1]
            # phi_eigen = np.matmul(phi_array, R_i)
            phi_eigen = phi_eigen_array[:, ikoopman:ikoopman + 1]
            phi_eigen_mesh = phi_eigen.reshape((ndraw, ndraw))

            # draw || mag. of eigenfunction
            plt.figure(figsize=FIG_SIZE)
            plt.xlabel(r'$x_1$',fontsize = 32)
            plt.ylabel(r'$x_2$',fontsize = 32)
            plt.title(r'$\lambda$ = ' + "{0:.3f}".format(D_real[ikoopman]) + ' + ' + "{0:.3f}".format(D_imag[ikoopman]) + 'i')
            plt.contourf(x1_, x2_, np.abs(phi_eigen_mesh), 100, cmap=plt.cm.get_cmap('jet'))
            plt.colorbar(format=ticker.FuncFormatter(fmt))
            plt.savefig(save_path + '/koopmanEigFunct_MAG_mode_' + str(ikoopman + 1) + '.png', bbox_inches='tight')
            plt.close()

            # draw phase angle of eigenfunction
            plt.figure(figsize=FIG_SIZE)
            plt.xlabel(r'$x_1$',fontsize = 32)
            plt.ylabel(r'$x_2$',fontsize = 32)
            plt.title(r'$\lambda$ = ' + "{0:.3f}".format(D_real[ikoopman]) + ' + ' + "{0:.3f}".format(D_imag[ikoopman]) + 'i')
            plt.contourf(x1_, x2_, np.angle(phi_eigen_mesh), 100, cmap=plt.cm.get_cmap('jet'))
            plt.colorbar(format=ticker.FuncFormatter(fmt))
            plt.savefig(save_path + '/koopmanEigFunct_ANG_mode_' + str(ikoopman + 1) + '.png', bbox_inches='tight')
            plt.close()


    def pps_eigenfunction(self):

        self.plot_eigenfunction_given_index_and_path(index_of_eigens=self.index_selected_in_full, save_path=self.pps_dir)



class ClassEDMDPPS(ClassDictPPS):

    def pps_singular_value(self):

        # load full singular value data

        fig_data = np.load(self.eval_dir + '/sv.npz')
        full_sv = fig_data['full_sv']  # this is the Ghat eigenvalues without any truncation

        # plot singular value decay

        plt.figure(figsize=FIG_SIZE)
        plt.plot(np.arange(1, 1+len(full_sv)), full_sv, 'k-o', markersize=1)
        plt.xlabel('number of terms',fontsize = 32)
        plt.yscale('log')
        plt.ylabel(r'singular values of $\Phi_X$',fontsize = 32)
        lgd = plt.legend(bbox_to_anchor=(1, 0.5))
        plt.savefig(self.pps_dir + '/sv_full_phi_x.png', bbox_inches='tight', bbox_extra_artists=(lgd,))
        plt.close()

    def __init__(self, pps_dir, eval_dir, model_dir, params,draw_eigen_function_plot=True):
        super(ClassEDMDPPS, self).__init__(pps_dir, eval_dir, model_dir, params,draw_eigen_function_plot)
        model_path = model_dir + '/edmd.model'
        self.model = pickle.load(open(model_path, "rb"))
        self.type = self.model.type

    def pps_2d_simple_lusch_effect_svd_on_phi(self):

        ## LOAD data: compute_save_svd_effect_on_phi_edmd
        fig_data = np.load(self.eval_dir + '/compute_save_svd_effect_on_phi_edmd.npz')
        phi_ev_after_svd_list = fig_data['phi_r_ev_list']
        phi_ev_before_svd = fig_data['phi_ev']

        cmap = self.get_cmap(len(phi_ev_after_svd_list))

        plt.figure(figsize=FIG_SIZE)
        for i in range(len(phi_ev_after_svd_list)):
            plt.plot(range(1, phi_ev_before_svd.shape[1] + 1), np.linalg.norm(phi_ev_after_svd_list[i], axis=0),
                     color=cmap(i), label='r = '+str(i+1),alpha=0.7)
        plt.plot(range(1, phi_ev_before_svd.shape[1] + 1), np.linalg.norm(phi_ev_before_svd, axis=0), 'o-' ,
                 label='before SVD')
        plt.xlabel(r'feature index',fontsize = 32)
        plt.yscale('log')
        plt.ylabel(r'$\Vert \Phi V_i \Vert_2$',fontsize = 32)
        lgd = plt.legend(bbox_to_anchor=(1, 0.5))
        plt.savefig(self.pps_dir + '/effect_svd_on_phi.png', bbox_inches='tight', bbox_extra_artists=(lgd,))
        # plt.show()
        plt.close()

    def pps_lsq_spectrum(self):

        X = self.model.Phi_X_i_sample
        Y = self.model.Phi_Xdot_i_sample

        np.savez(self.pps_dir + '/lsq_spectrum.npz', X=X,Y=Y)


class ClassKDMDPPS(ClassDictPPS):

    def __init__(self, pps_dir, eval_dir, model_dir, params,draw_eigen_function_plot=True,compare_against_spdmd=False):
        super(ClassKDMDPPS, self).__init__(pps_dir, eval_dir, model_dir, params, draw_eigen_function_plot, compare_against_spdmd)
        model_path = model_dir + '/kdmd.model'
        self.model = pickle.load(open(model_path, "rb"))
        self.type = self.model.type
        if self.type == 'd':
            self.dt = self.model.dt
            print('discrete mode, with dt = ', self.dt)

    def pps_singular_value(self):

        # load full singular value data

        fig_data = np.load(self.eval_dir + '/sv_squared.npz')
        full_sv_squared = fig_data['full_sv_squared']  # this is the Ghat eigenvalues without any truncation

        # cut off at 1e-7 due to double precision

        full_sv_squared = full_sv_squared[np.abs(full_sv_squared) > 1e-14]

        # plot singular value decay

        plt.figure(figsize=FIG_SIZE)
        plt.plot(np.arange(1, 1+len(full_sv_squared)), np.sqrt(np.abs(full_sv_squared[::-1])),'k-o',markersize=1)
        plt.xlabel('number of terms',fontsize = 32)
        plt.yscale('log')
        plt.ylabel(r'singular values of $\Phi_X$',fontsize = 32)
        lgd = plt.legend(bbox_to_anchor=(1, 0.5))
        plt.savefig(self.pps_dir + '/sv_full_phi_x.png', bbox_inches='tight', bbox_extra_artists=(lgd,))
        plt.close()

#
# if __name__=='__main__':
#     # testing on
#     # case = '2d_lusch'
#     case = '2d_duffing'
#
#     if case == '2d_lusch':
#
#
#
#     elif case == '2d_duffing':
#
#         # 3. postprocessing for DLDMD model + apo eval
#         pps_dldmd = ClassDLPPS(pps_dir='./2d_duffing_otto2017/dldmd',
#                                eval_dir='../eval_src/2d_duffing_otto2017-dl',
#                                model_dir='../model_src/result/2d_duffing_otto2017/dldmd_2019-02-16-03-58-22/model_saved',
#                                params={}
#                                )
#
#         # plot eigenvalues
#         pps_dldmd.pps_eigenvalues()
#
#         # plot eigenfunctions
#         pps_dldmd.pps_eigenfunction()
#
#         # plot aposteriori component trj comparison
#         pps_dldmd.pps_component_trj()
#
#         # plot learning curve
#         pps_dldmd.pps_learning_curve()
