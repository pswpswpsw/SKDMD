#!/usr/bin/env python
# coding: utf-8

# In[1]:

import pandas as pd
from matplotlib import pyplot as plt

# plt.style.use('siads')

class aprior_plot(object):

    def __init__(self, csv_path):
        self.path = csv_path
        self.df = pd.read_csv(csv_path)

    def plot(self):
        groups = self.df.groupby('rank')
        colormap = self.get_cmap(len(groups)+1)

        # error plot
        plt.figure(figsize=(10,8))
        index = 0
        for name, group in groups:
            # c=np.random.rand(3,)
            c = colormap(index)
            plt.loglog(group['sigma'], group['ave. train error'],'-o',c=c,label='rank = '+ str(name))
            plt.loglog(group['sigma'], group['ave. test error'],'--o',c=c,label='rank = '+ str(name))
            plt.loglog(group['sigma'], group['ave. train rec error'], '-^', c=c, label='rank = ' + str(name))
            plt.loglog(group['sigma'], group['ave. test rec error'], '--^', c=c, label='rank = ' + str(name))

            index += 1

            print("===================")
            print("")
            print("rank = ", str(name))
            print("Sigma corresponds to MIN single step error on test data = ", group['sigma'][group['ave. test error'].idxmin()])
            print("Corresponding MIN single step error = ", group['ave. test error'][group['ave. test error'].idxmin()])

        plt.xlabel('sigma', fontsize=32)
        plt.ylabel('error', fontsize=32)
        plt.ylim([0.8*self.df['ave. train error'].min(), 1e2])
        lgd = plt.legend(bbox_to_anchor=(1, 0.5))
        plt.savefig('result-error.png',dpi=300, bbox_extra_artists=(lgd,),bbox_inches='tight')
        plt.close()

        # num plot
        plt.figure(figsize=(10, 8))
        index = 0
        for name, group in groups:
            c = colormap(index)
            # although it is "train", in the code, it is actually test .num.
            plt.semilogx(group['sigma'], group['ave. num for both train and test'], '-o', c=c, label='rank = ' + str(name))
            # plt.semilogy(group['sigma'], group['ave. test num'], '--o', c=c, label='rank = ' + str(name))
            index += 1

            print("===================")
            print("")
            print("for rank = ", str(name))
            print("Sigma corresponds to MAX num. good eigenfunction in both data = ", group['sigma'][group['ave. num for both train and test'].idxmax()])
            print("Corresponding MAX num. good eigenfunction = ", group['ave. num for both train and test'][group['ave. num for both train and test'].idxmax()])

        plt.ylim([0, 1.1*self.df['ave. num for both train and test'].max()])
        plt.xlabel(r'$\sigma$', fontsize=32)
        plt.ylabel('number of eigenfunctions with\n mean relative error below threshold', fontsize=32)
        lgd = plt.legend(bbox_to_anchor=(1, 0.5))
        plt.savefig('result-num.png', dpi=300, bbox_extra_artists=(lgd,),bbox_inches='tight')
        plt.close()

    @staticmethod
    def get_cmap(n, name='hsv'):
        '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
        RGB color; the keyword argument name must be a standard mpl colormap name.'''
        return plt.cm.get_cmap(name, n)
