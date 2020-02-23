#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generating data using LHS according to the physics governing equation"""

import numpy as np
import sys
sys.path.insert(0, '../../../')
sys.dont_write_bytecode = True

from pyDOE import lhs
from SKDMD.PREP_DATA_SRC.source_code.lib.utilities import mkdir

class ClassGenerateXXDotFromPhysics(object):
    """Generate time derivative data according to the physics :math:`\dot{x}=F(x)`.

    Args:
        directory (:obj:`str`): Path to put my all cases. Default value is ``../../EXAMPLES/``

        case_name (:obj:`str`): Name of my problem case.

        noise_level (:obj:`int`): Percentage of signal-to-noise ratio.


    Attributes:
        dir (:obj:`str`): Path to put cases.

        sub_dir (:obj:`str`): Path to put each data within each cases. The size and noisy
            level of the data can be varied.

        noise_level (:obj:`int`): Signal-to-noise percentage in terms of percentage.

        XdotTrain (:obj:`numpy.ndarray`): Data contains time derivatives to be saved.
            ``shape`` = (``n_samples``, ``n_dim``)

        Xtrain (:obj:`numpy.ndarray`): Data contains states to be saved.
            ``shape`` = (``n_samples``, ``n_dim``)

        JFTrain (:obj:`numpy.ndarray`): Data contains the Jacobian to be saved.
            ``shape`` = (``n_samples``, ``n_dim``)

        case_name (:obj:`str`): Name for the problem case.

        num_samples (:obj:`int`): Total number of data samples.

        num_samples_each_dim (:obj:`int`): Number of samples on each dimension.

    """

    def __init__(self, directory='../../EXAMPLES/', case_name=None, noise_level=0):
        self.dir = directory
        self.sub_dir = self.dir + case_name + '_noise_level_' + str(noise_level)
        self.noise_level = noise_level
        self.XdotTrain = None
        self.Xtrain = None
        self.JFTrain = None
        self.case_name = case_name
        self.num_samples = None
        self.num_samples_each_dim=None

    def make_case_dir(self):
        """Create directory for cases"""

        mkdir(directory=self.dir)
        mkdir(directory=self.sub_dir)

    def samplingX_Xdot(self, F, range_of_X, num_samples_each_dim=10):
        """Sampling X and Xdot given ground true governing equation using LatinHyperCubic from pyDOE_

        .. _pyDOE: https://pythonhosted.org/pyDOE/

        .. _LatinHyperCubic: https://pythonhosted.org/pyDOE/randomized.html#latin-hypercube

        In default, we use LatinHyperCubic_, :meth:`pyDOE.lhs` to generate the sampling
        for the design of experiments (DOE) in the hyper-cubic on :math:`[0, 1]`. Then we scale it back
        according to :attr:`range_of_X`. In default, we use `center` criterion and only one iteration.
        Then we call function :attr:`F` to get the time derivative :math:`\dot{x}` and Jacobian. However,
        in the most recent code, we don't need Jacobian. So one can simply return 0 array.

        Args:

            F (:obj:`function`): a Python function that calls the state :math:`x` and return the
                time derivative :math:`\dot{x}` and Jacobian. However Jacobian is not used any more so one
                can just leave it as zero array.

            range_of_X (:obj:`numpy.ndarray`): The range for each component to get sampled from.

            num_samples_each_dim (:obj:`int`): Number of samples for each component. Default value is 10.

        """

        # We first get the number of components.

        num_dim_X = range_of_X.shape[0]

        # We set the number of samples per dimension.

        self.num_samples_each_dim = num_samples_each_dim

        # We compute the mean and factor for later scaling back from [0, 1] to our desired range.

        mean_of_range = np.mean(range_of_X, axis=1)
        factor_of_range = (range_of_X[:, 1] - range_of_X[:, 0]) / 2.0

        # Generate latin-hypercubic in [0, 1]

        zero_to_one_LHS_location = lhs(n=num_dim_X,
                                       samples=self.num_samples_each_dim ** num_dim_X,
                                       criterion='center',
                                       iterations=1)

        # We get the number of samples

        self.num_samples = zero_to_one_LHS_location.shape[0]

        # For each dimension, we scale the sample points into desired location in real physical space

        for index_var in range(num_dim_X):
            for index_sample in range(self.num_samples):
                zero_to_one_LHS_location[index_sample, index_var] = \
                    (2 * zero_to_one_LHS_location[index_sample, index_var] - 1) * factor_of_range[index_var] \
                    + mean_of_range[index_var]

        # The states locations are overwritten with the location.

        self.Xtrain = zero_to_one_LHS_location

        # We now call F to generate time derivative and Jacobian.

        self.XdotTrain = np.zeros(zero_to_one_LHS_location.shape)
        self.JFTrain = np.zeros((self.XdotTrain.shape[0], self.XdotTrain.shape[1], self.XdotTrain.shape[1]))
        for index_sample in range(self.num_samples):
            self.XdotTrain[index_sample, :], self.JFTrain[index_sample, :, :] = F(self.Xtrain[index_sample, :])

        # Deappreciated: Adding noises into the data

        scale_x = np.std(self.Xtrain, axis=0)
        scale_xdot = np.std(self.XdotTrain, axis=0)
        scale_jf = np.std(self.JFTrain, axis=0)

        for index_sample in range(self.num_samples):
            self.Xtrain[index_sample, :]    += np.random.normal(0, self.noise_level / 100.0 * scale_x)
            self.XdotTrain[index_sample, :] += np.random.normal(0, self.noise_level / 100.0 * scale_xdot)
            self.JFTrain[index_sample, :]   += np.random.normal(0, self.noise_level / 100.0 * scale_jf)

    def save_X_Xdot(self):
        """Save data with :attr:`XdotTrain`, :attr:`Xtrain`, :attr:`JFTrain` as keys in the :attr:`sub_dir`
        with ``num_samples_trainData.npz``.

        """

        np.savez(self.sub_dir + '/' + str(self.num_samples) + '_trainData.npz',
                 XdotTrain=self.XdotTrain,
                 Xtrain=self.Xtrain,
                 JFTrain=self.JFTrain)