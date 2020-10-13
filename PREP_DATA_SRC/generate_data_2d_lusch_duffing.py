import sys
sys.path.insert(0, "../..")
sys.dont_write_bytecode = True

import numpy as np

from SKDMD.MODEL_SRC.ClassGenerateDataFromPhysics import ClassGenerateXXDotFromPhysics
from lib_analytic_model import F_simple_2d_system
from lib_analytic_model import F_duffing_2d_system


def main(case):

    if case == '2d_simple_lusch2017':

        # options

        noise_level = 0
        F_governing = F_simple_2d_system
        phase_space_range = [[-.5,.5],[-.5,.5]]
        num_samples_each_dim = 40
        range_of_X = np.array(phase_space_range)

        # generate data using ClassGenerateXXDotFromPhysics

        data_generator = ClassGenerateXXDotFromPhysics(directory='../EXAMPLES/lusch/', case_name=case,
                                                       noise_level=noise_level)
        data_generator.make_case_dir()
        data_generator.samplingX_Xdot(F=F_governing, range_of_X=range_of_X, num_samples_each_dim=num_samples_each_dim)
        data_generator.save_X_Xdot()

    elif case == '2d_duffing_otto2017':

        # options

        noise_level = 0
        F_governing = F_duffing_2d_system
        phase_space_range = [[-2, 2], [-2, 2]]
        num_samples_each_dim = 100
        range_of_X = np.array(phase_space_range)

        # generate data using ClassGenerateXXDotFromPhysics
        data_generator = ClassGenerateXXDotFromPhysics(directory='../EXAMPLES/duffing/', case_name=case,
                                                       noise_level=noise_level)
        data_generator.make_case_dir()
        data_generator.samplingX_Xdot(F=F_governing, range_of_X=range_of_X, num_samples_each_dim=num_samples_each_dim)
        data_generator.save_X_Xdot()


if __name__ == '__main__':
    case = '2d_simple_lusch2017'
    main(case)
