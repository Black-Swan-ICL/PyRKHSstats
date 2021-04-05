import numpy as np

import GPy


# TODO document
# TODO test
# TODO, why input dim = 1 ? Make more general
# TODO should independence_test_func enclose the test level ?
def two_stage_gp_resit(variable_x, variable_y, variable_z, locations_s,
                       independence_test_func, kernel_x_on_s=GPy.kern.RBF,
                       kernel_y_on_s=GPy.kern.RBF, kernel_z_on_s=GPy.kern.RBF,
                       kernel_x_on_z=GPy.kern.RBF, kernel_y_on_z=GPy.kern.RBF):

    def gp_regression(x, y, kernel):

        # Regress Y on X using a Gaussian Process with the specified kernel
        kernel = kernel(input_dim=1)
        mean_y_on_x = GPy.models.GPRegression(x, y, kernel)
        residuals_y_on_x = y - mean_y_on_x.predict(x)[0]

        return residuals_y_on_x

    def pre_whitening(x, y, z, s, kern_x_on_s, kern_y_on_s, kern_z_on_s):
        # Regress X on locations S using a GP regression
        res_x_on_s = gp_regression(s, x, kern_x_on_s)
        # Regress Y on locations S using a GP regression
        res_y_on_s = gp_regression(s, y, kern_y_on_s)
        # Regress Z on locations S using a GP regression
        res_z_on_s = gp_regression(s, z, kern_z_on_s)

        return res_x_on_s, res_y_on_s, res_z_on_s

    if locations_s is None:
        res_x = variable_x
        res_y = variable_y
        res_z = variable_z
    else:
        res_x, res_y, res_z = pre_whitening(x=variable_x,
                                            y=variable_y,
                                            z=variable_z,
                                            kern_x_on_s=kernel_x_on_s,
                                            kern_y_on_s=kernel_y_on_s,
                                            kern_z_on_s=kernel_z_on_s)

    # Regress "X on Z" and "Y on Z" once the dependence on S has been removed
    res_xz = gp_regression(res_z, res_x, kernel_x_on_z)
    res_yz = gp_regression(res_z, res_y, kernel_y_on_z)

    # Test for independence
    independent = independence_test_func(res_xz, res_yz)

    return independent
