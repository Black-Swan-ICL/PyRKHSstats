import numpy as np

import GPy

from PyRKHSstats.data_handling import to_numpy_array


# TODO document
# TODO test
# TODO, why input dim = 1 ? Make more general. Locations could well be multidim
def two_stage_gp_resit(variable_x, variable_y, variable_z, locations_s,
                       independence_test_func, kernel_x_on_s=GPy.kern.RBF,
                       kernel_y_on_s=GPy.kern.RBF, kernel_z_on_s=GPy.kern.RBF,
                       kernel_x_on_z=GPy.kern.RBF, kernel_y_on_z=GPy.kern.RBF):
    """Uses GP regression for non-i.i.d. conditional independence testing.

    The method was developed by S. R. Flaxman, D. B. Neill and A. J. Smola in
    [1]_.and assumes the noise models are additive throughout. Regression is
    (optionally) first carried out on a set of locations :math:`S` to eliminate
    potential temporal/spatial/network dependency between the variables
    :math:`X, Y, Z`. A second round of Gaussian Regression is performed to
    reduce conditional independence testing of :math:`X` and :math:`Y` given
    :math:`Z` to a problem of unconditional independence testing.

    Parameters
    ----------
    variable_x : array_like
    variable_y : array_like
    variable_z : array_like
    locations_s : array_like
    independence_test_func : callable
    kernel_x_on_s
    kernel_y_on_s
    kernel_z_on_s
    kernel_x_on_z
    kernel_y_on_z

    Returns
    -------
    bool
        Whether :math:`X` and :math:`Y` are conditionally independent given
        :math:`Z` (and :math:`S`).

    Notes
    -----
    .. [1] Flaxman, S. R., Neill, D. B. and Smola, A. J. "Gaussian Processes for
    Independence Tests with Non-iid Data in Causal Inference". *ACM Transactions
    on Intelligent Systems and Technology*, volume 7, no. 2, article 22,
    November 2015.

    """
    # TODO check same lengths for the variables

    def reformat(x):

        if len(x) == 2:
            return x
        else:
            return np.reshape(x, (x.shape[0], 1))

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

    variable_x = reformat(to_numpy_array(variable_x))
    variable_y = reformat(to_numpy_array(variable_y))
    variable_z = reformat(to_numpy_array(variable_z))
    locations_s = reformat(to_numpy_array(locations_s))

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
