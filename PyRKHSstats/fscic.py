"""
This module implements the Finite Set Conditional Independence Criterion
(FSCIC) developed in 'Kernel Measures of Conditional Independence' (J. Park,
2019 ; MSc thesis, Seminar for Statistics, Department of Mathematics, ETHZ)
which will be referred to as 'the paper' in this module. Numpy is used.
"""
import numpy as np


# TODO test
def compute_fscic(z, mat_K_X_tilda, mat_K_Y_tilda, mat_W, func_vec_k_Z):
    """
    Computes the empirical FSCIC, as defined in the paper.

    The empirical FSCIC is defined by

    ..math::
        \widehat{G}^2_{X, Y | Z}(z) = (1/J) V(z)^T V(z),

    where :math:`J` is the number of test locations and :math:`V(z)` is
    given by

    ..math::
        V(z) = (\tilde{K}_X \odot \tilde{K}_Y)^T W^T k_Z -
        ((\tilde{K}_X W^T k_Z) \odot (\tilde{K}_Y W^T k_Z)).

    Parameters
    ----------
    z : array_like
        The evaluation point :math:`z`.
    mat_K_X_tilda : array_like
        The mixed kernel matrix observations - test locations for the :math:`X`
        space :math:`[k_X(x_i, v_j)]`.
    mat_K_Y_tilda : array_like
        The mixed kernel matrix observations - test locations for the :math:`Y`
        space :math:`[k_Y(y_i, v_j)]`.
    mat_W : array_like
        The matrix :math:`W`, as defined in the paper.
    func_vec_k_Z : callable
        The function :math:`k_Z(z)`, as defined in the paper.

    Returns
    -------
    float
        The empirical FSCIC evaluated at :math:`z`.
    """

    nb_test_locations = mat_K_X_tilda.shape[1]

    k_Z_in_z = func_vec_k_Z(z)

    mat_W_k_Z_in_z = mat_W.T @ k_Z_in_z

    term_1 = np.multiply(mat_K_X_tilda, mat_K_Y_tilda).T @ mat_W_k_Z_in_z
    term_2 = np.multiply(mat_K_X_tilda @ mat_W_k_Z_in_z,
                         mat_K_Y_tilda @ mat_W_k_Z_in_z)
    aux = term_1 - term_2

    res = (1 / nb_test_locations) * aux.T @ aux

    return res
