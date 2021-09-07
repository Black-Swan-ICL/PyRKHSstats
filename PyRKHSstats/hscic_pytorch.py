import math
import torch
import numpy as np


def compute_mat_W(mat_K_Z, regularisation_constant):
    """
    Computes the matrix :math:`W`, as specified in the paper.

    Matrix :math:`W` is defined as follows :

    .. math::
        W := (K_Z + n \lambda I_n)^{-1}

    Parameters
    ----------
    mat_K_Z : array_like
        The kernel Gram matrix for the :math:`Z` variable :math:`K_Z`.
    regularisation_constant : float
        The regularisation constant.

    Returns
    -------
    array_like
        The matrix :math:`W` as defined in the paper.
    """
    torch_mat_K_Z = torch.from_numpy(mat_K_Z)
    n = torch_mat_K_Z.shape[0]
    mat_W = (
            torch_mat_K_Z +
            n * regularisation_constant * torch.diag(torch.ones(n))
    ).inverse().numpy()

    return mat_W


def compute_vec_k_Z_in_z(z, data_z, kernel_z):
    """
    Evaluates the function :math:`k_Z(z)`, as defined in the paper.

    Function :math:`k_Z(z)` is defined on :math:`\mathcal{Z}` by

    .. math::
        k_Z(z) = (k_Z(z_1, z), \dots , k_Z(z_n, z))^T

    Parameters
    ----------
    z : array_like
        The evaluation point :math:`z`.
    data_z : array_like
        The observations (data) in :math:`\mathcal{Z}` domain.
    kernel_z : KernelWrapper
        The reproducing kernel associated to the RKHS on domain
        :math:`\mathcal{Z}` .

    Returns
    -------
    array_like
        The vector valued function evaluated at :math:`z`, :math:`k_Z(z)`.
    """
    n = len(data_z)
    res = np.zeros((n, 1))
    for i in range(n):

        res[i, 0] = kernel_z.evaluate(data_z[i], z)

    return res


def compute_hscic(z, mat_K_X, mat_K_Y, hadamard_K_X_K_Y, mat_W, func_vec_k_Z):
    """
    Computes the empirical HSCIC, as defined in the paper.

    The empirical HSCIC is defined by

    .. math::
        \widehat{H}^2_{X, Y | Z}(z) = k_Z(z)^T W (K_X \odot K_Y) W^T k_Z(z) -
        2 k_Z(z)^T W ((K_X W^T k_Z(z)) \odot (K_Y W^T k_Z(z))) +
        (k_Z(z)^T W K_X W^T k_Z(z))(k_Z(z)^T W K_Y W^T k_Z(z))

    Parameters
    ----------
    z : array_like
        The evaluation point :math:`z`.
    mat_K_X : array_like
        The kernel Gram matrix for the :math:`X` variable :math:`K_X`.
    mat_K_Y : array_like
        The kernel Gram matrix for the :math:`Y` variable :math:`K_Y`.
    hadamard_K_X_K_Y : array_like
        The Hadamard product of :math:`K_X` and :math:`K_Y`.
    mat_W : array_like
        The matrix :math:`W`, as defined in the paper.
    func_vec_k_Z : callable
        The function :math:`k_Z(z)`, as defined in the paper.

    Returns
    -------
    float
        The empirical HSCIC evaluated at :math:`z`.
    """
    k_Z_in_z = func_vec_k_Z(z)

    # Moving to PyTorch
    torch_k_Z_in_z = torch.from_numpy(k_Z_in_z)
    torch_mat_W = torch.from_numpy(mat_W)
    torch_mat_K_X = torch.from_numpy(mat_K_X)
    torch_mat_K_Y = torch.from_numpy(mat_K_Y)
    torch_mat_hadamard_K_X_K_Y = torch.from_numpy(hadamard_K_X_K_Y)
    # TODO test whether it helps in very hgh dimensions #######################
    # Moving to GPU
    # torch_k_Z_in_z = torch_k_Z_in_z.cuda()
    # torch_mat_W = torch_mat_W.cuda()
    # torch_mat_K_X = torch_mat_K_X.cuda()
    # torch_mat_K_Y = torch_mat_K_Y.cuda()
    # torch_mat_hadamard_K_X_K_Y = torch_mat_hadamard_K_X_K_Y.cuda()

    # To avoid repeated matrix computations
    torch_k_Z_in_z_W = torch_k_Z_in_z.t() @ torch_mat_W
    torch_K_X_k_Z_in_z_W = torch_mat_K_X @ torch_k_Z_in_z_W.t()
    torch_K_Y_k_Z_in_z_W = torch_mat_K_Y @ torch_k_Z_in_z_W.t()

    term_1 = torch_k_Z_in_z_W @ torch_mat_hadamard_K_X_K_Y @ torch_k_Z_in_z_W.T
    term_2 = torch_k_Z_in_z_W @ (torch_K_X_k_Z_in_z_W * torch_K_Y_k_Z_in_z_W)
    term_3 = ((torch_k_Z_in_z_W @ torch_K_X_k_Z_in_z_W) *
              (torch_k_Z_in_z_W @ torch_K_Y_k_Z_in_z_W))

    res = math.sqrt((term_1 - 2 * term_2 + term_3)[0, 0])

    return res
