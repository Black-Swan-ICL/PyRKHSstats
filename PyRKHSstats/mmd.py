"""
This module contains the code to compute the Maximum Mean Discrepancy (MMD)
presented in 'A Kernel Two-Sample Test', A. Gretton, K. M. Borgwardt, M. J.
Rasch, B. Sch\"{o}lkopf and A. Smola (Journal of Machine Learning Research #13,
2012) which will be referred to as 'the paper' in the module.
"""


# TODO code is very slow...
def compute_unbiased_mmd(data_x, data_y, kernel):
    """

    Parameters
    ----------
    data_x
    data_y
    kernel : KernelWrapper

    Returns
    -------

    """

    nx = data_x.shape[0]
    ny = data_y.shape[0]

    if nx == ny:
        # TODO this could be mde more efficient by computing the kernel Gram
        #  matrices and using them appropriately

        unbiased_mmd = 0
        for i in range(nx):
            for j in range(nx):

                if i == j:
                    pass
                else:
                    unbiased_mmd += (
                            kernel.evaluate(data_x[i], data_x[j]) +
                            kernel.evaluate(data_y[i], data_y[j]) -
                            kernel.evaluate(data_x[i], data_y[j]) -
                            kernel.evaluate(data_x[j], data_y[i])
                    )
        unbiased_mmd /= (nx * (nx - 1))

        return unbiased_mmd

    else:

        mat_Kx = kernel.compute_kernelised_gram_matrix(data_x)
        mat_Ky = kernel.compute_kernelised_gram_matrix(data_y)
        mat_Kxy = kernel.compute_rectangular_kernel_matrix(data_x, data_y)

        for i in range(nx):
            mat_Kx[i, i] = 0

        for i in range(ny):
            mat_Ky[i, i] = 0

        unbiased_mmd = (
                mat_Kx.sum() / (nx * (nx - 1)) +
                mat_Ky.sum() / (ny * (ny - 1)) -
                2 * mat_Kxy.sum() / (nx * ny)
        )

        return unbiased_mmd
