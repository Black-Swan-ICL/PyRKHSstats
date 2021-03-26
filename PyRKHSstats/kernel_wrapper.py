import numpy as np
from sklearn.gaussian_process.kernels import Kernel


class KernelWrapper:

    def __init__(self, kernel):
        """
        A class that abstracts the representation of a kernel whether it is
        actually a kernel implemented in scikit-learn or a user-defined kernel
        the implementation of which is not vectorised.

        Parameters
        ----------
        kernel : Kernel or callable
            The kernel function.
        """
        self.kernel = kernel
        self.is_sklearn_kernel = isinstance(kernel, Kernel)

    def evaluate(self, x, y):
        """
        Evaluates the kernel at (x, y).

        Parameters
        ----------
        x : array_like
            The first argument.
        y : array_like
            The second argument.

        Returns
        -------
        float
            The kernel evaluated at (x, y).
        """
        if not self.is_sklearn_kernel:

            res = self.kernel(x, y)

        else:

            # x and y must be 2D-arrays for scikit-learn kernels
            x = np.asarray(x).reshape(-1, 1)
            y = np.asarray(y).reshape(-1, 1)

            res = self.kernel.__call__(x, y).flatten()[0]

        return res

    def compute_kernelised_gram_matrix(self, data):
        """
        Given observations data, returns the kernelised Gram matrix
        (kernel(data[i], data[j]))_{i, j}.

        Parameters
        ----------
        data : array_like
            The observations.

        Returns
        -------
        array_like
            The kernelised Gram matrix.
        """
        if not self.is_sklearn_kernel:

            m = data.shape[0]
            kernelised_gram_matrix = np.empty((m, m), dtype=np.double)

            for i in range(m):
                for j in range(m):
                    kernelised_gram_matrix[i, j] = self.kernel(data[i], data[j])

        else:

            kernelised_gram_matrix = self.kernel.__call__(data, data)

        return kernelised_gram_matrix
