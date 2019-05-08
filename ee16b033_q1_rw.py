"""Run this script to solve q1."""
import argparse
import matplotlib.pyplot as plt
import numpy as np
import cvxpy as cp
import scipy.io


# %%
def solve(y, rho=0.2, re_eps=0.01, re_iters=1):
    """Solve question 1 (total variation minimization).

    Uses reweighted-l1 minimization approach to enhance sparsity.

    Reference:
    Emmanuel J Candes, Michael B Wakin, and Stephen P Boyd. Enhancing sparsity
    by reweighted l1 minimization. Journal of Fourier analysis and applications
    14(5-6):877–905, 2008.

    Parameters
    ----------
    y : numpy array
        Input data.
    rho : float
        Weight given to l1 norm of TV (the default is 0.2).
    re_eps : float
        Value of epsilon used in reweighted l1 (the default is 0.01).
    re_iters : int
        Number of reweighting iterations (the default is 1).
        Note that setting this to 1 reduces to normal l1 minimization.
        (This can be used conveniently for comparing the two methods)

    Returns
    -------
    numpy array
        Reconstructed piecewise constant signal.

    """
    n = y.shape[0]

    # initialize weights
    w = np.ones(n - 1)
    w = w.reshape(-1, 1)

    # create TV matrix (called A in the report)
    diff1 = np.zeros((n - 1, n))
    diff1[:, 1:] = np.eye(n - 1)
    diff1[:, :-1] -= np.eye(n - 1)

    for i in range(re_iters):

        x = cp.Variable((n, 1))  # reconstruction variable
        error = cp.pnorm(x - y, 2)  # l2 loss
        v = diff1 @ x  # first difference vector

        # weighted l1 norm (or weighted tv)
        tv = cp.pnorm(cp.multiply(w, v), 1)

        # unconstrained objective which is weighted sum of error and tv norm
        obj = error + rho * tv
        prob = cp.Problem(cp.Minimize(obj))

        # solve weighted problem
        prob.solve(verbose=True)

        # update weights
        w = 1 / (np.abs(v.value) + re_eps)

    return x.value


# %%
def jumps(x, thresh=1e-4):
    """Find number of jumps by thresholding absolute first difference.

    Parameters
    ----------
    x : numpy array
        Signal to count jumps in.
    thresh : float
        Threshold value (the default is 1e-4).

    Returns
    -------
    int
        Number of jumps in x.

    """
    n = x.shape[0]
    diff1 = np.zeros((n - 1, n))
    diff1[:, 1:] = np.eye(n - 1)
    diff1[:, :-1] -= np.eye(n - 1)

    tv = diff1 @ x
    return len(np.where(abs(tv) > thresh)[0])


# %%
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--rho', nargs='?', default=0.2, type=float,
                        help='weight given to reweighted l1 norm')

    parser.add_argument('--re_eps', nargs='?', default=0.01, type=float,
                        help='epsilon in reweighted l1 norm')

    parser.add_argument('--re_iters', nargs='?', default=1, type=int,
                        help='number of reweighted l1 iterations')

    v = parser.parse_args()
    print(v)

    # load data
    mat = scipy.io.loadmat('piecewise_constant_data.mat')
    y = mat['y']

    # reconstruct signal
    x_hat = solve(y, v.rho, v.re_eps, v.re_iters)

    # output stuff and plots
    e = np.linalg.norm(y - x_hat)
    print(f"Optimal value of error e = {e}")

    js = jumps(x_hat)
    print(f"Number of jumps = {js}")

    s = f"$\\epsilon = {v.re_eps}$\n"
    s += f"iters $ = {v.re_iters}$\n"
    s += f"Optimal value of error $e = {e:.4f}$\n"
    s += f"Number of jumps $ = {js}$\n"

    plt.rcParams['font.size'] = 15
    plt.figure(figsize=(9, 9))
    plt.title(f"Total variation reconstruction with $\\rho = {v.rho:.5f}$")
    plt.plot(y, '-', alpha=0.5)
    plt.plot(x_hat)
    plt.grid()
    plt.legend([r"$y$", r"$\hat x$"])
    plt.text(1, 3, s)
    plt.show()
