"""Run this script to solve q1."""
import argparse
import matplotlib.pyplot as plt
import numpy as np
import cvxpy as cp
import scipy.io


# %%
def solve(y, rho=0.2):
    """Solve question 1 (total variation minimization).

    Uses the SOCP approach formulated in the report.

    Parameters
    ----------
    y : numpy array
        Input data.
    rho : float
        Weight given to l1 norm of TV (the default is 0.2).

    Returns
    -------
    numpy array
        Reconstructed piecewise constant signal.

    """
    n = y.shape[0]

    # create TV matrix (called A in the report)
    diff1 = np.zeros((n - 1, n))
    diff1[:, 1:] = np.eye(n - 1)
    diff1[:, :-1] -= np.eye(n - 1)

    # create x variable - the reconstruction
    x = cp.Variable((n, 1))
    error = cp.pnorm(x - y, 2)  # l2 error
    tv = diff1 @ x  # find the first difference

    # auxiliary variables for SOCP formulation
    t = cp.Variable((1, 1))
    u = cp.Variable((n - 1, 1))

    # second order cone constraint for l2 error
    constraints = [error <= t]

    # 2*(n-1) linear constraints for l1 norm
    l1_constraints1 = [tv[i] <= u[i] for i in range(n - 1)]
    l1_constraints2 = [-u[i] <= tv[i] for i in range(n - 1)]

    constraints.extend(l1_constraints1)
    constraints.extend(l1_constraints2)

    # objective function in epigraph form
    obj = t + cp.sum(u)

    # solve the problem
    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve(verbose=True)

    # return reconstructed signal value
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

    v = parser.parse_args()
    print(v)

    # load data
    mat = scipy.io.loadmat('piecewise_constant_data.mat')
    y = mat['y']

    # reconstruct signal
    x_hat = solve(y, v.rho)

    # output error
    e = np.linalg.norm(y - x_hat)
    print(f"Optimal value of error e = {e}")

    # number of jumps
    js = jumps(x_hat)
    print(f"Number of jumps = {js}")

    s = f"Optimal value of error $e = {e:.4f}$\n"
    s += f"Number of jumps $ = {js}$\n"

    # plot stuff
    plt.rcParams['font.size'] = 15
    plt.figure(figsize=(9, 9))
    plt.title(
        f"Total variation reconstruction with $\\rho = {v.rho:.5f}$ (SOCP)")
    plt.plot(y, '-', alpha=0.5)
    plt.plot(x_hat)
    plt.grid()
    plt.legend([r"$y$", r"$\hat x$"])
    plt.text(1, 3, s)
    plt.show()
