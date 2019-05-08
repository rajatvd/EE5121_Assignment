"""Run this script to solve q1."""
import argparse
import matplotlib.pyplot as plt
import numpy as np
import cvxpy as cp
import scipy.io


# %%
def solve(y, rho=0.2, re_eps=0.01, re_iters=1):
    n = y.shape[0]
    w = np.ones(n - 1)
    w = w.reshape(-1, 1)

    diff1 = np.zeros((n - 1, n))
    diff1[:, 1:] = np.eye(n - 1)
    diff1[:, :-1] -= np.eye(n - 1)

    for i in range(re_iters):
        x = cp.Variable((n, 1))
        error = cp.pnorm(x - y, 2)
        v = diff1 @ x
        tv = cp.pnorm(cp.multiply(w, v), 1)

        obj = error + rho * tv
        prob = cp.Problem(cp.Minimize(obj))

        prob.solve(verbose=True)
        w = 1 / (np.abs(v.value) + re_eps)

    return x.value

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

    mat = scipy.io.loadmat('piecewise_constant_data.mat')
    y = mat['y']
    x_hat = solve(y, v.rho, v.re_eps, v.re_iters)

    plt.rcParams['font.size'] = 15
    plt.figure(figsize=(9, 9))
    plt.title(f"Total variation reconstruction with $\\rho = {v.rho:.5f}$")
    plt.plot(y, '-', alpha=0.5)
    plt.plot(x_hat)
    plt.grid()
    plt.legend([r"$y$", r"$\hat x$"])
    plt.text(1, 4, f"$\\epsilon = {v.re_eps}$ \niters $ = {v.re_iters}$")
    plt.show()
