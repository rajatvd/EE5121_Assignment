"""Run this script to solve q3."""
import matplotlib.pyplot as plt
import numpy as np
import cvxpy as cp
import scipy.io


# %%
def solve(X):
    """Solve question 3.

    Uses SDP relaxation and trace approximation for rank.

    Parameters
    ----------
    X : numpy array
        Problem data with missing entries labeled with 0.

    Returns
    -------
    numpy array
        Completed matrix.

    """
    # %%
    m, n = X.shape
    Xv = cp.Variable((m, n))
    Y = cp.Variable((m, m))
    Z = cp.Variable((n, n))

    ii = np.where(X != 0)
    obj = cp.trace(Y) + cp.trace(Z)

    S = cp.bmat([
        [Y, Xv],
        [Xv.T, Z]
    ])

    constraints = [
        Y >> 0,
        Z >> 0,
        S >> 0,
        Xv[ii] == X[ii],
    ]

    prob = cp.Problem(cp.Minimize(obj), constraints)
    # %%
    prob.solve(solver='SCS', verbose=True)
    return Xv.value


# %%
if __name__ == '__main__':

    # load data
    mat = scipy.io.loadmat('Ratings.mat')
    X = mat['X']

    Xc = solve(X)
    print(f"Rank of completed matrix = {np.linalg.matrix_rank(Xc)}")

    plt.figure(figsize=(5, 6))
    plt.imshow(X)
    plt.show()

    plt.figure(figsize=(5, 6))
    plt.imshow(Xc)
    plt.show()

    plt.figure(figsize=(5, 6))
    plt.imshow(Xc - X)
    plt.show()
