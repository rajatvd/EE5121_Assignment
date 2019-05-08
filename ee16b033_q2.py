"""Run this script to solve q2."""
import numpy as np
import cvxpy as cp


# %%
def solve(A, c_max, p, p_disc, q):
    """Solve question 2 (resource allocation).

    Returns
    -------
        activity levels, revenue for each activity

    """
    n = p.shape[0]

    x = cp.Variable((n, 1))
    t = cp.Variable((n, 1))

    # epigraph linear objective
    obj = cp.sum(t)

    # affine resource constraint and non-negative activity constraint
    constraints = [A @ x <= c_max, x >= 0]

    # splitting max function into two affine constraints (epigraph form)
    c1 = [-p[j] * x[j] <= t[j] for j in range(n)]
    c2 = [-p[j] * q[j] - p_disc[j] * (x[j] - q[j]) <= t[j] for j in range(n)]

    constraints.extend(c1)
    constraints.extend(c2)

    prob = cp.Problem(cp.Minimize(obj), constraints)

    prob.solve(verbose=True, solver='ECOS')
    # print(t.value)
    # print(np.max(np.array([-p*x.value, -p*q -p_disc*(x.value-q)]), axis=0))
    return x.value, -t.value


# %%
if __name__ == '__main__':

    # %% setup problem data
    A = np.array([
        [1, 2, 0, 1],
        [0, 0, 3, 1],
        [0, 3, 1, 1],
        [2, 1, 2, 5],
        [1, 0, 3, 2],
    ])
    c_max = 100 * np.ones((5, 1))
    p = np.array([3, 2, 7, 6]).reshape(-1, 1)
    p_disc = np.array([2, 1, 4, 2]).reshape(-1, 1)
    q = np.array([4, 10, 5, 10]).reshape(-1, 1)
    # %%

    # solve problem
    x, r = solve(A, c_max, p, p_disc, q)

    # output stuff
    print(f"Optimal activity levels: \n{x}")
    print(f"Revenue of each activity: \n{r}")
    print(f"Total revenue = {r.sum()}")
    print(f"Average price of each activity: \n{r/x}")
