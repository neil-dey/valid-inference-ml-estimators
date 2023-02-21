import numpy as np
from sklearn.linear_model import Lasso
from scipy.optimize import fsolve
from numpy.linalg import norm
import matplotlib.pyplot as plt
import warnings
np.random.seed(0)

"""
Computes the solution to the constrained optimization problem
    minimize ||Y - Xb||_2^2 subject to ||b||_1 <= t
in the case where n >= p

Achieved by QR decomposing X and using the closed form solution
for LASSO on orthogonal matrices. That is, we change variables to
    minimize ||Y - Qg||_2^2 subject to ||R^-1g||_1 <= t
since we use g = Rb to yield Xb = Qg (as X = QR)
"""
def _qr_lasso(X, Y, t):
    def pos(x):
        return 0 if x <= 0 else x
    Q, R = np.linalg.qr(X)

    gamma_OLS = np.linalg.lstsq(Q, Y, rcond=None)[0]

    Rinv = np.linalg.inv(R)

    # For orthogonal design, b_LASSO(j) = sign(b_OLS(j)) * (|b_OLS(j)| - n * lambda)^+
    lasso_gamma_solution = lambda l : np.array([np.sign(g) * pos(abs(g) - n*l[0]) for g in gamma_OLS])

    # The complementary slackness condition of KKT conditions: Sum[|b_LASSO|] = t
    l = fsolve(lambda l : norm(Rinv @ lasso_gamma_solution(l), ord = 1) - t, [0.001])[0]
    l = 0 if l < 0 else l

    gamma_lasso = lasso_gamma_solution([l])

    beta_lasso = Rinv @ np.array([np.sign(g) * pos(abs(g) - n*l) for g in gamma_OLS])

    return beta_lasso

"""
Computes the solution to the constrained optimization problem
    minimize ||Y - Xb||_2^2 subject to ||b||_1 <= t

Requires computing many LASSO solutions; this is inefficient if n >= p
"""
def _hd_lasso(X, Y, t):
    def lasso_solution(l):
        # lambda must be nonnegative; if lambda < 0, we need to make sure the provided parameter doesn't
        # solve solve the KKT condition, so just return a zero vector
        if l[0] < 0:
            return [0]
        return Lasso(alpha = l[0], max_iter=10000).fit(X, Y).coef_

    # The complementary slackness condition of KKT conditions: Sum[|b_LASSO|] = t
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message = "The iteration is not making good progress")
        l = fsolve(lambda l : norm(lasso_solution(l), ord = 1) - t, [0.000001])[0]

    # If the necessary lambda is 0, then the LASSO problem reduces to OLS
    if l == 0:
        return np.linalg.lstsq(X, Y, rcond=None)[0]

    return lasso_solution([l])


"""
Computes the solution to the constrained optimization problem
    minimize ||Y - Xb||_2^2 subject to ||b||_1 <= t
"""
def constrained_lasso(X, Y, t):
    n, p = X.shape

    if n >= p:
        return _qr_lasso(X, Y, t)
    else:
        return _hd_lasso(X, Y, t)

t = 10
p = 10
beta_0 = np.random.rand(p)*2 - 1
print(norm(beta_0, ord=1))
n = 1000

for alpha, color, marker in zip([0.05], ["blue"], ["."]):
    print(alpha)
    epsilon = 8*(norm(beta_0, ord=1) + t)**2/n * np.log(2/alpha)
    tps = np.linspace(1.0, 1.5, num=50)
    t_threshold = 0
    pls = []
    for tp in tps:
        pl = 0
        iters = 10000
        for _ in range(iters):
            X = np.random.rand(n, p)*2 - 1
            Y = X @ beta_0 + (np.random.rand(n)*2 - 1)

            lasso_t_coefs = constrained_lasso(X, Y, t)
            lasso_t2_coefs = constrained_lasso(X, Y, tp)

            if norm(Y - X @ lasso_t2_coefs) - norm(Y - X @ lasso_t_coefs)  < epsilon:
                pl += 1
        pl /= iters
        if(pl < 1-alpha):
            t_threshold = tp
        print(pl)
        pls.append(pl)

    plt.xlabel("t'")
    plt.ylabel("Pl")
    plt.axhline(1-alpha)
    plt.axvline(t_threshold)
    plt.scatter(tps, pls)
plt.show()
