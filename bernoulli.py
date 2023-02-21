import numpy as np
from scipy.stats import binom
from scipy.stats import norm as normal
from math import ceil
import matplotlib.pyplot as plt

np.random.seed(0)

def ucp_prob(m, eps):
    precision = 10000
    min_prob = 1
    for k in range(precision + 1):
        p = k/precision
        ucp = binom.cdf(int(m*(p+eps)), m, p) - binom.cdf(ceil(m*(p-eps)), m, p) + binom.pmf(ceil(m*(p-eps)), m, p)
        # An alternate, approximately correct ucp for large samples:
        #ucp = normal.cdf(m*(p+eps), loc = m*p, scale = (m*p*(1-p))**0.5) - normal.cdf(m*(p-eps), loc = m*p, scale = (m*p*(1-p))**0.5)
        if ucp < min_prob:
            min_prob = ucp
    return min_prob


def get_tolerance(m, alpha):
    precision = 10000
    # Find the lowest epsilon allowing confidence >= 1-alpha via binary search
    epsilons = [k/precision for k in range(precision + 1)]
    lower = 0
    upper = len(epsilons) - 1
    while lower < upper - 1:
        conf = ucp_prob(m, epsilons[(upper + lower)//2])
        if conf >= 1 - alpha:
            upper = (upper + lower)//2
        else:
            lower = (upper + lower)//2
    if ucp_prob(m, epsilons[lower]) >= 1 - alpha:
        return epsilons[lower]
    return epsilons[upper]

def get_sample_size(eps, alpha):
    # Find the smallest m allowing confidence >= 1-alpha via binary search
    ms = [m for m in range(1000000)]
    lower = 0
    upper = len(ms) - 1
    while lower < upper - 1:
        conf = ucp_prob(ms[(upper+lower)//2], eps)
        if conf >= 1 - alpha:
            upper = (upper + lower)//2
        else:
            lower = (upper + lower)//2
    if ucp_prob(m[lower], eps) >= 1 - alpha:
        return ms[lower]
    return ms[upper]


