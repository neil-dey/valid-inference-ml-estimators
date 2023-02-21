from bernoulli import *

p = 0.499
alpha = 0.05
gamma = alpha/2

if p < 0.5:
    A = 0
elif p > 0.5:
    A = 1
else:
    print("p = 0.5 requires that A = {0, 1} which necessarily has plausibility 1")
    exit()

sample_sizes = [i for i in range(1000000, 2000000)]
lower = 0
upper = len(sample_sizes) - 1
while lower < upper - 1:
    print(sample_sizes[lower], sample_sizes[upper])
    eps = get_tolerance(sample_sizes[(lower+upper)//2], alpha - gamma)
    if (1-2*eps)/2 <= p and p <= (1+2*eps)/2:
        lower = (upper + lower)//2
    else:
        upper = (upper + lower)//2
print("Sample size necessary for validity:", sample_sizes[upper])

ms = [int(10**(i/4)) for i in range(1, 30)]
coverages = []
print("(m, tolerance, confidence, SE)")
for m in ms:
    eps = get_tolerance(m, alpha - gamma)
    """
    if (1-2*eps)/2 <= p and p <= (1+2*eps)/2:
        print("Warning: sample size " + str(m) + " too small for validity")
    """
    conf = 0
    iterations = 1000
    for _ in range(iterations):
        sample = binom.rvs(m, p)
        B = ceil((4*gamma+3)*np.log(1/gamma)/(6*gamma**2))
        pl_hat = 0
        for i in range(B):
            subsample = binom.rvs(m, sample/m)

            empirical_risk_0 = subsample/m
            empirical_risk_1 = (m - subsample)/m
            inf_emp_risk = min([empirical_risk_0, empirical_risk_1])

            erm = []
            if empirical_risk_0 <= inf_emp_risk + 2*eps:
                erm.append(0)
            if empirical_risk_1 <= inf_emp_risk + 2*eps:
                erm.append(1)
            if A in erm:
                pl_hat += 1
        pl_hat /= B
        if pl_hat >= 1-alpha:
            conf += 1
    conf = conf/iterations
    print(m, 2*eps, conf, (conf*(1-conf)/iterations)**0.5)
    coverages.append(conf)

plt.scatter(ms[4:], coverages[4:])
plt.xscale('log')
plt.xlabel('Sample Size')
plt.ylabel('Coverage')
plt.axhline(y=0.95)
plt.axvline(x=1549987)
plt.savefig("bernoulli_bootstrap_coverage.png")
