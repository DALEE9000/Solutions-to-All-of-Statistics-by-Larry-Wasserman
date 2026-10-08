# Coverage of the bootstrapped Pivotal interval for the skewness of X = e^Y, Y ~ N(0,1).
# Vectorized and seeded (no multiprocessing needed): every repetition and every bootstrap
# resample is one array operation.
import time

import numpy as np
from scipy.stats import skew
import matplotlib.pyplot as plt
import scienceplots

import sys; sys.path.insert(0, r"C:\Users\david\Documents\math")
from plotstyle import C, FG, BG, savefig

rng = np.random.default_rng(2024)

# Estimating coverage of the intervals
B = 50 # number of bootstrap estimates
R_repet = 1000 # number of repetitions (changing repetitions) 1000
R_sample = 100 # number of repetitions (changing sample size) 100
N = 1000 # number of samples 1000

# true skew of X = e^Y, Y ~ N(0,1): (e + 2) sqrt(e - 1), exactly
trueskew = (np.e + 2)*np.sqrt(np.e - 1)

def coverage(R, size):
    """Fraction of R repetitions whose interval, from `size` observations and B bootstrap
    resamples (with replacement), contains trueskew."""
    x = np.exp(rng.normal(0, 1, size=(R, size)))
    skewhat = skew(x, axis=1)
    idx = rng.integers(0, size, size=(R, B, size))
    boot = skew(x[np.arange(R)[:, None, None], idx], axis=2)   # (R, B) bootstrap skews
    lower = 2*skewhat - np.quantile(boot, 0.975, axis=1)
    upper = 2*skewhat - np.quantile(boot, 0.025, axis=1)
    return np.count_nonzero((trueskew > lower) & (trueskew < upper)) / R

# Pivotal interval (many repetitions): samples of 50, R = 1..R_repet
def iterationrepet(R):
    return coverage(R, 50)

# Pivotal interval (changing samples): R_sample repetitions, sample size N
def iterationsample(N):
    return coverage(R_sample, N)


if __name__ == "__main__":
    t1 = time.time()
    print("True Skew", trueskew)

    coverage_repet = [iterationrepet(R) for R in range(1, int(R_repet)+1)] # use for repeated trials
    coverage_sample = [iterationsample(n) for n in range(10, N+1)] # use for changing sampling

    print("Delta:", time.time() - t1)

    fig, axs = plt.subplots(1, 2, figsize=(9, 3), sharey=True)
    # graph for changing repetition
    axs[0].scatter(range(1,int(R_repet)+1), coverage_repet, marker='.', color=C.cyan, linewidths=3)
    axs[0].set_title('Coverage for the Bootstrapped Pivotal Interval', fontsize=10)
    axs[0].set_xlabel("$R$ iterations", fontsize=16)
    axs[0].minorticks_off()

    # graph for changing sample size
    axs[1].scatter(np.arange(10,N+1), coverage_sample, marker='.', color=C.cyan, linewidths=3)
    axs[1].set_title('Coverage for the Bootstrapped Pivotal Interval ($R = {}$)'.format(R_sample), fontsize=10)
    axs[1].set_xlabel("$N$ samples", fontsize=16)
    axs[1].minorticks_off()

    savefig(fig, 'chap8ex2pivotal.eps', bbox_inches=None)
