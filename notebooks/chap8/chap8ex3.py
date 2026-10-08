# Coverage of three bootstrapped intervals for theta = (q_.75 - q_.25)/1.34 of a t_3 sample.
# Vectorized and seeded (no multiprocessing needed).
import time

import numpy as np
import matplotlib.pyplot as plt
import scienceplots

import sys; sys.path.insert(0, r"C:\Users\david\Documents\math")
from plotstyle import C, FG, BG, savefig

rng = np.random.default_rng(2024)

# Estimating coverage of the intervals
B = 25 # number of bootstrap samples
R = 1000 # number of repetitions
N = 25 # size of t-distribution sample (n = 25 in the exercise)
n = N # size of each bootstrap sample: resample the whole sample
q_75 = 0.765
q_25 = -0.765
truetheta = (q_75 - q_25) / 1.34

print('truetheta', truetheta)

def theta(x, axis=-1):
    q75, q25 = np.quantile(x, [0.75, 0.25], axis=axis)
    return (q75 - q25) / 1.34

def draws(R):
    """R repetitions: N t_3 points each, their theta-hat, and B bootstrap thetas
    from resamples of size n (with replacement)."""
    tpoints = rng.standard_t(3, size=(R, N))
    idx = rng.integers(0, N, size=(R, B, n))
    boot = theta(tpoints[np.arange(R)[:, None, None], idx], axis=2)   # (R, B)
    return theta(tpoints, axis=1), boot

def covered(lower, upper):
    return np.count_nonzero((truetheta > lower) & (truetheta < upper)) / len(lower)

# Normal interval
def normal(R):
    thetahat, boot = draws(R)
    seboot = boot.std(axis=1)
    return covered(thetahat - 1.96*seboot, thetahat + 1.96*seboot)

# Pivotal interval
def pivotal(R):
    thetahat, boot = draws(R)
    return covered(2*thetahat - np.quantile(boot, 0.975, axis=1), 2*thetahat - np.quantile(boot, 0.025, axis=1))

# Percentile interval
def percentile(R):
    thetahat, boot = draws(R)
    return covered(np.quantile(boot, 0.025, axis=1), np.quantile(boot, 0.975, axis=1))


if __name__ == "__main__":
    t1 = time.time()

    coverage_normal = [normal(r) for r in range(1, R+1)]
    coverage_pivotal = [pivotal(r) for r in range(1, R+1)]
    coverage_percentile = [percentile(r) for r in range(1, R+1)]

    print("Delta:", time.time() - t1)

    fig, axs = plt.subplots(3, 1, figsize=(6, 6), sharex=True)
    # normal
    axs[0].scatter(range(1,int(R)+1), coverage_normal, marker='.', color=C.cyan, linewidths=3)
    axs[0].set_title('Coverage for the Bootstrapped Normal Interval', fontsize=12)
    axs[0].minorticks_off()

    # pivotal
    axs[1].scatter(range(1,int(R)+1), coverage_pivotal, marker='.', color=C.cyan, linewidths=3)
    axs[1].set_title('Coverage for the Bootstrapped Pivotal Interval', fontsize=12)
    axs[1].minorticks_off()

    # percentile
    axs[2].scatter(range(1,int(R)+1), coverage_percentile, marker='.', color=C.cyan, linewidths=3)
    axs[2].set_title('Coverage for the Bootstrapped Percentile Interval', fontsize=12)
    axs[2].set_xlabel("$R$ iterations", fontsize=16)
    axs[2].minorticks_off()

    savefig(fig, 'chap8ex3.eps', bbox_inches=None)
