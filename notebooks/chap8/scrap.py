#import multiprocessing
import time
from time import sleep
#from multiprocessing import Process
#from multiprocessing import current_process
#from multiprocessing import parent_process
#from multiprocessing import active_children
from multiprocessing import Pool

import numpy as np
import random
from random import choices 
#from scipy.stats import norm
from scipy.stats import skew 
#import matplotlib.pyplot as plt
# import scienceplots
#from scipy.stats import pearsonr

import numpy as np

# Estimating coverage of the intervals
B = 10
N = 100
#x = np.zeros(50)
samples = 1e8

# true skew
def expnormal(samples):
    np.random.seed()
    ytrueskew = np.random.normal(0,1,int(samples))
    xtrueskew = np.exp(ytrueskew)
    return skew(xtrueskew);

trueskew = expnormal(samples)

# Normal interval
def iteration(N):
    coverage = []
    for i in range(0,int(N)):
        np.random.seed()
        y = np.random.normal(0,1,50)
        x = np.exp(y)
        skewhat = skew(x)
        # Bootstrap
        def bootstrap(B):
            bootstrapsamples = []
            for i in range(0,int(B)):
                random.seed()
                bootstrap = choices(x, k=len(x))
                bootstrap = skew(bootstrap)
                bootstrapsamples.append(bootstrap)
            return bootstrapsamples;
        seboot = np.std(bootstrap(B))
        coverage.append((trueskew > (skewhat - 1.96*seboot)) & (trueskew < (skewhat + 1.96*seboot)))
    return coverage.count(True);

if __name__ == "__main__":
    # number of processes
    num_parallel_blocks = 6
    pool = Pool(processes=num_parallel_blocks)
    # number of expnormal per worker
    num_expnormal_per_worker = samples / num_parallel_blocks
    print("Making {:,} expnormal per {} workers". format(num_expnormal_per_worker,
                                                      num_parallel_blocks))
    expnormal_trials_per_process = [num_expnormal_per_worker] * num_parallel_blocks
    # number of bootstrap per worker
    #num_bootstrap_per_worker = B / num_parallel_blocks
    #print("Making {:,} bootstrap per {} workers". format(num_bootstrap_per_worker,
    #                                                  num_parallel_blocks))
    #bootstrap_trials_per_process = [num_bootstrap_per_worker] * num_parallel_blocks
    # number of iteration per worker
    num_iteration_per_worker = N / num_parallel_blocks
    print("Making {:,} iteration per {} workers". format(num_iteration_per_worker,
                                                      num_parallel_blocks))
    iteration_trials_per_process = [num_iteration_per_worker] * num_parallel_blocks
    # pooling functions
    t1 = time.time()
    expnormal_func = pool.map(expnormal, expnormal_trials_per_process)
    trueskew = np.mean(expnormal_func)
    print("True Skew", trueskew)
    #bootstrap_func = pool.map(bootstrap, bootstrap_trials_per_process)
    iteration_func = pool.map(iteration, iteration_trials_per_process)

    coverage = sum(iteration_func) / N
    print("Coverage:", coverage)
    print("Delta:", time.time() - t1)