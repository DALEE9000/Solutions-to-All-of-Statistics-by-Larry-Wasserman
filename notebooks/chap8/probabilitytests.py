from multiprocessing import Pool
import time
import random
import os

import numpy as np
from scipy.stats import norm
from scipy.stats import skew 

def expnormal(samples):
    ytrueskew = np.random.normal(0,1,int(samples))
    xtrueskew = np.exp(ytrueskew)
    return skew(xtrueskew);

if __name__ == "__main__":
    samples = 1e8
    nbr_parallel_blocks = 6
    pool = Pool(processes=nbr_parallel_blocks)
    nbr_samples_per_worker = samples / nbr_parallel_blocks
    print("Making {:,} samples per {} worker". format(nbr_samples_per_worker,
                                                      nbr_parallel_blocks))
    nbr_trials_per_process = [nbr_samples_per_worker] * nbr_parallel_blocks
    t1 = time.time()
    myfunction = pool.map(expnormal, nbr_trials_per_process)
    skewval = np.mean(myfunction)
    print("Skew", skewval)
    print("Delta:", time.time() - t1)