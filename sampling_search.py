from pcfg import *
import random
from math import sqrt
from heapq import heappush, heappop
import copy
import timeit
import itertools

# -----------------------------------
# ------ SAMPLING ALGORITHM ---------
# -----------------------------------

def sample_rule(cumulative):
    low, high = 0, len(cumulative)-1
    threshold = random.random()
    
    while low <= high:
        mid = (high+low)//2
        if cumulative[mid] < threshold:
            low = mid+1
        else:
            high = mid-1

    res = mid+1 if cumulative[mid] < threshold else mid
    return res
            

        
def sample_derivation(start: str , G: PCFG, cumulatives: dict):
    f, symbols, w = G.rules[start][sample_rule(cumulatives[start])]
    args_f = []
    for S in symbols:
        args_f+=[sample_derivation(S, G, cumulatives)]
    return [f,args_f]

def sampling(G: PCFG):
    '''
    A generator that samples terms according to the PCFG G
    '''
    # pre-processing to compute the cumulative distribution for any derivation rule from a given symbol
    cumulatives = G.cumulatives
    S0 = G.start
    while True:
        yield sample_derivation(S0, G, cumulatives)

def sampling_sqrt(G: PCFG):
    '''
    A generator that samples terms according to the PCFG G
    '''
    # pre-processing to compute the cumulative distribution for any derivation rule from a given symbol
    SQRT = sqrt_PCFG(G)
    cumulatives = SQRT.cumulatives
    S0 = SQRT.start
    while True:
        yield sample_derivation(S0, SQRT, cumulatives)
