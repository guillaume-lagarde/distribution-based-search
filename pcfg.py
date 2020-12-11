import random
from math import sqrt
from heapq import heappush, heappop
import copy
import timeit
import itertools as it
import numpy as np

class PCFG:
    '''
    Object that represents a probabilistic context-free grammar
    start: a start symbol
    rules: a dictionary of type {'V': [['F',(S1,S2,S3,...), w], ['G', ...], ...]} where the list rules['V'] is supposed to be ordered with respect to the w, and (S1,S2,..) is interpreted as the argument of 'F' and w is the weight of the derivation V -> F(S1,S2,..)
    '''
    def __init__(self, start: str, rules: dict):
        self.start = start
        self.rules = rules
        self.proba = {}
        self.arities = {}
        self.clean()
        for S in self.rules:
            for l in self.rules[S]:
                self.arities[l[0]] = l[1]
                self.proba[l[0]] = l[2]
        self.cumulatives = {S: [sum([self.rules[S][j][2] for j in range(i+1)]) for i in range(len(self.rules[S]))] for S in self.rules}

    def restart(self):
        self.proba = {}
        self.arities = {}
        for S in self.rules:
            for l in self.rules[S]:
                self.arities[l[0]] = l[1]
                self.proba[l[0]] = l[2]
        self.cumulatives = {S: [sum([self.rules[S][j][2] for j in range(i+1)]) for i in range(len(self.rules[S]))] for S in self.rules}
        
        
    def print(self):
        for S in self.rules:
            print('#\n', S)
            for e in self.rules[S]:
                print('   ', e)
        print("\n \n arities:", self.arities)


    def clean(self):
        '''
        remove unreachable symbols
        '''
        seen = set()
        self.collect(self.start, seen) # collect unreachable symbols
        for S in (set(self.rules) - seen):
            del self.rules[S]
        
    def collect(self, X, seen):
        seen.add(X)
        for f, args,w in self.rules[X]:
            for a in (set(args) - seen):
                self.collect(a, seen)


# -----------------------------------
# ------------ EXAMPLES -------------
# -----------------------------------


rules = {
    'S0': [['f1',['S0','S1'], 0.7],
           ['f2', ['S1'], 0.3]],
    
    'S1': [['g',['S2'],1]],
    
    'S2':[['x',[],0.9],
          ['y',[],0.1]]
}

rules = {
    'S0': [['f',['S0','S1'], 0.7],
           ['g', ['S1'], 0.3]],
    
    'S1': [['h',['S2'],0.5],
           ['i',['S0','S2'],0.5]],
    
    'S2':[['x',[],0.6],
          ['y',[],0.4]]
}

# rules = {
#     'S0': [['f1',['S0','S1'], 0.8],
#            ['f2', ['S1'], 0.2]],
#     'S1': [['blabla',[],0.6], ['blibli',[],0.4]]
# }

G = PCFG('S0', rules)


                
# -----------------------------------
# ---------- USEFUL TOOLS -----------
# -----------------------------------


def get_value(t, i):
    '''
    get the value in term t given by the tuple of indice i
    '''
    nt = copy.deepcopy(t)
    nt = t
    for i in i:
        nt = nt[i]
    return nt

def set_value(t, i, v):
    '''
    set the value in term t at position i (a tuple) to v
    '''

    ref = t
    for j in range(len(i)-1):
        ref = ref[i[j]]
    ref[i[-1]] = v


def set_max_tuple(G: PCFG, X, seen, dictionary):
    '''
    fill the given dictionary with, for each symbol X the pair (max tuple from X, max proba from X)
    '''
    if X in dictionary: return

    seen.add(X)

    for f, args, w in G.rules[X]:
        for A in args:
            if A not in seen:
                set_max_tuple(G, A, seen, dictionary)
    max_t = -1
    max_proba = -1
    for f, args, w in G.rules[X]:
        weight = w
        not_in = False
        for a in args:
            if a not in dictionary:
                not_in = True
                break
            weight*=dictionary[a][1]
        if not_in:continue
        if weight > max_proba:
            max_t = [f, [dictionary[a][0] for a in args]]
            max_proba = weight
    dictionary[X] = (max_t, max_proba)



def compute_Z(G):
    '''
    take a WCFG and compute the partition functions Z as a dictionary {X: Z^X}
    '''
    Z = {X: 1 for X in G.rules}
    for i in range(1000):
        for X in G.rules:
            s = 0
            for f, args, w in G.rules[X]:
                prod = w
                for symbol in args:
                    prod*=Z[symbol]
                s+=prod
            Z[X] = s
        return Z

    
def alpha_PCFG(G: PCFG, power = 1/2, threshold = 1000000):
    '''
    Output a PCFG that is G^alpha. If Z > threshold, return -1
    '''

    start = G.start
    partition_function = {X: 1 for X in G.rules}

    G = copy.deepcopy(G)
    for X in G.rules:
        for i in range(len(G.rules[X])):
            G.rules[X][i][2] = G.rules[X][i][2]**(power)
    
    for i in range(100):
        for X in G.rules:
            s = 0
            for f, args, w in G.rules[X]:
                prod = w
                for symbol in args:
                    prod*=partition_function[symbol]
                s+=prod
            partition_function[X] = s
    # print(partition_function[start])
    if partition_function[start] > threshold:
        print("sum sqrt(G) probably divergent")
        return -1
        
    r = copy.deepcopy(G.rules)
    for X in r:
        for i in range(len(r[X])):
            for s in r[X][i][1]:
                r[X][i][2]*=partition_function[s]
            r[X][i][2]*=(1/partition_function[X])
            
    return PCFG(start,r)

def sqrt_PCFG(G: PCFG, threshold = 20):
    return alpha_PCFG(G, 1/2)


def find_best_alpha(G: PCFG, threshold = 20, accuracy = 0.01 ):
    alpha_inf = 1/2
    alpha_sup = 1
    res = -1

    # first check if 1/2 is ok
    G2 = sqrt_PCFG(G, threshold = threshold)
    if G2 != -1: return G2, 1/2
    
    while alpha_sup - alpha_inf > accuracy:
        alpha = alpha_inf+(alpha_sup- alpha_inf)/2
        G2 = alpha_PCFG(G, power=alpha, threshold = threshold)
        if G2 == -1:
            alpha_inf = alpha
        else:
            alpha_sup = alpha
            res = G2
    if res == -1:
        res = alpha_PCFG(G, power=alpha_sup, threshold = threshold)
    return res, alpha_sup



def cfg_iterate(G: PCFG, d):
    for S in G.rules:
        for f, args, w in G.rules[S]:
            poss = []
            for a in args:
                poss.append(list(d[a]))
            for l in itertools.product(*poss):          
                d[S].append([f,[*l]])
    return d

def generate_all_programs(G: PCFG, n = 1, only_first = True):
    d = {S: [] for S in G.rules}
    for i in range(n):
        d = cfg_iterate(G, d)
    S = G.start
    if only_first:
        return set([str(p) for p in d[S]])
    return d

def probability(term, proba):
    res = 1
    symbol, sub_terms = term[0], term[1]
    # print("SUBTERM", sub_terms)
    for t in sub_terms:
        # print("MM",t)
        res*=probability(t, proba)
        
    return res*proba[symbol]


def put_random_weight(G, alpha = 0.9):
    '''
    return a grammar with the same structure but with random weights on the transitions
    alpha = 1 is equivalent to uniform
    alpha < 1 gives an exponential decrease in the weights of order alpha**k for the k-th weight
    '''
    G2 = copy.deepcopy(G)
    for X in G2.rules:
        out_degree = len(G2.rules[X])
        weights = [random.random()*(alpha**i) for i in range(out_degree)] # weights with alpha-exponential decrease
        S = sum(weights)
        weights = [e/S for e in weights] # normalization
        random_permutation = list(np.random.permutation([i for i in range(out_degree)]))
        # print(random_permutation)
        for i in range(out_degree):
            G2.rules[X][i][2] = weights[random_permutation[i]]
        G2.rules[X].sort(key = lambda x: -x[2])
    G2.restart()
    return G2
