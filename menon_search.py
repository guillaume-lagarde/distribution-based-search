from pcfg import *
import random
from math import sqrt
from heapq import heappush, heappop
import copy
import timeit
import itertools

# -----------------------------------
# ------ MENON ALGORITHM ---------
# ----------------------------------- 


# def get_value(t, i):
#     '''
#     get the value in term t given by the tuple of indice i
#     '''
#     nt = copy.deepcopy(t)
#     nt = t
#     for i in i:
#         nt = nt[i]
#     return nt

# def set_value(t, i, v):
#     '''
#     set the value in term t at position i (a tuple) to v
#     '''

#     ref = t
#     for j in range(len(i)-1):
#         ref = ref[i[j]]
#     ref[i[-1]] = v

def menon_search(G: PCFG, scale_factor: float):
    threshold = 0.1
    dictionary = {}
    seen = set()
    
    for X in G.rules:
        set_max_tuple(G, X, seen, dictionary)
        
    max_weights = {X:dictionary[X][1] for X in G.rules}
        
    gen = menon(G, threshold, max_weights)
    while True:
        try:
            yield next(gen)
        except StopIteration:
            threshold/=scale_factor
            gen = menon(G, threshold, max_weights)
    

def menon(G : PCFG, threshold: float, max_weights):
    '''
    A generator that samples all terms with proba greater than or equal to threshold
    '''
    
    T = [([G.start], [[0]], max_weights[G.start])] # partials term and the indices where we should try to expand in LIFO style
    
    while T:
        term, indices, max_w = T.pop()
        index_to_change = indices.pop()
        ns = get_value(term,index_to_change) # symbol to be replaced in the term nt
                
        for f, args, w in G.rules[ns]:
            new_weights = 1
            for s in args:
                new_weights*=max_weights[s] # product of the maximum weights for the new elements
            new_term = copy.deepcopy(term)
            set_value(new_term, index_to_change, copy.deepcopy([f,args]))
            
            new_indices = copy.deepcopy(indices)
            for j in range(len(args)-1,-1,-1):
                new_index = copy.deepcopy(index_to_change)
                new_index.extend([1,j])
                new_indices.append(new_index)
            new_max_weight = w*max_w*new_weights/max_weights[ns]
            if new_max_weight >= threshold:
                if len(new_indices) == 0:
                    yield new_term[0]# , new_max_weight
                else:
                    T.append((new_term,new_indices, new_max_weight)) #weight of derivation* max all new symbols / max symbol that has been replaced
