import unittest
from pcfg import *

# Import algorithms
from threshold_search import threshold
from sort_and_add import sort_and_add
from a_star import a_star
from sqrt_sampling import sqrt_sampling
from heap_search import heap_search


from benchmarks.flashfill import *
from benchmarks.deepcoder import *
from benchmarks.circuits import *
from benchmarks.menon import *

class TestSum(unittest.TestCase):


    def test_completeness_heap_search(self):
        '''
        Check if heap_search does not miss any program and if it outputs programs in decreasing order.
        '''

        N = 25000 # number of program to be output by the algorithm
        K = 3000 # number of sampling
        G = PCFG('start_0', rules_finite_flashfill)
        G = put_random_weight(G)
        G.restart()

        S = sampling(G)
        H = heap_search(G)
        seen_sampling = set()
        seen_heaps = set()

        proba_current = 1
        for i in range(N):
            if (100*i//N) != (100*(i+1)//N):
                print(100*(i+1)//N, " %")
            t = next(H)
            proba_t = G.probability(t)
            self.assertLessEqual(proba_t, proba_current) # check if in decreasing order
            proba_current = proba_t
            seen_heaps.add(str(t))

        min_proba = proba_current
        
        while len(seen_sampling) < K:
            t = next(S)
            proba_t = G.probability(t)
            if proba_t > min_proba:
                seen_sampling.add(str(t))

        diff = seen_sampling - seen_heaps
            
        self.assertEqual(0, len(diff))

        

    def test_completeness_a_star(self):
        '''
        Check if A_star does not miss any program and if it outputs programs in decreasing order.
        '''

        N = 15000 # number of program to be output by the algorithm
        K = 1000 # number of sampling        
        G = PCFG('start_0', rules_finite_flashfill)
        G = put_random_weight(G)
        G.restart()

        S = sampling(G)
        A = a_star(G)
        seen_sampling = set()
        seen_astar = set()

        proba_current = 1
        for i in range(N):
            if (100*i//N) != (100*(i+1)//N):
                print(100*(i+1)//N, " %")
            t = next(A)
            proba_t = G.probability(t)
            self.assertLessEqual(proba_t, proba_current) # check if in decreasing order
            proba_current = proba_t
            seen_astar.add(str(t))

        min_proba = proba_current
        
        while len(seen_sampling) < K:
            t = next(S)
            proba_t = G.probability(t)
            if proba_t > min_proba:
                seen_sampling.add(str(t))

        diff = seen_sampling - seen_astar
            
        self.assertEqual(0, len(diff))

    def test_threshold(self):
        '''
        Test if Threshold does not miss any program and output programs above the given threshold
        '''

        THR = 0.000001
        G = PCFG('start_0', rules_finite_flashfill)
        G = put_random_weight(G)
        G.restart()

        # Compute max probabilities
        dictionary = {}
        seen = set()
        for X in G.rules:
            set_max_tuple(G, X, seen, dictionary)        
        max_weights = {X:dictionary[X][1] for X in G.rules}
        # End max proba

        M = threshold(G, THR, max_weights)
        
        seen_threshold = set()

        while True:
            try:
                t = next(M)
                proba_t = G.probability(t)
                self.assertLessEqual(THR, proba_t) # check if the program is above THR
                seen_threshold.add(str(t))
            except StopIteration:
                break
        K = len(seen_threshold)//2

        S = sampling(G)
        seen_sampling = set()
        while len(seen_sampling) < K:
            t = next(S)
            proba_t = G.probability(t)
            if proba_t >= THR:
                seen_sampling.add(str(t))

        diff = seen_sampling - seen_threshold
        
        self.assertEqual(0, len(diff))
            
    def test_equivalence_heaps_astar(self):
        '''
        Test if heap_search and A* are equivalent
        '''
        
        N = 10000 # number of program to be output by the algorithm        

        G = PCFG('start_0', rules_finite_flashfill)
        G = PCFG('start', rules_deepcoder)
        G = put_random_weight(G)
        G.restart()

        A = a_star(G)
        H = heap_search(G)
        for i in range(N):
            if (100*i//N) != (100*(i+1)//N):
                print(100*(i+1)//N, " %")
            t_heap = next(H)
            t_astar = next(A)
            p1 = G.probability(t_heap)
            p2 = G.probability(t_astar)
            self.assertAlmostEqual(p1, p2, places = 14)
            
    
    def test_sampling(self):
        '''
        test if the sampling algorithm samples according to the true probabilities
        '''
        K = 3_000_000 # number of programs sampled
        L = 100 # we test the probabilities of the first L programs are ok
        G = PCFG('start_0', rules_finite_flashfill)
        G = put_random_weight(G)
        G.restart()

        H = heap_search(G) # to generate the L first programs
        
        S = sampling(G)
        programs = [next(H) for _ in range(L)]
        count = {str(t): 0 for t in programs}
        
        for i in range(K):
            if (100*i//K) != (100*(i+1)//K):
                print(100*(i+1)//K, " %")
            t = next(S)
            t_hashed = str(t)
            if t_hashed in count:
                count[t_hashed]+=1

        for t in programs:
            proba_t = G.probability(t)
            self.assertAlmostEqual(proba_t,count[str(t)]/K, places = 2)

        
        
if __name__ == '__main__':
    unittest.main()
