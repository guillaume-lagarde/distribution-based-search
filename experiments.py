import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
from math import log, log10, exp

# Import PCFG class
from pcfg import *

# Import algorithms
from threshold_search import threshold_search
from sort_and_add import sort_and_add
from a_star import a_star
from sqrt_sampling import sqrt_sampling
from heap_search import heap_search

# Import DSLs
from benchmarks.flashfill import *
from benchmarks.deepcoder import *
from benchmarks.circuits import *
from benchmarks.menon import *

G_circuits = PCFG('start', rules_circuits)
G_deepcoder = PCFG('start', rules_deepcoder)
G_menon = PCFG('P', rules_menon)
G_flashfill = PCFG('start_0', rules_finite_flashfill)

def create_PCFG(CFG):
	PCFG = put_random_weight(CFG, alpha)
	return PCFG

def create_dataset(PCFG):
	'''
	Create a dataset, which is a list of number_samples programs with proba in [1O^(-(i+1),1O^(-i)] for i in [imin, imax]
	'''
	dataset = []
	size_dataset = [0 for i in range(imax)]
	finished = False

	gen = sampling(PCFG)
	while(not finished):
		program = next(gen)
		proba = PCFG.probability(program)
		i = int(-log10(proba))
		if (i >= imin and i < imax and size_dataset[i] < number_samples):
#            print("This program has probability %0.10f, it goes in bucket %u.\n" % (proba, i))
			dataset.append((program,proba))
			size_dataset[i] += 1
			if size_dataset[i] == number_samples:
				j = imin
				finished = True
				while(finished and j < imax):
					finished = size_dataset[j] == number_samples
					j += 1
	# We sort the dataset by decreasing probability
	dataset = sorted(dataset, key = lambda pair: pair[1], reverse = True)
#	print(dataset)
	return dataset

def run_algorithm(PCFG, algorithm, param):
	'''
	Run the algorithm until either timeout or 3M programs, and for each program record probability and time of output
	'''
	print("Running: %s" % algorithm.__name__)
	result = []
	guess = -1
	N = 0
	chrono = 0
	gen = algorithm(PCFG, *param)
	program = next(gen)
	while (chrono < timeout and N < total_number_programs):
		N += 1
		chrono -= time.perf_counter()
		program = next(gen)
		chrono += time.perf_counter()
		result.append((program, PCFG.probability(program), chrono))
	print("Run successful, output %u programs" % len(result))
#    print(result)
	return result


####################################################
###### First experiment: Probability VS Search time
####################################################

def experiment_probability_vs_search_time(dataset, result):
	result = sorted(result, key = lambda pair: pair[1], reverse = True)
	search_result = []
	index = 0
	for (program, probability_program) in dataset:
		if (index == len(result)):
			search_result.append((proba, timeout))
		else:
			(guess, probability_guess, chrono) = result[index]
			while(probability_guess > probability_program and index+1 < len(result)):
				index += 1
				(guess, probability_guess, chrono) = result[index]

			local_index = index
			while(probability_guess == probability_program and str(program) != str(guess) and local_index+1 < len(result)):
				local_index += 1
				(guess, probability_guess, chrono) = result[local_index]

			if (local_index < len(result) and str(program) == str(guess)):
				search_result.append((probability_program, chrono))
			else:
				search_result.append((probability_program, timeout))
#	print("Number of programs not found: %u" % nb_not_found)
#    print(search_result)
	return(search_result)

def print_plot_probability_vs_search_time(list_algorithms):
	'''
	Retrieve the results and plot
	'''
	for (algorithm, name, param) in list_algorithms:
		if param == []:
			with open('experiment_results/proba_vs_search_time_%s_%s_%s.bin' % (G_name, alpha, algorithm.__name__), 'rb') as f:
				result = pickle.load(f)
		else:
			with open('experiment_results/proba_vs_search_time_%s_%s_%s_%s.bin' % (G_name, alpha, algorithm.__name__, param[0]), 'rb') as f:
				result = pickle.load(f)
		plt.scatter([x for (x,y) in result], [y for (x,y) in result], label = name, s = 8)

	plt.legend()
	plt.xlabel("probability")
	plt.ylabel("search time (in seconds)")
	plt.title(G_name)
	plt.xscale('log')
	plt.yscale('log')
	plt.savefig("images/proba_vs_search_time_%s.png" % G_name, dpi=300, bbox_inches='tight')
	plt.clf()

def probability_vs_search_time(PCFG, dataset, list_algorithms):
	'''
	Run the probability VS search time experiment for all algorithms and plot
	'''
	for (algorithm, name, param) in list_algorithms:
		result = run_algorithm(PCFG, algorithm, param)
		search_result = experiment_probability_vs_search_time(dataset, result)
		if param == []:
			with open('experiment_results/proba_vs_search_time_%s_%s_%s.bin' % (G_name, alpha, algorithm.__name__), 'wb') as f:
				pickle.dump(search_result, f)
		else:
			with open('experiment_results/proba_vs_search_time_%s_%s_%s_%s.bin' % (G_name, alpha, algorithm.__name__, param[0]), 'wb') as f:
				pickle.dump(search_result, f)
	print_plot_probability_vs_search_time(list_algorithms)



####################################################
###### Second experiment: Enumeration time
####################################################

def experiment_enumeration_time(result):
	chrono_result = [chrono for (program, probability, chrono) in result]
	return(chrono_result)

def print_plot_enumeration_time(list_algorithms):
	'''
	Retrieve the results and plot
	'''
	for (algorithm, name, param) in list_algorithms:
		with open('experiment_results/enumeration_time_%s_%s_%s_%s.bin' % (G_name, alpha, algorithm.__name__, param), 'rb') as f:
			result = pickle.load(f)
		plt.plot(range(1, total_number_programs + 1), result, label=name)

	plt.legend()
	plt.xlabel("n first programs")
	plt.ylabel("enumeration time (in seconds)")
	plt.title(G_name)
	plt.savefig("images/enumeration_time_%s.png" % G_name, dpi=300, bbox_inches='tight')
	plt.clf()


def enumeration_time(PCFG, list_algorithms):
	'''
	Run the enumeration time experiment for all algorithms and plot
	'''
	for (algorithm, name, param) in list_algorithms:
		result = run_algorithm(PCFG, algorithm, param)
		chrono_result = experiment_enumeration_time(result)
		with open('experiment_results/enumeration_time_%s_%s_%s_%s.bin' % (G_name, alpha, algorithm.__name__, param), 'wb') as f:
			pickle.dump(chrono_result, f)
	print_plot_enumeration_time(list_algorithms)



####################################################
###### Run the experiments
####################################################

CFG = G_flashfill
G_name = "FlashFill"
alpha = 0.6


### Create a new PCFG
PCFG = create_PCFG(CFG)
with open('experiment_results/PCFG_%s_%s.bin' % (G_name, alpha), 'wb') as f:
	pickle.dump(PCFG, f)

#### Or use an existing one
# with open('PCFG_%s.bin' % G_name, 'rb') as f:
#     PCFG = pickle.load(f)

#print(dataset)


####### First experiment
number_samples = 30 # number of program samples for a given grammar
imin = 3
imax = 7

timeout = 50  # in seconds
total_number_programs = 1000000 # 1M programs

### Create a new dataset
dataset = create_dataset(PCFG)
with open('experiment_results/proba_vs_search_time_dataset_%s_%s.bin' % (G_name, alpha), 'wb') as f:
	pickle.dump(dataset, f)
print("Dataset created\n")

### Or use an existing one
# with open('proba_vs_search_time_dataset_%s_%s.bin' % (G_name, alpha), 'rb') as f:
#     dataset = pickle.load(f)

list_algorithms_probability_vs_search_time = [\
	 # (threshold_search, "Threshold Search(2)", [2]), \
	 (threshold_search, "Threshold Search(10)", [10]), \
	 (sort_and_add, "Sort and Add(1)", [1]), \
	 # (sort_and_add, "Sort and Add(2)", [2]), \
	 (a_star, "A*", []), \
	 (sqrt_sampling, "SQRT Sampling", []), \
	 (heap_search, "Heap Search", [])\
	]

probability_vs_search_time(PCFG, dataset, list_algorithms_probability_vs_search_time)

####### Second experiment

CFG = G_circuits
G_name = "Circuits"
alpha = 0.6

### Create a new PCFG
PCFG = create_PCFG(CFG)
with open('experiment_results/PCFG_%s_%s.bin' % (G_name, alpha), 'wb') as f:
	pickle.dump(PCFG, f)

timeout = 600  # in seconds
total_number_programs = 200000 # 2M programs

list_algorithms_enumeration_time = [\
	(a_star, "A*", []), \
	(heap_search, "Heap Search", [])\
	]

enumeration_time(PCFG, list_algorithms_enumeration_time)
