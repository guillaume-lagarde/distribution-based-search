k = 38 # number of instructions
n = 5 # program size
instructions = ['i' + str(i) for i in range(k)]
rules_deepcoder = {
    'start': [['P', ['I']*n, 0]],
    'I': [ [i, [], 1/k] for i in instructions]
}
