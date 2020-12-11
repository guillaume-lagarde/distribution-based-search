constants_circuits = ['C1', 'C2', 'C3', 'C4']

rules_circuits = {
    'start': [['and_1', ['depth1','depth1'], 0],
              ['not_1', ['depth1'], 0],
              ['or_1', ['depth1','depth1'], 0],
              ['xor_1', ['depth1','depth1'], 0]],
    
    'depth1': [['and_2', ['depth2','depth2'], 0],
              ['not_2', ['depth2'], 0],
              ['or_2', ['depth2','depth2'], 0],
               ['xor_2', ['depth2','depth2'], 0]],
    
    'depth2': [['and_3', ['depth3','depth3'], 0],
              ['not_3', ['depth3'], 0],
              ['or_3', ['depth3','depth3'], 0],
               ['xor_3', ['depth3','depth3'], 0]],

    'depth3': [['and_4', ['depth4','depth4'], 0],
              ['not_4', ['depth4'], 0],
              ['or_4', ['depth4','depth4'], 0],
               ['xor_4', ['depth4','depth4'], 0]],
    
    'depth4': [],
    }

for X in rules_circuits:
    for c in constants_circuits:
        rules_circuits[X].append([c + "_" + X, [], 0])


