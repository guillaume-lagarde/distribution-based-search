constants_string = ['bla', 'bli']
constants_int = [str(i) for i in range(10)]
constants_bool = ['true', 'false']

rules_flashfill = {
    'start': [['String',['ntString'],0]],
    
    'ntString': [['str.++', ['ntString', 'ntString'], 0],
                 ['str.replace', ['ntString', 'ntString', 'ntString'], 0],
                 ['str.at', ['ntString', 'ntInt'], 0],
                 ['int.to.str', ['ntInt'], 0],
                 ['ite', ['ntBool', 'ntString', 'ntString'], 0],
                 ['str.substr', ['ntString', 'ntInt', 'ntInt'], 0]],

    'ntInt': [['+', ['ntInt', 'ntInt'], 0],
              ['-', ['ntInt', 'ntInt'], 0],
              ['str.len', ['ntString'], 0],
              ['str.to.int', ['ntString'], 0],
              ['ite', ['ntBool', 'ntInt', 'ntInt'], 0],
              ['str.indexof', ['ntString', 'ntString', 'ntInt'], 0]],

    'ntBool': [['=', ['ntInt', 'ntInt'], 0],
               ['str.prefixof', ['ntString', 'ntString'], 0],
               ['str.suffixof', ['ntString', 'ntString'], 0],
               ['str.contains', ['ntString', 'ntString'], 0]]
}


for c in constants_string:
    rules_flashfill['ntString'].append([c, [], 0])
for c in constants_int:
    rules_flashfill['ntInt'].append([c, [], 0])
for c in constants_bool:
    rules_flashfill['ntBool'].append([c, [], 0])        


def finite_grammar(h = 4):
    rules = {}
    for i in range(h):
        for S in rules_flashfill:
            S2 = S + "_" + str(i)
            rules[S2] = []
            for f, args, w in rules_flashfill[S]:
                if i < h-1 or len(args)==0:
                    rules[S2].append([f + "_" + S + "_" + str(i), [a + "_" + str(i+1) for a in args], w])
    return rules

rules_finite_flashfill = finite_grammar(4)
