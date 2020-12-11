rules_menon = {
    'P': [['join', ['LIST', 'DELIM'], 1]],

    'LIST': [['split', ['DELIM'], 0.3],
             ['concatList', ['CAT','CAT','CAT'], 0.1],
             ['concatList_()', ['CAT'], 0.2],
             ['dedup', ['LIST'], 0.2],
             ['count', ['LIST', 'LIST'], 0.2]],

    'CAT': [['cat1', ['LIST'], 0.7],
            ['cat2', ['DELIM'], 0.3]],

    'DELIM': [['\n', [], 0.5],
              [' ', [], 0.3],
              ['(', [], 0.1],
              [')', [], 0.1]]
    }
