from itertools import groupby

S = input()
result = list()
for operator, group in groupby(S):
    if operator == '-':
        result.append(len(list(group)))
    else:
        if result:
            print(*result)
            break