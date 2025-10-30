import itertools

s_ab, s_ac, s_bc = input().split()

for perm in itertools.permutations(['A', 'B', 'C']):
    age = {perm[0]: 1, perm[1]: 2, perm[2]: 3}
    if (age['A'] < age['B']) != (s_ab == '<'):
        continue
    if (age['A'] < age['C']) != (s_ac == '<'):
        continue
    if (age['B'] < age['C']) != (s_bc == '<'):
        continue
    print(perm[1])
    break