import itertools

ab, ac, bc = input().split()

for perm in itertools.permutations(['A', 'B', 'C']):
    valid = True
    # Check AB
    a_pos = perm.index('A')
    b_pos = perm.index('B')
    if a_pos < b_pos:
        if ab != '<':
            valid = False
    else:
        if ab != '>':
            valid = False
    if not valid:
        continue
    # Check AC
    a_pos = perm.index('A')
    c_pos = perm.index('C')
    if a_pos < c_pos:
        if ac != '<':
            valid = False
    else:
        if ac != '>':
            valid = False
    if not valid:
        continue
    # Check BC
    b_pos = perm.index('B')
    c_pos = perm.index('C')
    if b_pos < c_pos:
        if bc != '<':
            valid = False
    else:
        if bc != '>':
            valid = False
    if valid:
        print(perm[1])
        break