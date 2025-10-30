import itertools

# Read the input values
s_ab, s_ac, s_bc = input().split()

# Generate all permutations of A, B, C
for perm in itertools.permutations(['A', 'B', 'C']):
    # Check AB relation
    a_ab = perm.index('A')
    b_ab = perm.index('B')
    if (a_ab < b_ab and s_ab != '>') or (a_ab >= b_ab and s_ab != '<'):
        continue
    
    # Check AC relation
    a_ac = perm.index('A')
    c_ac = perm.index('C')
    if (a_ac < c_ac and s_ac != '>') or (a_ac >= c_ac and s_ac != '<'):
        continue
    
    # Check BC relation
    b_bc = perm.index('B')
    c_bc = perm.index('C')
    if (b_bc < c_bc and s_bc != '>') or (b_bc >= c_bc and s_bc != '<'):
        continue
    
    # If all conditions are satisfied, output the middle brother
    print(perm[1])
    break