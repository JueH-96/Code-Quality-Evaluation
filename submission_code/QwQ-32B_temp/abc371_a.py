import sys
from itertools import permutations

# Read the input values
s_ab, s_ac, s_bc = sys.stdin.readline().split()

# Iterate over all possible permutations of A, B, C
for perm in permutations(['A', 'B', 'C']):
    a_pos = perm.index('A')
    b_pos = perm.index('B')
    c_pos = perm.index('C')
    
    # Check each condition
    ab_ok = (a_pos < b_pos) if (s_ab == '>') else (a_pos > b_pos)
    ac_ok = (a_pos < c_pos) if (s_ac == '>') else (a_pos > c_pos)
    bc_ok = (b_pos < c_pos) if (s_bc == '>') else (b_pos > c_pos)
    
    if ab_ok and ac_ok and bc_ok:
        # The middle element is the second in the permutation (index 1)
        print(perm[1])
        sys.exit()