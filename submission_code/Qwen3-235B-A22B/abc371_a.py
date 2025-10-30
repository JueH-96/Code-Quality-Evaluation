import sys
import itertools

# Read input
s_ab, s_ac, s_bc = sys.stdin.read().split()

# Iterate through all permutations of A, B, C
for perm in itertools.permutations(['A', 'B', 'C']):
    # Find positions of each brother in the permutation
    a_pos = perm.index('A')
    b_pos = perm.index('B')
    c_pos = perm.index('C')
    
    # Check each condition
    cond1 = (s_ab == '<' and a_pos > b_pos) or (s_ab == '>' and a_pos < b_pos)
    cond2 = (s_ac == '<' and a_pos > c_pos) or (s_ac == '>' and a_pos < c_pos)
    cond3 = (s_bc == '<' and b_pos > c_pos) or (s_bc == '>' and b_pos < c_pos)
    
    # If all conditions are satisfied
    if cond1 and cond2 and cond3:
        print(perm[1])
        break