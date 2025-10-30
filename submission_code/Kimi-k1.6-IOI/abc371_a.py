import itertools

# Read the input
s_ab, s_ac, s_bc = input().split()

# Generate all permutations of the brothers' names
for perm in itertools.permutations(['A', 'B', 'C']):
    # Create a position dictionary for each brother in the current permutation
    pos = {brother: idx for idx, brother in enumerate(perm)}
    
    # Check if the current permutation satisfies all three conditions
    ab_ok = (pos['A'] < pos['B']) == (s_ab == '<')
    ac_ok = (pos['A'] < pos['C']) == (s_ac == '<')
    bc_ok = (pos['B'] < pos['C']) == (s_bc == '<')
    
    if ab_ok and ac_ok and bc_ok:
        # Output the middle brother and exit
        print(perm[1])
        break