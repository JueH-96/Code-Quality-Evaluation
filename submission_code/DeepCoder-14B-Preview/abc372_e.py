import itertools

# Read the input
s_ab, s_ac, s_bc = input().split()

# Generate all possible permutations of the brothers
for perm in itertools.permutations(['A', 'B', 'C']):
    # Create a dictionary to map each brother to their rank (1=oldest, 3=youngest)
    rank = {brother: i+1 for i, brother in enumerate(perm)}
    
    # Check condition for S_AB
    cond_ab = (s_ab == '<' and rank['A'] > rank['B']) or (s_ab == '>' and rank['A'] < rank['B'])
    
    # Check condition for S_AC
    cond_ac = (s_ac == '<' and rank['A'] > rank['C']) or (s_ac == '>' and rank['A'] < rank['C'])
    
    # Check condition for S_BC
    cond_bc = (s_bc == '<' and rank['B'] > rank['C']) or (s_bc == '>' and rank['B'] < rank['C'])
    
    # If all conditions are satisfied, print the middle brother
    if cond_ab and cond_ac and cond_bc:
        print(perm[1])
        exit()