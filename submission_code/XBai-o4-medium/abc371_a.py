import itertools

# Read input
s = input().split()
s_ab, s_ac, s_bc = s[0], s[1], s[2]

# Generate all possible permutations of A, B, C
for perm in itertools.permutations(['A', 'B', 'C']):
    a_pos = perm.index('A')
    b_pos = perm.index('B')
    c_pos = perm.index('C')
    
    # Generate the relations for this permutation
    generated_ab = '>' if a_pos < b_pos else '<'
    generated_ac = '>' if a_pos < c_pos else '<'
    generated_bc = '>' if b_pos < c_pos else '<'
    
    # Check if all match the input
    if (generated_ab == s_ab and 
        generated_ac == s_ac and 
        generated_bc == s_bc):
        print(perm[1])
        break