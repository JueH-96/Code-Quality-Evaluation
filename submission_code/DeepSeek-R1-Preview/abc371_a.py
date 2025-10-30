import itertools

# Read the input
s_ab, s_ac, s_bc = input().split()

# Generate all possible orderings of the three brothers
for order in itertools.permutations(['A', 'B', 'C']):
    a_pos = order.index('A')
    b_pos = order.index('B')
    c_pos = order.index('C')
    
    # Check if the current order satisfies all three conditions
    # Check S_AB
    if (a_pos < b_pos) != (s_ab == '>'):
        continue
    # Check S_AC
    if (a_pos < c_pos) != (s_ac == '>'):
        continue
    # Check S_BC
    if (b_pos < c_pos) != (s_bc == '>'):
        continue
    
    # If all conditions are satisfied, print the middle brother
    print(order[1])
    exit()