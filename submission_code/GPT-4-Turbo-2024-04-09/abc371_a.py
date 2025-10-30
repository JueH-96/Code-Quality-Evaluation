# Read input
S_AB, S_AC, S_BC = input().split()

# Determine the middle brother based on the relationships
if S_AB == '<' and S_AC == '<':
    # A < B and A < C
    if S_BC == '<':
        # A < B < C
        middle = 'B'
    else:
        # A < C < B
        middle = 'C'
elif S_AB == '>' and S_AC == '>':
    # A > B and A > C
    if S_BC == '>':
        # C < B < A
        middle = 'B'
    else:
        # B < C < A
        middle = 'C'
elif S_AB == '<' and S_AC == '>':
    # A < B and A > C
    if S_BC == '<':
        # C < A < B
        middle = 'A'
    else:
        # A < C < B
        middle = 'C'
elif S_AB == '>' and S_AC == '<':
    # A > B and A < C
    if S_BC == '>':
        # B < A < C
        middle = 'A'
    else:
        # B < C < A
        middle = 'B'

# Output the middle brother
print(middle)