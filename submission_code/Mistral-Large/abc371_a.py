import sys

# Read input from stdin
S_AB, S_AC, S_BC = sys.stdin.read().strip().split()

# Determine the age relationships
if S_AB == '<':
    if S_AC == '<':
        if S_BC == '<':
            middle = 'B'
        else:  # S_BC == '>'
            middle = 'C'
    else:  # S_AC == '>'
        if S_BC == '<':
            middle = 'A'
        else:  # S_BC == '>'
            middle = 'C'
else:  # S_AB == '>'
    if S_AC == '<':
        if S_BC == '<':
            middle = 'A'
        else:  # S_BC == '>'
            middle = 'B'
    else:  # S_AC == '>'
        if S_BC == '<':
            middle = 'A'
        else:  # S_BC == '>'
            middle = 'B'

# Print the result to stdout
print(middle)