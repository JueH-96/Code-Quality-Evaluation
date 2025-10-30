# YOUR CODE HERE
def find_middle_brother(S_AB, S_AC, S_BC):
    # Determine the relative ages based on the given relationships
    if S_AB == '<' and S_AC == '<' and S_BC == '<':
        return 'B'
    elif S_AB == '<' and S_AC == '<' and S_BC == '>':
        return 'C'
    elif S_AB == '<' and S_AC == '>' and S_BC == '<':
        return 'A'
    elif S_AB == '<' and S_AC == '>' and S_BC == '>':
        return 'A'
    elif S_AB == '>' and S_AC == '<' and S_BC == '<':
        return 'A'
    elif S_AB == '>' and S_AC == '<' and S_BC == '>':
        return 'C'
    elif S_AB == '>' and S_AC == '>' and S_BC == '<':
        return 'B'
    elif S_AB == '>' and S_AC == '>' and S_BC == '>':
        return 'B'

import sys
input = sys.stdin.read().strip()
S_AB, S_AC, S_BC = input.split()
print(find_middle_brother(S_AB, S_AC, S_BC))