# YOUR CODE HERE
def find_middle_brother(S_AB, S_AC, S_BC):
    if S_AB == '<' and S_AC == '<' and S_BC == '<':
        return 'B'
    elif S_AB == '<' and S_AC == '<' and S_BC == '>':
        return 'C'
    elif S_AB == '<' and S_AC == '>' and S_BC == '<':
        return 'A'
    elif S_AB == '<' and S_AC == '>' and S_BC == '>':
        return 'C'
    elif S_AB == '>' and S_AC == '<' and S_BC == '<':
        return 'A'
    elif S_AB == '>' and S_AC == '<' and S_BC == '>':
        return 'B'
    elif S_AB == '>' and S_AC == '>' and S_BC == '<':
        return 'B'
    elif S_AB == '>' and S_AC == '>' and S_BC == '>':
        return 'A'

import sys
input = sys.stdin.read
S_AB, S_AC, S_BC = input().strip().split()
print(find_middle_brother(S_AB, S_AC, S_BC))