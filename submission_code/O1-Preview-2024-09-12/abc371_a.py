# YOUR CODE HERE
import sys
S_AB, S_AC, S_BC = input().split()

import itertools
for perm in itertools.permutations(['A', 'B', 'C']):
    AgeOrder = {}
    AgeOrder[perm[0]] = 3  # oldest
    AgeOrder[perm[1]] = 2  # middle
    AgeOrder[perm[2]] = 1  # youngest
    valid = True

    # Check S_AB
    if S_AB == '<':
        if not AgeOrder['A'] < AgeOrder['B']:
            valid = False
    elif S_AB == '>':
        if not AgeOrder['A'] > AgeOrder['B']:
            valid = False

    # Check S_AC
    if S_AC == '<':
        if not AgeOrder['A'] < AgeOrder['C']:
            valid = False
    elif S_AC == '>':
        if not AgeOrder['A'] > AgeOrder['C']:
            valid = False

    # Check S_BC
    if S_BC == '<':
        if not AgeOrder['B'] < AgeOrder['C']:
            valid = False
    elif S_BC == '>':
        if not AgeOrder['B'] > AgeOrder['C']:
            valid = False

    if valid:
        print(perm[1])
        break