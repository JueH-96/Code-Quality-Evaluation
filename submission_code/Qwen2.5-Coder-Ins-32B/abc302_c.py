# YOUR CODE HERE
from itertools import permutations

def can_rearrange(strings):
    def differ_by_one(s1, s2):
        return sum(c1 != c2 for c1, c2 in zip(s1, s2)) == 1

    for perm in permutations(strings):
        if all(differ_by_one(perm[i], perm[i+1]) for i in range(len(perm) - 1)):
            return "Yes"
    return "No"

import sys
input = sys.stdin.read().split()
N = int(input[0])
M = int(input[1])
strings = input[2:]

print(can_rearrange(strings))