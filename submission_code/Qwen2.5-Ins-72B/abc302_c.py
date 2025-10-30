# YOUR CODE HERE
from itertools import permutations

def can_rearrange(strings):
    def is_one_char_diff(s1, s2):
        return sum(1 for c1, c2 in zip(s1, s2) if c1 != c2) == 1
    
    for perm in permutations(strings):
        if all(is_one_char_diff(perm[i], perm[i+1]) for i in range(len(perm) - 1)):
            return "Yes"
    return "No"

import sys
input = sys.stdin.read
data = input().split()

N, M = map(int, data[0].split())
strings = data[1:]

print(can_rearrange(strings))