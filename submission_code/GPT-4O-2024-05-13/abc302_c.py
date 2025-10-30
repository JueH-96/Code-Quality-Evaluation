# YOUR CODE HERE
import itertools

def can_rearrange(N, M, strings):
    def differ_by_one_char(s1, s2):
        diff_count = sum(1 for a, b in zip(s1, s2) if a != b)
        return diff_count == 1

    for perm in itertools.permutations(strings):
        if all(differ_by_one_char(perm[i], perm[i+1]) for i in range(N-1)):
            return "Yes"
    return "No"

import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
strings = data[2:]

print(can_rearrange(N, M, strings))