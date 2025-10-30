import sys
from itertools import permutations

def can_transform(s1, s2):
    return sum(c1 != c2 for c1, c2 in zip(s1, s2)) == 1

def solve(N, M, strings):
    for perm in permutations(strings):
        if all(can_transform(perm[i], perm[i+1]) for i in range(N-1)):
            return "Yes"
    return "No"

input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
strings = data[2:]

print(solve(N, M, strings))