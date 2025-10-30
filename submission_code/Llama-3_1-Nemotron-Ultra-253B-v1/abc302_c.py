import sys
from itertools import permutations

def diff_by_one(a, b):
    return sum(c1 != c2 for c1, c2 in zip(a, b)) == 1

n, m = map(int, sys.stdin.readline().split())
strings = [sys.stdin.readline().strip() for _ in range(n)]

for perm in permutations(strings):
    valid = True
    for i in range(len(perm) - 1):
        if not diff_by_one(perm[i], perm[i+1]):
            valid = False
            break
    if valid:
        print("Yes")
        sys.exit()

print("No")