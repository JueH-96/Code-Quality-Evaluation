import sys
from itertools import permutations

def hamming(a, b):
    return sum(1 for x, y in zip(a, b) if x != y)

n, m = map(int, sys.stdin.readline().split())
strings = [sys.stdin.readline().strip() for _ in range(n)]

for p in permutations(strings):
    valid = True
    for i in range(n-1):
        a = p[i]
        b = p[i+1]
        if hamming(a, b) != 1:
            valid = False
            break
    if valid:
        print("Yes")
        sys.exit()

print("No")