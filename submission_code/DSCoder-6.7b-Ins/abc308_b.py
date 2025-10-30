# YOUR CODE HERE

import sys

N, M = map(int, sys.stdin.readline().split())
C = sys.stdin.readline().split()
D = sys.stdin.readline().split()
P = list(map(int, sys.stdin.readline().split()))

total = 0
for i in range(N):
    if C[i] in D:
        total += P[D.index(C[i])]
    else:
        total += P[0]

print(total)