import sys
from collections import defaultdict

N, Q = map(int, sys.stdin.readline().split())
queries = list(map(int, sys.stdin.readline().split()))

S = set()
A = [0]*N
total = 0

for x in queries:
    x -= 1
    if x in S:
        S.remove(x)
        total -= len(S)
    else:
        total += len(S)
        S.add(x)
    A[x] += total

for i in range(1, N):
    A[i] += A[i-1]

for i in range(N):
    if i+1 not in S:
        A[i] -= Q

sys.stdout.write(' '.join(map(str, A)) + '
')