from heapq import *
from sys import stdin

N, M, K = map(int, stdin.readline().split())
A = list(map(int, stdin.readline().split()))
A = [(a, i) for i, a in enumerate(A)]
heapify(A)
B = [0]*N
C = [0]*N
S = sum(a for a, i in A)
T = [0]*(N+1)
for i in range(N-1, -1, -1):
    a, j = A[i]
    B[j] = max(0, a - (K-S) + 1)
    S -= a
    while len(T) > M and T[-M-1] <= a:
        S -= T.pop()
    C[j] = max(0, a - (K-S) + 1)
    heappush(T, a)
for b, c in zip(B, C):
    print(-1 if b > c else c)