import sys
from itertools import accumulate
from collections import Counter

def readln(): return tuple(map(int, sys.stdin.readline().split()))

N, Q = readln()
A = list(readln())
B = list(readln())
accA, accB = list(accumulate(A, func=lambda x,y: x + [x[-1][y] + 1] if len(x)>0 and x[-1][y] >= 0 else x + [-1], initial=[-1 for _ in range(N+1)]))[:-1], list(accumulate(B, func=lambda x,y: x + [x[-1][y] + 1] if len(x)>0 and x[-1][y] >= 0 else x + [-1], initial=[-1 for _ in range(N+1)]))[:-1]
queries = [readln() for _ in range(Q)]
for l, r, L, R in queries:
    cntA, cntB = Counter(), Counter()
    for x in range(N+1):
        if accA[r-1][x] - (accA[l-2][x] if l>1 else 0) > 0: cntA[x] += 1
        if accB[R-1][x] - (accB[L-2][x] if L>1 else 0) > 0: cntB[x] += 1
    print('Yes' if cntA == cntB else 'No')