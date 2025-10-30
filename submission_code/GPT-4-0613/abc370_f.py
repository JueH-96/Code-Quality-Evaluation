import sys
from collections import deque
from bisect import bisect_right

def solve():
    N, K = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    A += A
    S = [0]*(2*N+1)
    for i in range(2*N):
        S[i+1] = S[i] + A[i]
    l = 0
    r = sum(A)
    while r - l > 1:
        m = (l + r) // 2
        ok = False
        for i in range(N):
            if S[i+K] - S[i] >= m and S[i+N] - S[i+K] >= m:
                ok = True
                break
        if ok:
            l = m
        else:
            r = m
    B = deque()
    for i in range(N):
        if S[i+K] - S[i] >= l and S[i+N] - S[i+K] >= l:
            B.append(i)
    C = [0]*N
    for i in range(N):
        C[i] = bisect_right(B, i)
    D = [0]*N
    for i in range(N-1, -1, -1):
        D[i] = max(D[(i+1)%N], C[i])
    print(l, D[0])

solve()