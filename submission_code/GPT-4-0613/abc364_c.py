import sys
from heapq import heapify, heappop, heappush
from bisect import bisect_left

N, X, Y = map(int, sys.stdin.readline().split())
A = list(map(int, sys.stdin.readline().split()))
B = list(map(int, sys.stdin.readline().split()))

AB = sorted([(a, b) for a, b in zip(A, B)], key=lambda x: -x[1])

INF = 10**18
dp = [[INF]*(Y+1) for _ in range(N+1)]
dp[0][0] = 0

for i in range(N):
    a, b = AB[i]
    for j in range(i+1)[::-1]:
        for k in range(Y+1):
            if k+b <= Y:
                dp[j+1][k+b] = min(dp[j+1][k+b], dp[j][k]+a)

for i in range(N+1)[::-1]:
    if min(dp[i]) <= X:
        print(i)
        break