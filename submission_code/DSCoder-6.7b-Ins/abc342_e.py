import sys
from heapq import heappop, heappush
from collections import defaultdict

def solve():
    N, M = map(int, sys.stdin.readline().split())
    trains = []
    for _ in range(M):
        l, d, k, c, A, B = map(int, sys.stdin.readline().split())
        for i in range(k):
            trains.append((l + i * d, c, A, B))
    trains.sort()

    dp = [0] * (N + 1)
    dp[1] = float('inf')
    hq = [(0, 1)]
    for t, c, A, B in trains:
        while hq[0][0] < t:
            _, B = heappop(hq)
        if B == N:
            continue
        dp[B] = max(dp[B], dp[A] + c)
        heappush(hq, (-dp[B], B))

    print(max(dp[i] for i in range(1, N) if dp[i] != 0))

solve()