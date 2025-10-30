import sys
from heapq import heappop, heappush

def main():
    N, M = map(int, sys.stdin.readline().split())
    trains = [[] for _ in range(N+1)]
    for _ in range(M):
        l, d, k, c, A, B = map(int, sys.stdin.readline().split())
        for i in range(k):
            t = l + i * d
            trains[A].append((t, t + c, B))
    for i in range(1, N+1):
        trains[i].sort(reverse=True)
    INF = 10**18
    dp = [-INF] * (N+1)
    dp[N] = INF
    que = [(INF, N)]
    while que:
        t, v = heappop(que)
        while trains[v] and trains[v][-1][0] <= t:
            _, t2, to = trains[v].pop()
            if dp[to] < t2:
                dp[to] = t2
                heappush(que, (t2, to))
    for i in range(1, N):
        print(dp[i] if dp[i] != -INF else 'Unreachable')

if __name__ == "__main__":
    main()