import sys
import math

def main():
    sys.setrecursionlimit(1 << 25)
    N, M = map(int, sys.stdin.readline().split())
    adj = [[] for _ in range(N + 1)]
    for _ in range(M):
        u, v, b, c = map(int, sys.stdin.readline().split())
        adj[u].append((v, b, c))
    
    low = 0.0
    high = 1e18
    
    for _ in range(100):
        mid = (low + high) / 2
        dp = [-math.inf] * (N + 1)
        dp[1] = 0.0
        for u in range(1, N + 1):
            if dp[u] == -math.inf:
                continue
            for (v, b, c) in adj[u]:
                weight = b - mid * c
                if dp[v] < dp[u] + weight:
                    dp[v] = dp[u] + weight
        if dp[N] >= 0:
            low = mid
        else:
            high = mid
    
    print("{0:.15f}".format(low))

if __name__ == '__main__':
    main()