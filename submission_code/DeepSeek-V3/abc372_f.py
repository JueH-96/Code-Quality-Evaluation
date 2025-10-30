import sys
from collections import defaultdict

MOD = 998244353

def main():
    N, M, K = map(int, sys.stdin.readline().split())
    adj = defaultdict(list)
    for i in range(1, N+1):
        adj[i].append((i % N) + 1)
    for _ in range(M):
        X, Y = map(int, sys.stdin.readline().split())
        adj[X].append(Y)
    
    dp = [0] * (N + 1)
    dp[1] = 1
    for _ in range(K):
        new_dp = [0] * (N + 1)
        for u in range(1, N+1):
            if dp[u] == 0:
                continue
            for v in adj[u]:
                new_dp[v] = (new_dp[v] + dp[u]) % MOD
        dp = new_dp
    print(dp[1] % MOD)

if __name__ == "__main__":
    main()