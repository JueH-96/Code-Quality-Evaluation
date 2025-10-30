# YOUR CODE HERE
import sys
input = sys.stdin.readline

def solve():
    N, M, K = map(int, input().split())
    MOD = 998244353
    graph = [[] for _ in range(N)]
    for i in range(N):
        graph[i].append((i+1) % N)
    for _ in range(M):
        X, Y = map(int, input().split())
        graph[X-1].append(Y-1)
    
    dp = [[0] * N for _ in range(K+1)]
    dp[0][0] = 1
    
    for k in range(K):
        for v in range(N):
            for nv in graph[v]:
                dp[k+1][nv] += dp[k][v]
                dp[k+1][nv] %= MOD
    
    print(dp[K][0])

solve()