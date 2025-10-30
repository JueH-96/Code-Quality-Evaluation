import sys
input = sys.stdin.readline

MOD = 998244353

def main():
    N, M, K = map(int, input().split())
    edges = []
    for _ in range(M):
        X, Y = map(int, input().split())
        edges.append((X, Y))

    # Create adjacency list
    adj = [[] for _ in range(N + 1)]
    for i in range(1, N + 1):
        adj[i].append((i % N) + 1)
    for X, Y in edges:
        adj[X].append(Y)

    # Initialize dp table
    dp = [[0] * (N + 1) for _ in range(K + 1)]
    dp[0][1] = 1

    # Fill dp table
    for k in range(1, K + 1):
        for v in range(1, N + 1):
            for u in adj[v]:
                dp[k][u] = (dp[k][u] + dp[k - 1][v]) % MOD

    # Calculate answer
    ans = sum(dp[K]) % MOD
    print(ans)

if __name__ == "__main__":
    main()