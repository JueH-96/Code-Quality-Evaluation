import sys
input = sys.stdin.read

MOD = 998244353

def main():
    N, M, K = map(int, input().split())
    edges = [tuple(map(int, input().split())) for _ in range(M)]

    # Initialize the adjacency list for the graph
    graph = [[] for _ in range(N+1)]
    for i in range(1, N+1):
        graph[i].append(i % N + 1)
    for x, y in edges:
        graph[x].append(y)

    # Initialize the dp array
    dp = [[0] * (K+1) for _ in range(N+1)]
    dp[1][0] = 1

    # Fill the dp array
    for k in range(1, K+1):
        for v in range(1, N+1):
            for u in graph[v]:
                dp[u][k] = (dp[u][k] + dp[v][k-1]) % MOD

    # Sum up the number of ways to reach any vertex after K moves
    result = sum(dp[v][K] for v in range(1, N+1)) % MOD
    print(result)

if __name__ == "__main__":
    main()