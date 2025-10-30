MOD = 998244353

def main():
    N, M, K = map(int, input().split())
    additional_edges = [tuple(map(int, input().split())) for _ in range(M)]

    # Initialize adjacency list
    adjacency_list = {i: [(i % N) + 1] for i in range(1, N + 1)}
    for x, y in additional_edges:
        adjacency_list.setdefault(x, []).append(y)

    # Initialize dp table
    dp = [[0] * (N + 1) for _ in range(K + 1)]
    dp[0][1] = 1

    # Dynamic programming to count paths
    for i in range(1, K + 1):
        for u in range(1, N + 1):
            for v in adjacency_list.get(u, []):
                dp[i][v] = (dp[i][v] + dp[i - 1][u]) % MOD

    # Sum up the ways to reach each vertex in K steps
    result = sum(dp[K]) % MOD
    print(result)

if __name__ == "__main__":
    main()