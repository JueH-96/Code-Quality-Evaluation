MOD = 998244353

def count_ways(N, M, K, edges):
    # Create adjacency list for the graph
    graph = [[] for _ in range(N + 1)]
    for i in range(1, N):
        graph[i].append(i + 1)
    graph[N].append(1)

    for X, Y in edges:
        graph[X].append(Y)

    # Initialize dp array
    dp = [[0] * (N + 1) for _ in range(K + 1)]
    dp[0][1] = 1

    # Fill dp array
    for k in range(1, K + 1):
        for v in range(1, N + 1):
            for u in graph[v]:
                dp[k][u] = (dp[k][u] + dp[k - 1][v]) % MOD

    # Sum up all possible ending vertices
    result = sum(dp[K]) % MOD
    return result

# Read input
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
K = int(data[2])

edges = []
for i in range(M):
    X = int(data[3 + 2 * i])
    Y = int(data[4 + 2 * i])
    edges.append((X, Y))

# Calculate and print the result
result = count_ways(N, M, K, edges)
print(result)