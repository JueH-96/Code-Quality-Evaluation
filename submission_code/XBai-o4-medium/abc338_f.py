import sys
import itertools

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr]); ptr += 1
    M = int(input[ptr]); ptr += 1

    INF = 10**18

    # Initialize distance matrix
    dist = [[INF] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        dist[i][i] = 0

    # Read edges
    for _ in range(M):
        u = int(input[ptr]); ptr += 1
        v = int(input[ptr]); ptr += 1
        w = int(input[ptr]); ptr += 1
        if dist[u][v] > w:
            dist[u][v] = w

    # Floyd-Warshall algorithm
    for k in range(1, N + 1):
        for i in range(1, N + 1):
            for j in range(1, N + 1):
                if dist[i][j] > dist[i][k] + dist[k][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]

    # DP initialization
    size = 1 << N
    dp = [[INF] * (N + 1) for _ in range(size)]
    for u in range(1, N + 1):
        mask = 1 << (u - 1)
        dp[mask][u] = 0

    # Process all combinations
    for bit_count in range(1, N):
        for comb in itertools.combinations(range(1, N + 1), bit_count):
            nodes_in_comb = set(comb)
            mask = 0
            for node in comb:
                mask |= 1 << (node - 1)
            # For each u in comb
            for u in comb:
                if dp[mask][u] == INF:
                    continue
                # Try to add all v not in comb
                for v in range(1, N + 1):
                    if v in nodes_in_comb:
                        continue
                    if dist[u][v] == INF:
                        continue
                    new_mask = mask | (1 << (v - 1))
                    if dp[new_mask][v] > dp[mask][u] + dist[u][v]:
                        dp[new_mask][v] = dp[mask][u] + dist[u][v]

    # Check the full mask
    full_mask = (1 << N) - 1
    res = INF
    for u in range(1, N + 1):
        if dp[full_mask][u] < res:
            res = dp[full_mask][u]

    if res == INF:
        print("No")
    else:
        print(res)

if __name__ == "__main__":
    main()