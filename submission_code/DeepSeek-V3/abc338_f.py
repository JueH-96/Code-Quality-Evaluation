import sys
import heapq

def main():
    N, M = map(int, sys.stdin.readline().split())
    edges = [[] for _ in range(N+1)]
    for _ in range(M):
        U, V, W = map(int, sys.stdin.readline().split())
        edges[U].append((V, W))
    
    # Precompute shortest paths between all pairs using Floyd-Warshall
    INF = float('inf')
    dist = [[INF] * (N+1) for _ in range(N+1)]
    for i in range(1, N+1):
        dist[i][i] = 0
    for u in range(1, N+1):
        for v, w in edges[u]:
            if dist[u][v] > w:
                dist[u][v] = w
    for k in range(1, N+1):
        for i in range(1, N+1):
            for j in range(1, N+1):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
    
    # Now, we need to find a walk that visits all vertices at least once
    # This is equivalent to finding a path that covers all vertices, possibly with revisits
    # We can model this as a Traveling Salesman Problem (TSP) with revisits allowed
    # Since the graph is strongly connected (as per the problem statement), we can find such a path
    
    # To find the minimum weight, we can use dynamic programming
    # dp[mask][u] represents the minimum cost to visit all vertices in mask, ending at u
    # mask is a bitmask representing the visited vertices
    # Initialize dp[1 << (u-1)][u] = 0 for all u
    # Then, for each mask, for each u in mask, for each v not in mask, update dp[mask | (1 << (v-1))][v] = min(dp[mask | (1 << (v-1))][v], dp[mask][u] + dist[u][v])
    # Finally, the answer is the minimum dp[(1 << N) - 1][u] for all u
    
    dp = [[INF] * (N+1) for _ in range(1 << N)]
    for u in range(1, N+1):
        dp[1 << (u-1)][u] = 0
    
    for mask in range(1 << N):
        for u in range(1, N+1):
            if not (mask & (1 << (u-1))):
                continue
            for v in range(1, N+1):
                if mask & (1 << (v-1)):
                    continue
                new_mask = mask | (1 << (v-1))
                if dp[new_mask][v] > dp[mask][u] + dist[u][v]:
                    dp[new_mask][v] = dp[mask][u] + dist[u][v]
    
    min_cost = INF
    for u in range(1, N+1):
        if dp[(1 << N) - 1][u] < min_cost:
            min_cost = dp[(1 << N) - 1][u]
    
    if min_cost == INF:
        print("No")
    else:
        print(min_cost)

if __name__ == "__main__":
    main()