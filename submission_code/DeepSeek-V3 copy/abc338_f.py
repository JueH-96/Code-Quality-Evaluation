import sys
import itertools

def main():
    N, M = map(int, sys.stdin.readline().split())
    edges = []
    for _ in range(M):
        U, V, W = map(int, sys.stdin.readline().split())
        edges.append((U-1, V-1, W))  # 0-based indexing
    
    # Initialize distance matrix
    INF = float('inf')
    dist = [[INF for _ in range(N)] for _ in range(N)]
    for i in range(N):
        dist[i][i] = 0
    for u, v, w in edges:
        if dist[u][v] > w:
            dist[u][v] = w
    
    # Floyd-Warshall to find shortest paths
    for k in range(N):
        for i in range(N):
            for j in range(N):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
    
    # Check if all vertices are reachable from each other
    for i in range(N):
        for j in range(N):
            if dist[i][j] == INF:
                print("No")
                return
    
    # Now, find the minimum weight walk that visits all vertices
    # This is equivalent to finding the minimum weight Hamiltonian path
    # Since the graph is strongly connected, we can start from any vertex
    # We will use dynamic programming to find the minimum weight walk
    
    # dp[mask][u] represents the minimum weight to reach u with mask
    # mask is a bitmask representing visited vertices
    dp = [[INF for _ in range(N)] for _ in range(1 << N)]
    for u in range(N):
        dp[1 << u][u] = 0
    
    for mask in range(1 << N):
        for u in range(N):
            if not (mask & (1 << u)):
                continue
            for v in range(N):
                if mask & (1 << v):
                    continue
                new_mask = mask | (1 << v)
                if dp[new_mask][v] > dp[mask][u] + dist[u][v]:
                    dp[new_mask][v] = dp[mask][u] + dist[u][v]
    
    # Now, find the minimum weight among all masks that have all bits set
    final_mask = (1 << N) - 1
    min_weight = INF
    for u in range(N):
        if dp[final_mask][u] < min_weight:
            min_weight = dp[final_mask][u]
    
    if min_weight == INF:
        print("No")
    else:
        print(min_weight)

if __name__ == "__main__":
    main()