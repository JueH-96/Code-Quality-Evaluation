import sys
import heapq
from collections import defaultdict, deque

def read_input():
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    idx += 1
    M = int(data[idx])
    idx += 1
    edges = []
    for _ in range(M):
        U = int(data[idx])
        idx += 1
        V = int(data[idx])
        idx += 1
        W = int(data[idx])
        idx += 1
        edges.append((U, V, W))
    return N, M, edges

def solve():
    N, M, edges = read_input()
    
    # Graph adjacency list
    graph = defaultdict(list)
    for u, v, w in edges:
        graph[u].append((v, w))
    
    # Check if all nodes are reachable from any node using BFS or DFS
    def is_connected():
        visited = [False] * (N + 1)
        def dfs(node):
            stack = [node]
            count = 0
            while stack:
                n = stack.pop()
                if not visited[n]:
                    visited[n] = True
                    count += 1
                    for neighbor, _ in graph[n]:
                        if not visited[neighbor]:
                            stack.append(neighbor)
            return count == N
        
        # Check if all nodes are reachable from any node
        for i in range(1, N + 1):
            if len(graph[i]) > 0:
                visited = [False] * (N + 1)
                if dfs(i):
                    return True
        return False
    
    if not is_connected():
        print("No")
        return
    
    # Use Floyd-Warshall to find the shortest paths between all pairs of vertices
    inf = float('inf')
    dist = [[inf] * (N + 1) for _ in range(N + 1)]
    for u, v, w in edges:
        dist[u][v] = min(dist[u][v], w)
    
    for i in range(1, N + 1):
        dist[i][i] = 0
    
    for k in range(1, N + 1):
        for i in range(1, N + 1):
            for j in range(1, N + 1):
                if dist[i][k] < inf and dist[k][j] < inf:
                    dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
    
    # Use Held-Karp algorithm to find the minimum weight Hamiltonian path
    # dp[mask][i] represents the minimum cost to visit all vertices in `mask` ending at vertex `i`
    dp = [[inf] * (N + 1) for _ in range(1 << N)]
    for i in range(1, N + 1):
        dp[1 << (i - 1)][i] = 0
    
    for mask in range(1 << N):
        for u in range(1, N + 1):
            if mask & (1 << (u - 1)):
                for v in range(1, N + 1):
                    if not mask & (1 << (v - 1)) and dist[u][v] < inf:
                        new_mask = mask | (1 << (v - 1))
                        dp[new_mask][v] = min(dp[new_mask][v], dp[mask][u] + dist[u][v])
    
    # Find the minimum cost to visit all vertices
    min_cost = inf
    final_mask = (1 << N) - 1
    for i in range(1, N + 1):
        min_cost = min(min_cost, dp[final_mask][i])
    
    if min_cost == inf:
        print("No")
    else:
        print(min_cost)