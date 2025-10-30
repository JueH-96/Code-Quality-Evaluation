import sys
from heapq import heappush, heappop

def dijkstra(graph, start):
    distances = [float('inf')] * (len(graph) + 1)
    distances[start] = 0
    pq = [(0, start)]
    
    while pq:
        dist, node = heappop(pq)
        if dist > distances[node]:
            continue
        for neighbor, weight in graph[node]:
            new_dist = max(dist, weight)
            if new_dist < distances[neighbor]:
                distances[neighbor] = new_dist
                heappush(pq, (new_dist, neighbor))
    
    return distances

def min_cost_matching(cost_matrix):
    n = len(cost_matrix)
    u = [0] * n
    v = [0] * n
    p = [-1] * n
    
    for i in range(n):
        p[0] = i
        j0 = 0
        dist = [float('inf')] * n
        used = [False] * n
        
        while p[j0] != -1:
            used[j0] = True
            i0 = p[j0]
            delta = float('inf')
            j1 = -1
            
            for j in range(n):
                if not used[j]:
                    cur = cost_matrix[i0][j] - u[i0] - v[j]
                    if cur < dist[j]:
                        dist[j] = cur
                        p[j] = j0
                    if dist[j] < delta:
                        delta = dist[j]
                        j1 = j
            
            for j in range(n):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    dist[j] -= delta
            
            j0 = j1
        
        while j0 != -1:
            j1 = p[j0]
            p[j0] = p[j1]
            j0 = j1
    
    return -v[0]

# Read input
N, M, K = map(int, input().split())
graph = [[] for _ in range(N + 1)]

for _ in range(M):
    u, v, w = map(int, input().split())
    graph[u].append((v, w))
    graph[v].append((u, w))

A = list(map(int, input().split()))
B = list(map(int, input().split()))

# Compute all-pairs shortest paths
distances = [dijkstra(graph, i) for i in range(1, N + 1)]

# Create cost matrix for matching
cost_matrix = [[distances[a - 1][b] for b in B] for a in A]

# Solve minimum cost matching
result = min_cost_matching(cost_matrix)

# Output result
print(result)