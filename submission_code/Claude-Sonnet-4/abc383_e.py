import sys
from collections import defaultdict
import heapq
from itertools import permutations

def solve():
    N, M, K = map(int, input().split())
    
    # Build adjacency list
    graph = defaultdict(list)
    for _ in range(M):
        u, v, w = map(int, input().split())
        graph[u].append((v, w))
        graph[v].append((u, w))
    
    A = list(map(int, input().split()))
    B = list(map(int, input().split()))
    
    # Function to find minimum bottleneck path from source to all vertices
    def min_bottleneck_paths(source):
        dist = [float('inf')] * (N + 1)
        dist[source] = 0
        pq = [(0, source)]
        
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue
                
            for v, w in graph[u]:
                new_dist = max(dist[u], w)
                if new_dist < dist[v]:
                    dist[v] = new_dist
                    heapq.heappush(pq, (new_dist, v))
        
        return dist
    
    # Compute f(a, b) for all required pairs
    cost_matrix = []
    
    for i in range(K):
        dist = min_bottleneck_paths(A[i])
        row = []
        for j in range(K):
            row.append(dist[B[j]])
        cost_matrix.append(row)
    
    # Solve assignment problem - try all permutations since K is small
    min_cost = float('inf')
    for perm in permutations(range(K)):
        cost = sum(cost_matrix[i][perm[i]] for i in range(K))
        min_cost = min(min_cost, cost)
    
    print(min_cost)

solve()