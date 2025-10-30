import heapq
import sys

def solve():
    N, M = map(int, input().split())
    A = list(map(int, input().split()))
    
    # Build adjacency list
    graph = [[] for _ in range(N + 1)]
    
    for _ in range(M):
        u, v, b = map(int, input().split())
        graph[u].append((v, b))
        graph[v].append((u, b))
    
    # Dijkstra's algorithm
    dist = [float('inf')] * (N + 1)
    dist[1] = A[0]  # A[0] is weight of vertex 1 (0-indexed)
    
    pq = [(A[0], 1)]  # (distance, vertex)
    
    while pq:
        d, u = heapq.heappop(pq)
        
        if d > dist[u]:
            continue
            
        for v, edge_weight in graph[u]:
            # Cost to reach v from u = current distance to u + edge weight + vertex weight of v
            new_dist = dist[u] + edge_weight + A[v - 1]  # A[v-1] because A is 0-indexed
            
            if new_dist < dist[v]:
                dist[v] = new_dist
                heapq.heappush(pq, (new_dist, v))
    
    # Output distances to vertices 2, 3, ..., N
    result = []
    for i in range(2, N + 1):
        result.append(str(dist[i]))
    
    print(' '.join(result))

solve()