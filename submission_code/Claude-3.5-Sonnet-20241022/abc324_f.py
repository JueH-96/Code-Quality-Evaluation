from collections import defaultdict
import heapq

def can_reach_target(mid, graph, N):
    # Returns True if we can find a path from 1 to N with ratio >= mid
    # Using Dijkstra's with modified weights: b - mid*c
    dist = [float('-inf')] * (N+1)
    dist[1] = 0
    pq = [(-0, 1)]  # (-value, vertex)
    
    while pq:
        val, u = heapq.heappop(pq)
        val = -val
        
        if val < dist[u]:
            continue
            
        for v, b, c in graph[u]:
            new_val = val + (b - mid*c)
            if new_val > dist[v]:
                dist[v] = new_val
                heapq.heappush(pq, (-new_val, v))
    
    return dist[N] >= 0

def solve():
    N, M = map(int, input().split())
    
    # Build adjacency list
    graph = defaultdict(list)
    for _ in range(M):
        u, v, b, c = map(int, input().split())
        graph[u].append((v, b, c))
    
    # Binary search on the ratio
    left = 0
    right = 10**4  # max possible ratio
    
    # Run binary search 100 times for sufficient precision
    for _ in range(100):
        mid = (left + right) / 2
        if can_reach_target(mid, graph, N):
            left = mid
        else:
            right = mid
    
    print(left)

solve()