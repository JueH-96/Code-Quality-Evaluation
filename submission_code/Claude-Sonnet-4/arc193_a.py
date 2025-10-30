import heapq
from collections import defaultdict
import sys

def intervals_dont_intersect(l1, r1, l2, r2):
    return r1 < l2 or r2 < l1

def dijkstra(graph, start, end, weights):
    # Priority queue: (distance, vertex)
    pq = [(weights[start], start)]
    distances = {}
    
    while pq:
        dist, u = heapq.heappop(pq)
        
        if u in distances:
            continue
            
        distances[u] = dist
        
        if u == end:
            return dist
            
        for v in graph[u]:
            if v not in distances:
                heapq.heappush(pq, (dist + weights[v], v))
    
    return -1

# Read input
N = int(input())
weights = list(map(int, input().split()))

intervals = []
for _ in range(N):
    l, r = map(int, input().split())
    intervals.append((l, r))

# Build graph
graph = defaultdict(list)
for i in range(N):
    for j in range(i + 1, N):
        l1, r1 = intervals[i]
        l2, r2 = intervals[j]
        if intervals_dont_intersect(l1, r1, l2, r2):
            graph[i].append(j)
            graph[j].append(i)

Q = int(input())
for _ in range(Q):
    s, t = map(int, input().split())
    s -= 1  # Convert to 0-indexed
    t -= 1  # Convert to 0-indexed
    
    result = dijkstra(graph, s, t, weights)
    print(result)