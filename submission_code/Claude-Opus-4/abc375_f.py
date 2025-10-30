import heapq
from collections import defaultdict

def dijkstra(n, graph, closed_roads, start, end):
    # Initialize distances
    dist = [float('inf')] * (n + 1)
    dist[start] = 0
    
    # Priority queue: (distance, node)
    pq = [(0, start)]
    
    while pq:
        d, u = heapq.heappop(pq)
        
        if d > dist[u]:
            continue
            
        if u == end:
            return dist[end]
        
        for v, road_id, weight in graph[u]:
            if road_id in closed_roads:
                continue
                
            if dist[u] + weight < dist[v]:
                dist[v] = dist[u] + weight
                heapq.heappush(pq, (dist[v], v))
    
    return -1 if dist[end] == float('inf') else dist[end]

# Read input
N, M, Q = map(int, input().split())

# Build adjacency list
graph = defaultdict(list)
for i in range(1, M + 1):
    a, b, c = map(int, input().split())
    graph[a].append((b, i, c))
    graph[b].append((a, i, c))

# Process queries
closed_roads = set()
for _ in range(Q):
    query = list(map(int, input().split()))
    
    if query[0] == 1:
        # Close road
        road_id = query[1]
        closed_roads.add(road_id)
    else:
        # Find shortest path
        x, y = query[1], query[2]
        result = dijkstra(N, graph, closed_roads, x, y)
        print(result)