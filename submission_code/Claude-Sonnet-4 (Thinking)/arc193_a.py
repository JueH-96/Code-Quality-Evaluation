import heapq

def dijkstra(graph, start, end, weights):
    dist = [float('inf')] * len(weights)
    dist[start] = weights[start]
    pq = [(weights[start], start)]
    visited = set()
    
    while pq:
        d, u = heapq.heappop(pq)
        if u == end:
            return d
        if u in visited:
            continue
        visited.add(u)
        
        for v in graph[u]:
            if v not in visited:
                new_dist = dist[u] + weights[v]
                if new_dist < dist[v]:
                    dist[v] = new_dist
                    heapq.heappush(pq, (new_dist, v))
    
    return -1

# Read input
N = int(input())
weights = list(map(int, input().split()))
intervals = []
for _ in range(N):
    l, r = map(int, input().split())
    intervals.append((l, r))

# Build graph
graph = [[] for _ in range(N)]
for i in range(N):
    for j in range(i + 1, N):
        l1, r1 = intervals[i]
        l2, r2 = intervals[j]
        if r1 < l2 or r2 < l1:  # No overlap
            graph[i].append(j)
            graph[j].append(i)

Q = int(input())
for _ in range(Q):
    s, t = map(int, input().split())
    s -= 1  # Convert to 0-indexed
    t -= 1  # Convert to 0-indexed
    result = dijkstra(graph, s, t, weights)
    print(result)