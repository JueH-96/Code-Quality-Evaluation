import heapq

# Read input
N = int(input())
W = list(map(int, input().split()))

intervals = []
for _ in range(N):
    l, r = map(int, input().split())
    intervals.append((l, r))

# Build graph
graph = [[] for _ in range(N)]
for i in range(N):
    for j in range(i + 1, N):
        # Check if intervals don't intersect
        l1, r1 = intervals[i]
        l2, r2 = intervals[j]
        if r1 < l2 or r2 < l1:
            # Add edge between i and j (0-indexed)
            graph[i].append(j)
            graph[j].append(i)

Q = int(input())

# Process queries
for _ in range(Q):
    s, t = map(int, input().split())
    s -= 1  # Convert to 0-indexed
    t -= 1
    
    # Dijkstra's algorithm
    dist = [float('inf')] * N
    dist[s] = W[s]
    pq = [(W[s], s)]
    
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        
        if u == t:
            break
        
        for v in graph[u]:
            if dist[u] + W[v] < dist[v]:
                dist[v] = dist[u] + W[v]
                heapq.heappush(pq, (dist[v], v))
    
    if dist[t] == float('inf'):
        print(-1)
    else:
        print(dist[t])