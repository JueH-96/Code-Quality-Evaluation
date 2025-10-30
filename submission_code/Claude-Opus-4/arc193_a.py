import heapq
from collections import defaultdict

# Read input
N = int(input())
W = list(map(int, input().split()))

intervals = []
for _ in range(N):
    L, R = map(int, input().split())
    intervals.append((L, R))

# Build adjacency list
adj = defaultdict(list)
for i in range(N):
    for j in range(i + 1, N):
        L_i, R_i = intervals[i]
        L_j, R_j = intervals[j]
        # Check if intervals don't overlap
        if R_i < L_j or R_j < L_i:
            adj[i].append(j)
            adj[j].append(i)

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
            
        for v in adj[u]:
            new_dist = dist[u] + W[v]
            if new_dist < dist[v]:
                dist[v] = new_dist
                heapq.heappush(pq, (new_dist, v))
    
    if dist[t] == float('inf'):
        print(-1)
    else:
        print(dist[t])