import heapq

def dijkstra(n, adj, excluded_edges=set()):
    dist = [float('inf')] * (n + 1)
    dist[1] = 0
    pq = [(0, 1)]
    
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        
        for v, w, edge_id in adj[u]:
            if edge_id not in excluded_edges and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                heapq.heappush(pq, (dist[v], v))
    
    return dist[n]

# Read input
n, m = map(int, input().split())
adj = [[] for _ in range(n + 1)]

for i in range(m):
    a, b, c = map(int, input().split())
    adj[a].append((b, c, i))
    adj[b].append((a, c, i))

# Calculate shortest distance with all roads
dist_with_all = dijkstra(n, adj)

# For each road
for i in range(m):
    dist_without_i = dijkstra(n, adj, {i})
    
    if dist_with_all != dist_without_i:
        print("Yes")
    else:
        print("No")