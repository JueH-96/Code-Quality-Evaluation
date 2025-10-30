from collections import defaultdict
import heapq

n, m = map(int, input().split())

# Store all edges
adj = [[] for _ in range(n + 1)]

for i in range(m):
    u, v, t = map(int, input().split())
    adj[u].append((v, t, i))
    adj[v].append((u, t, i))

q = int(input())

for _ in range(q):
    k = int(input())
    required = list(map(int, input().split()))
    required_map = {req - 1: idx for idx, req in enumerate(required)}  # bridge_id (0-indexed) -> bit position
    
    # DP with Dijkstra
    INF = float('inf')
    dist = defaultdict(lambda: defaultdict(lambda: INF))
    dist[0][1] = 0
    
    pq = [(0, 0, 1)]  # (distance, mask, vertex)
    
    while pq:
        d, mask, u = heapq.heappop(pq)
        
        if d > dist[mask][u]:
            continue
        
        if mask == (1 << k) - 1 and u == n:
            print(d)
            break
        
        for v, w, edge_id in adj[u]:
            if edge_id in required_map:
                # This is a required edge
                bit_pos = required_map[edge_id]
                new_mask = mask | (1 << bit_pos)
            else:
                # Not a required edge
                new_mask = mask
            
            if d + w < dist[new_mask][v]:
                dist[new_mask][v] = d + w
                heapq.heappush(pq, (d + w, new_mask, v))