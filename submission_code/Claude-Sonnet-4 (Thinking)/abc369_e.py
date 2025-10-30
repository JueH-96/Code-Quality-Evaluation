from collections import defaultdict
import heapq

N, M = map(int, input().split())

bridges = []
for i in range(M):
    u, v, t = map(int, input().split())
    bridges.append((u-1, v-1, t))  # 0-indexed

# Build adjacency list
graph = defaultdict(list)
for u, v, t in bridges:
    graph[u].append((v, t))
    graph[v].append((u, t))

# Precompute shortest paths between all pairs of islands
def dijkstra(start):
    dist = [float('inf')] * N
    dist[start] = 0
    pq = [(0, start)]
    
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, w in graph[u]:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                heapq.heappush(pq, (dist[v], v))
    
    return dist

all_dists = [dijkstra(i) for i in range(N)]

Q = int(input())

for _ in range(Q):
    line = list(map(int, input().split()))
    K = line[0]
    bridge_indices = [x-1 for x in line[1:]]  # 0-indexed
    
    # DP with bitmask
    dp = [[float('inf')] * N for _ in range(1 << K)]
    dp[0][0] = 0  # Start at island 0
    
    for mask in range(1 << K):
        for island in range(N):
            if dp[mask][island] == float('inf'):
                continue
            
            for i in range(K):
                if mask & (1 << i):
                    continue  # Bridge already used
                
                bridge_idx = bridge_indices[i]
                u, v, t = bridges[bridge_idx]
                new_mask = mask | (1 << i)
                
                # If we're at one end of the bridge, cross it directly
                if island == u:
                    dp[new_mask][v] = min(dp[new_mask][v], dp[mask][island] + t)
                elif island == v:
                    dp[new_mask][u] = min(dp[new_mask][u], dp[mask][island] + t)
                else:
                    # Go to one end, then cross
                    dp[new_mask][v] = min(dp[new_mask][v], dp[mask][island] + all_dists[island][u] + t)
                    dp[new_mask][u] = min(dp[new_mask][u], dp[mask][island] + all_dists[island][v] + t)
    
    # Find minimum cost to reach island N-1
    full_mask = (1 << K) - 1
    result = float('inf')
    for island in range(N):
        if dp[full_mask][island] != float('inf'):
            result = min(result, dp[full_mask][island] + all_dists[island][N-1])
    
    print(result)