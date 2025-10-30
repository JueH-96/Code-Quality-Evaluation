import sys
import heapq

input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])

edges = []
index = 2
for _ in range(M):
    u = int(data[index]) - 1
    v = int(data[index + 1]) - 1
    b = int(data[index + 2])
    c = int(data[index + 3])
    edges.append((u, v, b, c))
    index += 4

def can_achieve_ratio(x):
    # Create a graph with modified weights based on x
    adj = [[] for _ in range(N)]
    for u, v, b, c in edges:
        modified_weight = b - x * c
        adj[u].append((v, modified_weight))
    
    # Bellman-Ford like approach but using max-heap for maximum cost path
    # We use a max-heap to always extend the most promising path
    dist = [-float('inf')] * N
    dist[0] = 0
    pq = [(-dist[0], 0)]  # (negative of distance, node)
    
    while pq:
        current_dist, u = heapq.heappop(pq)
        current_dist = -current_dist
        
        if dist[u] < current_dist:
            continue
        
        for v, weight in adj[u]:
            if dist[v] < dist[u] + weight:
                dist[v] = dist[u] + weight
                heapq.heappush(pq, (-dist[v], v))
    
    # We need to check if we can reach node N-1
    return dist[N-1] > 0

# Binary search for the maximum ratio
low, high = 0, 10000  # since max beauty and cost are 10000, ratio cannot exceed 10000
precision = 1e-9

while high - low > precision:
    mid = (low + high) / 2
    if can_achieve_ratio(mid):
        low = mid
    else:
        high = mid

print(f"{low:.12f}")