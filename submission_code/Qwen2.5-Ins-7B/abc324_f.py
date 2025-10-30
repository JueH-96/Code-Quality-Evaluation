# YOUR CODE HERE
from collections import defaultdict
import heapq

N, M = map(int, input().split())
edges = defaultdict(list)
for _ in range(M):
    u, v, b, c = map(int, input().split())
    edges[u].append((-b/c, v))

pq = [(-0, 1)]
max_beauty_cost_ratio = -float('inf')
visited = set()

while pq:
    ratio, node = heapq.heappop(pq)
    if node == N:
        max_beauty_cost_ratio = -ratio
        break
    if node in visited:
        continue
    visited.add(node)
    for ratio, next_node in edges[node]:
        heapq.heappush(pq, (ratio, next_node))

print(max_beauty_cost_ratio)