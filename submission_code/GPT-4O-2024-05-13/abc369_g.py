# YOUR CODE HERE
import sys
import heapq
from collections import defaultdict, deque

input = sys.stdin.read
data = input().split()

N = int(data[0])
edges = []
graph = defaultdict(list)

index = 1
for i in range(N-1):
    U = int(data[index])
    V = int(data[index+1])
    L = int(data[index+2])
    edges.append((U, V, L))
    graph[U].append((V, L))
    graph[V].append((U, L))
    index += 3

def dijkstra(start):
    dist = {i: float('inf') for i in range(1, N+1)}
    dist[start] = 0
    pq = [(0, start)]
    while pq:
        current_dist, u = heapq.heappop(pq)
        if current_dist > dist[u]:
            continue
        for v, length in graph[u]:
            distance = current_dist + length
            if distance < dist[v]:
                dist[v] = distance
                heapq.heappush(pq, (distance, v))
    return dist

dist_from_1 = dijkstra(1)
farthest_node = max(dist_from_1, key=dist_from_1.get)
dist_from_farthest = dijkstra(farthest_node)
other_farthest_node = max(dist_from_farthest, key=dist_from_farthest.get)
diameter = dist_from_farthest[other_farthest_node]

total_edge_length = sum(L for _, _, L in edges) * 2

results = [0] * N
for K in range(1, N+1):
    if K == 1:
        results[K-1] = dist_from_1[farthest_node] * 2
    elif K == 2:
        results[K-1] = total_edge_length - diameter
    else:
        results[K-1] = total_edge_length

for result in results:
    print(result)