import sys
import heapq
from collections import defaultdict, deque

input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
X = int(data[2])

edges = []
index = 3
for _ in range(M):
    u = int(data[index]) - 1
    v = int(data[index + 1]) - 1
    edges.append((u, v))
    index += 2

def dijkstra(graph, start, n):
    # Distance array
    dist = [float('inf')] * n
    dist[start] = 0
    # Priority queue: (cost, node)
    pq = [(0, start)]
    heapq.heapify(pq)
    
    while pq:
        current_cost, u = heapq.heappop(pq)
        
        if current_cost > dist[u]:
            continue
        
        for v in graph[u]:
            next_cost = current_cost + 1
            if next_cost < dist[v]:
                dist[v] = next_cost
                heapq.heappush(pq, (next_cost, v))
    
    return dist

# Create adjacency list for the graph and its reverse
graph = defaultdict(list)
reverse_graph = defaultdict(list)

for u, v in edges:
    graph[u].append(v)
    reverse_graph[v].append(u)

# Perform Dijkstra from node 0 in both normal and reversed graphs
dist_from_start = dijkstra(graph, 0, N)
dist_to_end = dijkstra(reverse_graph, N-1, N)

# Minimum cost to reach from 1 to N directly
min_cost = dist_from_start[N-1]

# Try reversing the graph at each node and calculate the cost
for node in range(N):
    if dist_from_start[node] < float('inf') and dist_to_end[node] < float('inf'):
        # Cost to reach node + cost to reverse + cost to reach N from node
        cost_with_reverse = dist_from_start[node] + X + dist_to_end[node]
        min_cost = min(min_cost, cost_with_reverse)

print(min_cost)