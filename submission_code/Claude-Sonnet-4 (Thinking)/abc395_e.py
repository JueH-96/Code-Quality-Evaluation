import heapq
from collections import defaultdict

n, m, x = map(int, input().split())

# Build the original graph and reversed graph
original_graph = defaultdict(list)
reversed_graph = defaultdict(list)

for _ in range(m):
    u, v = map(int, input().split())
    original_graph[u].append(v)
    reversed_graph[v].append(u)

# Dijkstra's algorithm
# State: (cost, vertex, graph_state)
# graph_state: 0 = original, 1 = reversed
pq = [(0, 1, 0)]  # Start at vertex 1 with original graph
dist = {}

while pq:
    cost, vertex, graph_state = heapq.heappop(pq)
    
    if (vertex, graph_state) in dist:
        continue
    
    dist[(vertex, graph_state)] = cost
    
    if vertex == n:
        print(cost)
        break
    
    # Move along an edge
    if graph_state == 0:
        graph = original_graph
    else:
        graph = reversed_graph
    
    for neighbor in graph[vertex]:
        if (neighbor, graph_state) not in dist:
            heapq.heappush(pq, (cost + 1, neighbor, graph_state))
    
    # Reverse all edges
    new_graph_state = 1 - graph_state
    if (vertex, new_graph_state) not in dist:
        heapq.heappush(pq, (cost + x, vertex, new_graph_state))