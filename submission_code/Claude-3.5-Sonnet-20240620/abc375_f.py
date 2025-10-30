# YOUR CODE HERE
import sys
from heapq import heappush, heappop

def dijkstra(graph, start, end):
    distances = [float('inf')] * (len(graph) + 1)
    distances[start] = 0
    pq = [(0, start)]
    
    while pq:
        current_dist, current_node = heappop(pq)
        
        if current_node == end:
            return current_dist
        
        if current_dist > distances[current_node]:
            continue
        
        for neighbor, weight in graph[current_node].items():
            distance = current_dist + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heappush(pq, (distance, neighbor))
    
    return -1

# Read input
N, M, Q = map(int, input().split())

# Initialize graph
graph = {i: {} for i in range(1, N + 1)}

# Read roads
roads = []
for _ in range(M):
    a, b, c = map(int, input().split())
    graph[a][b] = c
    graph[b][a] = c
    roads.append((a, b, c))

# Process queries
for _ in range(Q):
    query = list(map(int, input().split()))
    
    if query[0] == 1:
        # Close road
        i = query[1] - 1
        a, b, c = roads[i]
        del graph[a][b]
        del graph[b][a]
    else:
        # Find shortest path
        x, y = query[1], query[2]
        result = dijkstra(graph, x, y)
        print(result)