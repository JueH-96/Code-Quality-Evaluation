# YOUR CODE HERE
import heapq
import sys

input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
A = list(map(int, data[2:N+2]))

edges = []
index = N + 2
for _ in range(M):
    U = int(data[index]) - 1
    V = int(data[index + 1]) - 1
    B = int(data[index + 2])
    edges.append((U, V, B))
    index += 3

# Create adjacency list
graph = [[] for _ in range(N)]
for U, V, B in edges:
    graph[U].append((V, B))
    graph[V].append((U, B))

# Dijkstra's algorithm to find the shortest path from vertex 1 to all other vertices
def dijkstra(start):
    dist = [float('inf')] * N
    dist[start] = A[start]
    pq = [(A[start], start)]
    heapq.heapify(pq)
    
    while pq:
        current_dist, u = heapq.heappop(pq)
        
        if current_dist > dist[u]:
            continue
        
        for v, weight in graph[u]:
            path_weight = current_dist + weight + A[v]
            if path_weight < dist[v]:
                dist[v] = path_weight
                heapq.heappush(pq, (path_weight, v))
    
    return dist

# Get the shortest paths from vertex 1 (index 0)
distances = dijkstra(0)

# Print the results for vertices 2 to N
print(" ".join(map(str, distances[1:])))