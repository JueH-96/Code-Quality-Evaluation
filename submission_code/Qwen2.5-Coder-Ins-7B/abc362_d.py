import sys
import heapq

input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])

A = list(map(int, data[2:2+N]))
edges = []

index = 2 + N
for _ in range(M):
    U = int(data[index]) - 1
    V = int(data[index + 1]) - 1
    B = int(data[index + 2])
    edges.append((U, V, B))
    index += 3

# Function to find the minimum weight path from vertex 1 to vertex i
def min_path_weight(i):
    # Priority queue to store (weight, current vertex)
    pq = [(A[0], 0)]
    # Distance array to store the minimum weight to reach each vertex
    dist = [float('inf')] * N
    dist[0] = A[0]
    
    while pq:
        current_weight, current_vertex = heapq.heappop(pq)
        
        if current_vertex == i:
            return current_weight
        
        if current_weight > dist[current_vertex]:
            continue
        
        for neighbor, weight in graph[current_vertex]:
            new_weight = current_weight + weight + A[neighbor]
            if new_weight < dist[neighbor]:
                dist[neighbor] = new_weight
                heapq.heappush(pq, (new_weight, neighbor))
    
    return dist[i]

# Build the graph
graph = [[] for _ in range(N)]
for U, V, B in edges:
    graph[U].append((V, B))
    graph[V].append((U, B))

# Find the minimum weight path for each vertex from 2 to N
results = [min_path_weight(i) for i in range(1, N)]

# Print the results
print(' '.join(map(str, results)))