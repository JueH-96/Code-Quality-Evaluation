import sys
import heapq

input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])

vertex_weights = list(map(int, data[2:N+2]))
edges = data[N+2:]

graph = [[] for _ in range(N)]

# Reading edges
index = 0
for _ in range(M):
    U = int(edges[index]) - 1
    V = int(edges[index + 1]) - 1
    B = int(edges[index + 2])
    graph[U].append((V, B))
    graph[V].append((U, B))
    index += 3

# Dijkstra's algorithm to find the shortest path from vertex 0
def dijkstra(start):
    # Distance from start to each vertex
    dist = [float('inf')] * N
    dist[start] = vertex_weights[start]
    priority_queue = [(vertex_weights[start], start)]
    
    while priority_queue:
        current_dist, u = heapq.heappop(priority_queue)
        
        if current_dist > dist[u]:
            continue
        
        for v, weight in graph[u]:
            path_weight = current_dist + vertex_weights[v] + weight
            if path_weight < dist[v]:
                dist[v] = path_weight
                heapq.heappush(priority_queue, (path_weight, v))
    
    return dist

# Get the minimum path weights from vertex 1 (index 0)
min_path_weights = dijkstra(0)

# Output the results for vertices 2 to N
print(" ".join(map(str, min_path_weights[1:])))