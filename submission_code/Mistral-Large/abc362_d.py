import sys
import heapq

input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
A = list(map(int, data[2:2+N]))
edges = []
for i in range(M):
    U = int(data[2+N+3*i])
    V = int(data[2+N+3*i+1])
    B = int(data[2+N+3*i+2])
    edges.append((U, V, B))
    edges.append((V, U, B))  # Since the graph is undirected

# Create adjacency list
adj = [[] for _ in range(N+1)]
for (U, V, B) in edges:
    adj[U].append((V, B))

# Dijkstra's algorithm
def dijkstra(start):
    dist = [float('inf')] * (N+1)
    dist[start] = A[start-1]
    priority_queue = [(A[start-1], start)]

    while priority_queue:
        current_dist, u = heapq.heappop(priority_queue)

        if current_dist > dist[u]:
            continue

        for v, weight in adj[u]:
            distance = current_dist + A[v-1] + weight

            if distance < dist[v]:
                dist[v] = distance
                heapq.heappush(priority_queue, (distance, v))

    return dist

# Find the minimum weight from vertex 1 to all other vertices
distances = dijkstra(1)

# Print the results for vertices 2 to N
print(' '.join(map(str, distances[2:])))