import heapq

# Read input
N, M = map(int, input().split())
A = list(map(int, input().split()))
edges = [tuple(map(int, input().split())) for _ in range(M)]

# Create adjacency list
graph = [[] for _ in range(N + 1)]
for u, v, b in edges:
    graph[u].append((v, b))
    graph[v].append((u, b))

# Dijkstra's algorithm to find the minimum weight path from vertex 1 to all other vertices
def dijkstra(start):
    dist = [float('inf')] * (N + 1)
    dist[start] = A[start - 1]
    pq = [(A[start - 1], start)]

    while pq:
        current_dist, u = heapq.heappop(pq)

        if current_dist > dist[u]:
            continue

        for v, b in graph[u]:
            weight = current_dist + b + A[v - 1]
            if weight < dist[v]:
                dist[v] = weight
                heapq.heappush(pq, (weight, v))

    return dist

# Calculate the minimum weight paths from vertex 1 to all other vertices
distances = dijkstra(1)

# Print the results
print(*distances[2:])