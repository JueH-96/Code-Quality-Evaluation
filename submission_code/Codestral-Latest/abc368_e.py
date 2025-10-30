import heapq

# Read input
N, M, X_1 = map(int, input().split())
trains = [tuple(map(int, input().split())) for _ in range(M)]

# Create a graph where each node represents a city
# and each edge represents a train with weight T_i - S_i
graph = [[] for _ in range(N + 1)]
for A, B, S, T in trains:
    graph[A].append((B, T - S))

# Initialize the delay for each train
delay = [0] * M

# Use Dijkstra's algorithm to find the minimum delay for each train
def dijkstra(start):
    pq = [(0, start)]
    dist = [float('inf')] * (N + 1)
    dist[start] = 0

    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, weight in graph[u]:
            if dist[u] + weight < dist[v]:
                dist[v] = dist[u] + weight
                heapq.heappush(pq, (dist[v], v))

    return dist

# Calculate the minimum delay for each train
for i in range(M):
    A, B, S, T = trains[i]
    dist = dijkstra(A)
    if dist[B] != float('inf'):
        delay[i] = max(0, dist[B] - (T - S))

# Print the delays
print(' '.join(map(str, delay)))