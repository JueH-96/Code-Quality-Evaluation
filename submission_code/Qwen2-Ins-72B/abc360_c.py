from collections import defaultdict
import heapq

N = int(input())
A = list(map(int, input().split()))
W = list(map(int, input().split()))

# Create a graph where nodes are boxes and edges are items
graph = defaultdict(list)
for i, (a, w) in enumerate(zip(A, W)):
    graph[a].append((w, i + 1))

# Dijkstra's algorithm to find minimum cost of moving items to their correct boxes
def dijkstra(graph, start):
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    queue = [(0, start)]
    while queue:
        current_distance, current_node = heapq.heappop(queue)
        if current_distance > distances[current_node]:
            continue
        for weight, neighbor in graph[current_node]:
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(queue, (distance, neighbor))
    return distances

# Calculate the minimum cost for each box to contain exactly one item
total_cost = 0
for box in range(1, N + 1):
    if box not in graph:
        continue
    distances = dijkstra(graph, box)
    for i in range(1, N + 1):
        if i in distances and distances[i] < float('inf'):
            total_cost += distances[i]
            graph[i] = []  # Remove the item from the graph after moving it

print(total_cost)