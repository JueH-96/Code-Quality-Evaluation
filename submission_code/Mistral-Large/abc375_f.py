import sys
import heapq
from collections import defaultdict, deque

input = sys.stdin.read
data = input().split()

index = 0
N = int(data[index])
index += 1
M = int(data[index])
index += 1
Q = int(data[index])
index += 1

edges = []
for i in range(M):
    A = int(data[index])
    index += 1
    B = int(data[index])
    index += 1
    C = int(data[index])
    index += 1
    edges.append((A, B, C))

queries = []
for i in range(Q):
    type_query = int(data[index])
    index += 1
    if type_query == 1:
        i = int(data[index])
        index += 1
        queries.append((1, i))
    else:
        x = int(data[index])
        index += 1
        y = int(data[index])
        index += 1
        queries.append((2, x, y))

# Initialize the graph
graph = defaultdict(list)
for i, (A, B, C) in enumerate(edges):
    graph[A].append((B, C, i))
    graph[B].append((A, C, i))

# To keep track of closed roads
closed_roads = [False] * M

def dijkstra(start, end):
    pq = [(0, start)]
    distances = {start: 0}
    while pq:
        current_distance, current_vertex = heapq.heappop(pq)
        if current_vertex == end:
            return current_distance
        for neighbor, weight, index in graph[current_vertex]:
            if closed_roads[index]:
                continue
            distance = current_distance + weight
            if neighbor not in distances or distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))
    return -1

results = []
for query in queries:
    if query[0] == 1:
        _, i = query
        closed_roads[i - 1] = True
    else:
        _, x, y = query
        result = dijkstra(x, y)
        results.append(result)

sys.stdout.write('
'.join(map(str, results)) + '
')