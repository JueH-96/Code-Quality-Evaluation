import sys
from collections import defaultdict
import heapq

input = sys.stdin.read
data = input().split()

index = 0
N = int(data[index])
index += 1
Q = int(data[index])
index += 1

graph = defaultdict(set)

results = []

for _ in range(Q):
    query_type = int(data[index])
    index += 1
    if query_type == 1:
        u = int(data[index])
        index += 1
        v = int(data[index])
        index += 1
        graph[u].add(v)
        graph[v].add(u)
    elif query_type == 2:
        v = int(data[index])
        index += 1
        k = int(data[index])
        index += 1
        connected_vertices = list(graph[v])
        connected_vertices.sort(reverse=True)
        if len(connected_vertices) < k:
            results.append(-1)
        else:
            results.append(connected_vertices[k-1])

for result in results:
    print(result)