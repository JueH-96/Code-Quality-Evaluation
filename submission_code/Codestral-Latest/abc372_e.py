import sys
from collections import defaultdict

input = sys.stdin.read
data = input().split()

index = 0
N = int(data[index])
Q = int(data[index + 1])
index += 2

graph = defaultdict(set)
results = []

for _ in range(Q):
    query_type = int(data[index])
    if query_type == 1:
        u = int(data[index + 1])
        v = int(data[index + 2])
        graph[u].add(v)
        graph[v].add(u)
        index += 3
    else:
        v = int(data[index + 1])
        k = int(data[index + 2])
        connected_vertices = sorted(graph[v], reverse=True)
        if len(connected_vertices) >= k:
            results.append(connected_vertices[k - 1])
        else:
            results.append(-1)
        index += 3

for result in results:
    print(result)