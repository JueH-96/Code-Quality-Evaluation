# YOUR CODE HERE
import sys
from collections import defaultdict

input = sys.stdin.read
data = input().split()

N = int(data[0])
Q = int(data[1])
queries = [(int(data[i*3]), int(data[i*3+1]), int(data[i*3+2])) for i in range(Q)]

graph = defaultdict(set)
isolated = set(range(1, N + 1))

for query in queries:
    if query[0] == 1:
        u, v = query[1], query[2]
        if u in isolated:
            isolated.remove(u)
        if v in isolated:
            isolated.remove(v)
        graph[u].add(v)
        graph[v].add(u)
    else:
        v = query[1]
        for neighbor in graph[v]:
            if len(graph[neighbor]) == 1:
                isolated.add(neighbor)
            graph[neighbor].remove(v)
        graph[v].clear()
        isolated.add(v)
    print(len(isolated))