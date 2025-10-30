# YOUR CODE HERE
from collections import defaultdict
import sys
from itertools import permutations

input = sys.stdin.read
data = input().split()
N = int(data[0])
M = int(data[1])
edges = []
for i in range(2, 2*M+2, 3):
    u = int(data[i])
    v = int(data[i+1])
    w = int(data[i+2])
    edges.append((u, v, w))

graph = defaultdict(list)
for u, v, w in edges:
    graph[u].append((v, w))

def dfs(node, visited, path, weight):
    visited[node] = True
    path.append(node)
    weight += sum([e[1] for e in graph[node]])
    for neighbor, w in graph[node]:
        if not visited[neighbor]:
            dfs(neighbor, visited, path, weight)
    visited[node] = False

def find_path():
    for start in range(1, N+1):
        visited = [False] * (N+1)
        path = []
        weight = 0
        dfs(start, visited, path, weight)
        if all(visited[1:]):
            return path, weight
    return None, None

path, weight = find_path()
if path is None:
    print("No")
else:
    print(weight)