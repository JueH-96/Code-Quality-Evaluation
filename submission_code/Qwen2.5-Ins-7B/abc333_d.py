# YOUR CODE HERE
import sys

def dfs(node, parent, graph):
    if len(graph[node]) == 1 and node != 1:
        return 1
    return sum(dfs(child, node, graph) for child in graph[node] if child != parent) + 1

N = int(input())
graph = {i: [] for i in range(1, N+1)}
for _ in range(N-1):
    u, v = map(int, input().split())
    graph[u].append(v)
    graph[v].append(u)

print(dfs(1, -1, graph))