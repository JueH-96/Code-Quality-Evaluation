# YOUR CODE HERE
import sys
from collections import defaultdict

def dfs(node, parent, depth, c, graph):
    dist = [0] * len(c)
    dist[node] = sum(c) - 2 * sum(c[:node]) - 2 * sum(c[node+1:])
    for child in graph[node]:
        if child != parent:
            dist = dfs(child, node, depth + 1, c, graph)
            for i in range(len(c)):
                dist[i] += depth * (c[child] if i == node else -c[child])
    return dist

n = int(input())
graph = defaultdict(list)
c = list(map(int, input().split()))
for _ in range(n-1):
    a, b = map(int, input().split())
    graph[a-1].append(b-1)
    graph[b-1].append(a-1)

dist = dfs(0, -1, 0, c, graph)
print(min(dist))