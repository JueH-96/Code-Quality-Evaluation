# YOUR CODE HERE
import sys
from collections import defaultdict
import heapq

def find(parent, i):
    if parent[i] != i:
        parent[i] = find(parent, parent[i])
    return parent[i]

def union(parent, rank, x, y):
    xroot = find(parent, x)
    yroot = find(parent, y)
    if rank[xroot] < rank[yroot]:
        parent[xroot] = yroot
    elif rank[xroot] > rank[yroot]:
        parent[yroot] = xroot
    else:
        parent[yroot] = xroot
        rank[xroot] += 1

def kruskal(graph, V):
    result = 0
    parent = list(range(V + 1))
    rank = [0] * (V + 1)
    edges = []
    
    for u in graph:
        for v, weight in graph[u]:
            heapq.heappush(edges, (weight, u, v))
    
    while edges:
        weight, u, v = heapq.heappop(edges)
        x = find(parent, u)
        y = find(parent, v)
        if x != y:
            result += weight
            union(parent, rank, x, y)
    
    return result if len(set(find(parent, i) for i in range(1, V + 1))) == 1 else -1

N, M = map(int, input().split())
graph = defaultdict(list)

for _ in range(M):
    line = list(map(int, input().split()))
    K, C = line[0], line[1]
    vertices = line[2:]
    for i in range(K):
        for j in range(i + 1, K):
            u, v = vertices[i], vertices[j]
            graph[u].append((v, C))
            graph[v].append((u, C))

result = kruskal(graph, N)
print(result)