# YOUR CODE HERE
from collections import defaultdict

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

N, M = map(int, input().split())

parent = list(range(N + 1))
rank = [0] * (N + 1)
graph = defaultdict(set)

for _ in range(M):
    a, b = map(int, input().split())
    union(parent, rank, a, b)
    graph[a].add(b)
    graph[b].add(a)

components = defaultdict(list)
for i in range(1, N + 1):
    components[find(parent, i)].append(i)

total_triangles = 0
for component in components.values():
    size = len(component)
    edges = sum(len(graph[node]) for node in component) // 2
    total_triangles += edges * (size - 2) - sum(len(graph[node]) * (len(graph[node]) - 1) // 2 for node in component)

print(total_triangles)