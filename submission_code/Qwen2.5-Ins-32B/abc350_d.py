import sys
from collections import defaultdict

def find(parent, i):
    if parent[i] == i:
        return i
    return find(parent, parent[i])

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

def solve(N, M, edges):
    parent = [i for i in range(N+1)]
    rank = [0] * (N+1)
    degree = defaultdict(int)
    
    for u, v in edges:
        union(parent, rank, u, v)
        degree[u] += 1
        degree[v] += 1
    
    components = defaultdict(int)
    for i in range(1, N+1):
        components[find(parent, i)] += degree[i]
    
    result = 0
    for comp in components.values():
        result += comp // 2
    
    return result - M

input = sys.stdin.read
data = input().split()
N, M = map(int, [data[0], data[1]])
edges = [(int(data[i]), int(data[i+1])) for i in range(2, 2*M+1, 2)]

print(solve(N, M, edges))