import sys
from collections import defaultdict
from heapq import heappush, heappop

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

def kruskal(edges, n):
    parent = [i for i in range(n+1)]
    rank = [0] * (n+1)
    total_weight = 0
    for edge in edges:
        u, v, w = edge
        if find(parent, u) != find(parent, v):
            union(parent, rank, u, v)
            total_weight += w
    if len(set(find(parent, i) for i in range(1, n+1))) > 1:
        return -1
    return total_weight

def solve():
    N, M = map(int, input().split())
    edges = []
    for _ in range(M):
        K, C = map(int, input().split())
        vertices = list(map(int, input().split()))
        for i in range(K):
            for j in range(i+1, K):
                edges.append((vertices[i], vertices[j], C))
    return kruskal(edges, N)

print(solve())