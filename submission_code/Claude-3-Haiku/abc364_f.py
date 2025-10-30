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
    parent = [i for i in range(n + 1)]
    rank = [0] * (n + 1)
    total_cost = 0
    for u, v, w in edges:
        if find(parent, u) != find(parent, v):
            union(parent, rank, u, v)
            total_cost += w
    if len(set(find(parent, i) for i in range(1, n + 1))) == 1:
        return total_cost
    else:
        return -1

def solve():
    n, q = map(int, input().split())
    edges = []
    for _ in range(q):
        l, r, c = map(int, input().split())
        for j in range(l, r + 1):
            edges.append((n + _, j, c))
    print(kruskal(edges, n + q))

solve()