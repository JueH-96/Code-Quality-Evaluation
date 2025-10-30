# YOUR CODE HERE
from collections import defaultdict

def find(x, parent):
    if parent[x] == x:
        return x
    parent[x] = find(parent[x], parent)
    return parent[x]

def union(x, y, parent, rank):
    xroot = find(x, parent)
    yroot = find(y, parent)
    if rank[xroot] < rank[yroot]:
        parent[xroot] = yroot
    elif rank[xroot] > rank[yroot]:
        parent[yroot] = xroot
    else:
        parent[yroot] = xroot
        rank[xroot] += 1

def is_good_graph(n, edges, k, pairs):
    parent = list(range(n + 1))
    rank = [0] * (n + 1)
    for u, v in edges:
        union(u, v, parent, rank)
    for x, y in pairs:
        if find(x, parent) == find(y, parent):
            return "No"
    return "Yes"

n, m = map(int, input().split())
edges = [tuple(map(int, input().split())) for _ in range(m)]
k = int(input())
pairs = [tuple(map(int, input().split())) for _ in range(k)]
q = int(input())
additional_pairs = [tuple(map(int, input().split())) for _ in range(q)]

for p, q in additional_pairs:
    edges.append((p, q))
    print(is_good_graph(n, edges, k, pairs))
    edges.pop()