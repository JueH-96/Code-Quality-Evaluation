# YOUR CODE HERE
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

N, M = map(int, input().split())
edges = [tuple(map(int, input().split())) for _ in range(M)]

K = int(input())
forbidden = [tuple(map(int, input().split())) for _ in range(K)]

Q = int(input())
queries = [tuple(map(int, input().split())) for _ in range(Q)]

parent = list(range(N + 1))
rank = [0] * (N + 1)

forbidden_set = set(forbidden)
for u, v in edges:
    if (u, v) not in forbidden_set and (v, u) not in forbidden_set:
        union(parent, rank, u, v)

for p, q in queries:
    if (p, q) in forbidden_set or (q, p) in forbidden_set:
        print("No")
    elif find(parent, p) == find(parent, q):
        print("No")
    else:
        print("Yes")