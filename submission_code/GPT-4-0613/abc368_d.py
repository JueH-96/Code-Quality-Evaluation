import sys
from heapq import *
input = sys.stdin.readline

def add_edge(u, v):
    G[u].append(v)
    G[v].append(u)

def dfs(v, p):
    for u in G[v]:
        if u == p: continue
        depth[u] = depth[v] + 1
        dfs(u, v)

def lca(u, v):
    if depth[u] > depth[v]: u, v = v, u
    for i in range(30, -1, -1):
        if ((depth[v] - depth[u]) >> i) & 1:
            v = parent[i][v]
    if u == v: return u
    for i in range(30, -1, -1):
        if parent[i][u] != parent[i][v]:
            u = parent[i][u]
            v = parent[i][v]
    return parent[0][u]

N, K = map(int, input().split())
G = [[] for _ in range(N)]
for _ in range(N - 1):
    u, v = map(int, input().split())
    u -= 1
    v -= 1
    add_edge(u, v)
V = list(map(int, input().split()))
for i in range(K): V[i] -= 1

depth = [0] * N
parent = [[0] * N for _ in range(31)]
dfs(0, -1)
for i in range(30):
    for v in range(N):
        parent[i + 1][v] = parent[i][parent[i][v]]

V.sort(key = lambda v: depth[v])
dist = [0] * K
for i in range(K - 1):
    u = V[i]
    v = V[i + 1]
    c = lca(u, v)
    dist[i + 1] = dist[i] + depth[u] + depth[v] - 2 * depth[c]

print(dist[-1] - K + 3)