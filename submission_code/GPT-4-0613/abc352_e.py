import sys
from heapq import heappop, heappush

def find(x):
    if par[x] == x:
        return x
    else:
        par[x] = find(par[x])
        return par[x]

def union(x, y):
    x = find(x)
    y = find(y)
    if x == y:
        return
    if rank[x] < rank[y]:
        par[x] = y
    else:
        par[y] = x
        if rank[x] == rank[y]:
            rank[x] += 1

def same(x, y):
    return find(x) == find(y)

def kruskal():
    edges.sort()
    res = 0
    for edge in edges:
        w, u, v = edge
        if not same(u, v):
            res += w
            union(u, v)
    return res

N, M = map(int, sys.stdin.readline().split())
par = [i for i in range(N+1)]
rank = [0 for _ in range(N+1)]
edges = []
for _ in range(M):
    K, C = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    A.sort()
    for i in range(K-1):
        edges.append((C, A[i], A[i+1]))
res = kruskal()
for i in range(1, N):
    if not same(i, i+1):
        print(-1)
        sys.exit()
print(res)