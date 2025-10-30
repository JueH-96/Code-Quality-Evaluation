import sys
import heapq

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        xr = self.find(x)
        yr = self.find(y)
        if self.rank[xr] < self.rank[yr]:
            self.parent[xr] = yr
        elif self.rank[xr] > self.rank[yr]:
            self.parent[yr] = xr
        else:
            self.parent[yr] = xr
            self.rank[xr] += 1

def solve():
    n, m = map(int, sys.stdin.readline().split())
    uf = UnionFind(n)
    edges = []
    for _ in range(m):
        k, c = map(int, sys.stdin.readline().split())
        a = list(map(int, sys.stdin.readline().split()))
        for i in range(k):
            for j in range(i+1, k):
                edges.append((c, a[i]-1, a[j]-1))
    edges.sort()
    total_weight = 0
    for c, u, v in edges:
        if uf.find(u) != uf.find(v):
            uf.union(u, v)
            total_weight += c
    if len(set(uf.find(i) for i in range(n))) > 1:
        print(-1)
    else:
        print(total_weight)

solve()