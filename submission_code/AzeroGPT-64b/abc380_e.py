import sys
from collections import defaultdict

class UnionFind:
    def __init__(self, n):
        self.parent = [-1] * n
        self.rank = [0] * n
        self.colors = [i for i in range(n)]
        self.color_counts = [1] * n

    def find(self, x):
        if self.parent[x] < 0:
            return x
        else:
            self.parent[x] = self.find(self.parent[x])
            return self.parent[x]

    def union(self, x, y, color):
        rx = self.find(x)
        ry = self.find(y)
        if rx != ry:
            if self.rank[rx] < self.rank[ry]:
                rx, ry = ry, rx
            self.parent[rx] += self.parent[ry]
            self.parent[ry] = rx
            if self.colors[rx] != color:
                self.colors[ry] = self.colors[rx] = color
                self.color_counts[color] += self.color_counts[self.colors[ry]]
                self.color_counts[self.colors[ry]] = 0
            
    def count(self, x):
        return self.color_counts[self.colors[self.find(x)]]

N, Q = map(int, input().split())
uf = UnionFind(N)
for i in range(Q):
    query = list(map(int, input().split()))
    if query[0] == 1:
        x, c = query[1] - 1, query[2] - 1
        if 0 <= x - 1 < N and uf.colors[x - 1] == uf.colors[x]:
            uf.union(x, x - 1, c)
        if 0 <= x + 1 < N and uf.colors[x + 1] == uf.colors[x]:
            uf.union(x + 1, x, c)
    else:
        print(uf.count(query[1] - 1))