from collections import defaultdict

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.size = [1] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        xr = self.find(x)
        yr = self.find(y)
        if xr != yr:
            if self.rank[xr] < self.rank[yr]:
                xr, yr = yr, xr
            if self.rank[xr] == self.rank[yr]:
                self.rank[xr] += 1
            self.parent[yr] = xr
            self.size[xr] += self.size[yr]

    def get_size(self, x):
        return self.size[self.find(x)]

def solve():
    n, q = map(int, input().split())
    uf = UnionFind(n)
    components = defaultdict(int)
    isolated = n
    for _ in range(q):
        query = list(map(int, input().split()))
        if query[0] == 1:
            u, v = query[1] - 1, query[2] - 1
            if uf.find(u) != uf.find(v):
                isolated -= min(uf.get_size(u), uf.get_size(v)) * 2
            uf.union(u, v)
        else:
            v = query[1] - 1
            size = uf.get_size(v)
            isolated += size * (size - 1)
            components[uf.find(v)] = size
        print(isolated)

solve()