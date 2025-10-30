import sys
from collections import defaultdict

class UnionFind:
    def __init__(self, n):
        self.n = n
        self.parents = [-1] * n
        self.size = [1] * n

    def find(self, x):
        if self.parents[x] < 0:
            return x
        else:
            self.parents[x] = self.find(self.parents[x])
            return self.parents[x]

    def union(self, x, y):
        x = self.find(x)
        y = self.find(y)

        if x == y:
            return

        if self.parents[x] > self.parents[y]:
            x, y = y, x

        self.parents[x] += self.parents[y]
        self.parents[y] = x
        self.size[x] += self.size[y]

    def size(self, x):
        return -self.parents[self.find(x)]

    def same(self, x, y):
        return self.find(x) == self.find(y)

    def members(self, x):
        root = self.find(x)
        return [i for i in range(self.n) if self.find(i) == root]

    def roots(self):
        return [i for i, x in enumerate(self.parents) if x < 0]

    def group_count(self):
        return len(self.roots())

    def all_group_members(self):
        return {r: self.members(r) for r in self.roots()}

    def __str__(self):
        return '
'.join('{}: {}'.format(r, self.members(r)) for r in self.roots())

N, Q = map(int, sys.stdin.readline().split())
uf = UnionFind(N)
isolated = set(range(N))
for _ in range(Q):
    query = list(map(int, sys.stdin.readline().split()))
    if query[0] == 1:
        u, v = query[1]-1, query[2]-1
        if u in isolated:
            isolated.remove(u)
        if v in isolated:
            isolated.remove(v)
        uf.union(u, v)
    else:
        v = query[1]-1
        if uf.find(v) in isolated:
            isolated.remove(uf.find(v))
        isolated.add(v)
    print(len(isolated))