from collections import defaultdict
from decimal import Decimal, getcontext

getcontext().prec = 10

class UnionFind:
    def __init__(self, n):
        self.n = n
        self.parents = [-1] * n

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

def solve():
    N, M = map(int, input().split())
    uf = UnionFind(N)
    edges = []
    for _ in range(M):
        A, B, X, Y = map(int, input().split())
        A -= 1
        B -= 1
        edges.append((A, B, X, Y))
        uf.union(A, B)

    groups = uf.all_group_members()
    group_info = {}
    for group in groups.values():
        group_info[group[0]] = [Decimal(0), Decimal(0)]
        for A, B, X, Y in edges:
            if A in group and B in group:
                group_info[group[0]][0] += Decimal(X)
                group_info[group[0]][1] += Decimal(Y)

    for i in range(N):
        if i in group_info:
            x, y = group_info[i]
            if x == Decimal(0) and y == Decimal(0):
                print("undecidable")
            else:
                print(f"{x} {y}")
        else:
            print("undecidable")

solve()