# YOUR CODE HERE
import sys

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.size = [1] * n
        self.n = n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        x = self.find(x)
        y = self.find(y)
        if x == y:
            return False
        if self.size[x] < self.size[y]:
            x, y = y, x
        self.parent[y] = x
        self.size[x] += self.size[y]
        self.n -= 1
        return True

def solve():
    N, M, K = map(int, sys.stdin.readline().split())
    edges = []
    for _ in range(M):
        u, v, w = map(int, sys.stdin.readline().split())
        edges.append((w, u - 1, v - 1))
    edges.sort()

    uf = UnionFind(N)
    total_cost = 0
    for w, u, v in edges:
        if uf.find(u) != uf.find(v):
            uf.union(u, v)
            total_cost += w
    print(total_cost % K)

solve()