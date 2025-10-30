import sys

class DisjointSet:
    def __init__(self, n):
        self.parent = list(range(n + 1))
        self.rank = [0] * (n + 1)
        self.size = [1] * (n + 1)

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x != root_y:
            if self.rank[root_x] > self.rank[root_y]:
                self.parent[root_y] = root_x
                self.size[root_x] += self.size[root_y]
            elif self.rank[root_x] < self.rank[root_y]:
                self.parent[root_x] = root_y
                self.size[root_y] += self.size[root_x]
            else:
                self.parent[root_y] = root_x
                self.size[root_x] += self.size[root_y]
                self.rank[root_x] += 1

    def get_size(self, x):
        return self.size[self.find(x)]

def solve():
    n, q = map(int, sys.stdin.readline().split())
    dsu = DisjointSet(n)
    edges = [[] for _ in range(n + 1)]

    for _ in range(q):
        query = list(map(int, sys.stdin.readline().split()))
        if query[0] == 1:
            u, v = query[1], query[2]
            if u not in edges[v]:
                edges[u].append(v)
                edges[v].append(u)
                dsu.union(u, v)
        else:
            v = query[1]
            for u in edges[v]:
                dsu.union(u, v)
                edges[u].remove(v)
            edges[v].clear()

        count = 0
        for i in range(1, n + 1):
            if dsu.get_size(i) == 1:
                count += 1
        print(count)

if __name__ == "__main__":
    solve()