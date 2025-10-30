# YOUR CODE HERE
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.size = [1] * n
        self.components = n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px != py:
            if self.size[px] < self.size[py]:
                px, py = py, px
            self.parent[py] = px
            self.size[px] += self.size[py]
            self.components -= 1

    def disconnect(self, x):
        if self.size[x] > 1:
            self.components += self.size[x] - 1
            self.size[x] = 1
            self.parent[x] = x

N, Q = map(int, input().split())
uf = UnionFind(N)
connected = [False] * N

for _ in range(Q):
    query = list(map(int, input().split()))
    if query[0] == 1:
        u, v = query[1] - 1, query[2] - 1
        if not connected[u]:
            uf.components -= 1
            connected[u] = True
        if not connected[v]:
            uf.components -= 1
            connected[v] = True
        uf.union(u, v)
    else:
        v = query[1] - 1
        if connected[v]:
            connected[v] = False
            uf.disconnect(v)
            uf.components += 1
    print(uf.components)