class DSU:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, i):
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            if self.rank[root_i] < self.rank[root_j]:
                self.parent[root_i] = root_j
            elif self.rank[root_i] > self.rank[root_j]:
                self.parent[root_j] = root_i
            else:
                self.parent[root_j] = root_i
                self.rank[root_i] += 1
            return True
        return False

def solve():
    n, m = map(int, input().split())
    edges = []
    for _ in range(m):
        line = list(map(int, input().split()))
        k = line[0]
        c = line[1]
        vertices = line[2:]
        for i in range(k):
            for j in range(i + 1, k):
                edges.append((c, vertices[i] - 1, vertices[j] - 1))

    edges.sort()

    dsu = DSU(n)
    mst_weight = 0
    num_edges = 0
    for weight, u, v in edges:
        if dsu.union(u, v):
            mst_weight += weight
            num_edges += 1

    if num_edges != n - 1:
        print(-1)
    else:
        print(mst_weight)

solve()