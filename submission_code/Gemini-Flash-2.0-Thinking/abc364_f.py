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
    n, q = map(int, input().split())
    operations = []
    for _ in range(q):
        l, r, c = map(int, input().split())
        operations.append((l, r, c))

    edges = []
    for i in range(q):
        l, r, c = operations[i]
        new_node = n + i
        for j in range(l, r + 1):
            edges.append((c, new_node, j - 1))

    edges.sort()

    dsu = DSU(n + q)
    mst_cost = 0
    num_edges_mst = 0

    for cost, u, v in edges:
        if dsu.union(u, v):
            mst_cost += cost
            num_edges_mst += 1

    num_components = 0
    for i in range(n + q):
        if dsu.parent[i] == i:
            num_components += 1

    if num_components > 1:
        print(-1)
    else:
        print(mst_cost)

solve()