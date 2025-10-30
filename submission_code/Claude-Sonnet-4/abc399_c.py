class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return False
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        return True

n, m = map(int, input().split())
uf = UnionFind(n)

for _ in range(m):
    u, v = map(int, input().split())
    uf.union(u-1, v-1)  # Convert to 0-indexed

# Count connected components
components = len(set(uf.find(i) for i in range(n)))

# Maximum edges in a forest with n vertices and c components is n-c
max_edges_in_forest = n - components
edges_to_delete = m - max_edges_in_forest

print(edges_to_delete)