n, m = map(int, input().split())
edges = []
for _ in range(m):
    a, b = map(int, input().split())
    edges.append((a, b))

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
            return
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1

uf = UnionFind(n + 1)
for a, b in edges:
    uf.union(a, b)

# Count vertices in each component
component_size = {}
for i in range(1, n + 1):
    root = uf.find(i)
    component_size[root] = component_size.get(root, 0) + 1

# Count edges in each component
component_edges = {}
for a, b in edges:
    root = uf.find(a)
    component_edges[root] = component_edges.get(root, 0) + 1

total_new_edges = 0
for root, size in component_size.items():
    current_edges = component_edges.get(root, 0)
    max_edges = size * (size - 1) // 2
    new_edges = max_edges - current_edges
    total_new_edges += new_edges

print(total_new_edges)