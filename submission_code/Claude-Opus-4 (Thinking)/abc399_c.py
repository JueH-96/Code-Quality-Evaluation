class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n + 1))
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        self.parent[self.find(x)] = self.find(y)

# Read input
n, m = map(int, input().split())
uf = UnionFind(n)

for _ in range(m):
    u, v = map(int, input().split())
    uf.union(u, v)

# Count the number of connected components
components = sum(1 for i in range(1, n + 1) if uf.find(i) == i)

# Calculate minimum edges to delete
print(m - (n - components))