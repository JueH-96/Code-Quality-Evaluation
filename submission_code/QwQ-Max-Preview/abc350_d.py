class DSU:
    def __init__(self, n):
        self.parent = list(range(n + 1))
        self.size = [1] * (n + 1)
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        x_root = self.find(x)
        y_root = self.find(y)
        if x_root == y_root:
            return
        if self.size[x_root] < self.size[y_root]:
            x_root, y_root = y_root, x_root
        self.parent[y_root] = x_root
        self.size[x_root] += self.size[y_root]

n, m = map(int, input().split())
dsu = DSU(n)
for _ in range(m):
    a, b = map(int, input().split())
    dsu.union(a, b)

roots = set()
total_edges = 0
for i in range(1, n + 1):
    root = dsu.find(i)
    if root not in roots:
        roots.add(root)
        total_edges += dsu.size[root] * (dsu.size[root] - 1) // 2

print(total_edges - m)