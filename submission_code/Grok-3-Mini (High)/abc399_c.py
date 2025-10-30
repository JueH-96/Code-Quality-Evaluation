import sys

class UnionFind:
    def __init__(self, size):
        self.parent = [i for i in range(size)]
        self.rank = [0] * size
        self.num_components = size

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        rootX = self.find(x)
        rootY = self.find(y)
        if rootX != rootY:
            if self.rank[rootX] > self.rank[rootY]:
                self.parent[rootY] = rootX
            elif self.rank[rootX] < self.rank[rootY]:
                self.parent[rootX] = rootY
            else:
                self.parent[rootY] = rootX
                self.rank[rootX] += 1
            self.num_components -= 1

# Read input
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
M = int(data[index])
index += 1

# Initialize Union-Find with size N+1 (1-based indexing)
uf = UnionFind(N + 1)

# Process each edge
for _ in range(M):
    u = int(data[index])
    v = int(data[index + 1])
    uf.union(u, v)
    index += 2

# Get number of components
C = uf.num_components

# Calculate minimum edges to delete
ans = M - N + C

# Output the answer
print(ans)