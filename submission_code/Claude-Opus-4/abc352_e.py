# YOUR CODE HERE
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.components = n
    
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
        self.components -= 1
        return True
    
    def is_connected(self):
        return self.components == 1

# Read input
N, M = map(int, input().split())

# Store all edges with their weights
edges = []

for _ in range(M):
    line = list(map(int, input().split()))
    K = line[0]
    C = line[1]
    vertices = line[2:]
    
    # Add edges between all pairs in the subset
    for i in range(K):
        for j in range(i + 1, K):
            u = vertices[i] - 1  # Convert to 0-indexed
            v = vertices[j] - 1
            edges.append((C, u, v))

# Sort edges by weight for Kruskal's algorithm
edges.sort()

# Apply Kruskal's algorithm
uf = UnionFind(N)
total_weight = 0
edges_added = 0

for weight, u, v in edges:
    if uf.union(u, v):
        total_weight += weight
        edges_added += 1
        if edges_added == N - 1:
            break

# Check if graph is connected
if uf.is_connected():
    print(total_weight)
else:
    print(-1)